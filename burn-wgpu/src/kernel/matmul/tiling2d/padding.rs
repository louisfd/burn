use std::ops::Range;

use burn_tensor::Shape;

use crate::{
    element::WgpuElement,
    kernel::{slice, slice_assign},
    tensor::WgpuTensor,
};

use super::base::empty_from_context;

/// Pads tensor with zeros to make tensor number of rows and columns
/// divisible by some quantity.
/// For instance tensor of shape [1000, 1000] with divisors 64 and 64
/// will be padded to [1024, 1024] with the las 24 elements being zeros
pub(super) fn pad_round<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    row_divisor: usize,
    col_divisor: usize,
) -> WgpuTensor<E, D> {
    let row_modulo = tensor.shape.dims[D - 2] % row_divisor;
    let col_modulo = tensor.shape.dims[D - 1] % col_divisor;
    if row_modulo == 0 && col_modulo == 0 {
        return tensor;
    }
    let mut padded_shape = Vec::with_capacity(D);
    for i in 0..D - 2 {
        padded_shape.push(tensor.shape.dims[i]);
    }
    padded_shape.push(tensor.shape.dims[D - 2] - row_modulo + row_divisor);
    padded_shape.push(tensor.shape.dims[D - 1] - col_modulo + col_divisor);
    padding::<E, D>(tensor, padded_shape.into())
}

/// Crops tensor to specified number of rows and columns.
/// For instance tensor of shape [1024, 1024] with keep_rows and keep_cols 1000
/// will be cropped to [1000, 1000] with the last 24 elements being deleted
// pub(super) fn crop<E: WgpuElement, const D: usize>(
//     tensor: WgpuTensor<E, D>,
//     keep_rows: usize,
//     keep_cols: usize,
// ) -> WgpuTensor<E, D> {
//     if tensor.shape.dims[D - 2] <= keep_rows && tensor.shape.dims[D - 1] <= keep_cols {
//         return tensor;
//     }
//     let mut cropped_shape = Vec::with_capacity(D);
//     for i in 0..D - 2 {
//         cropped_shape.push(tensor.shape.dims[i]);
//     }
//     cropped_shape.push(keep_rows);
//     cropped_shape.push(keep_cols);
//     crop(tensor, cropped_shape.into())
// }

/// Pads tensor by adding zeros when padded dim is larger than tensor dim
fn padding<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    padded_shape: Shape<D>,
) -> WgpuTensor<E, D> {
    let ranges = padded_shape
        .dims
        .iter()
        .map(|dim| 0..*dim)
        .collect::<Vec<Range<usize>>>()
        .try_into()
        .unwrap();
    slice_assign::<E, D, D>(
        empty_from_context(tensor.context.clone(), &padded_shape),
        ranges,
        tensor,
    )
}

/// Crops tensor by deleting values when cropped dim is smaller than tensor dim
pub(super) fn crop<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    cropped_shape: Shape<D>,
) -> WgpuTensor<E, D> {
    let ranges = cropped_shape
        .dims
        .iter()
        .map(|dim| 0..*dim)
        .collect::<Vec<Range<usize>>>()
        .try_into()
        .unwrap();
    slice::<E, D, D>(tensor, ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestTensor;

    #[test]
    fn padding_already_round_should_have_same_shape() {
        let row = 10;
        let row_divisor = 5;
        let col = 12;
        let col_divisor = 3;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [row, col].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor);

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_already_round_should_have_same_values() {
        let row = 10;
        let row_divisor = 5;
        let col = 12;
        let col_divisor = 3;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let padded = pad_round(tensor.clone().into_primitive(), row_divisor, col_divisor);

        let padded = TestTensor::from_primitive(padded);
        padded.into_data().assert_approx_eq(&tensor.into_data(), 3);
    }

    #[test]
    fn padding_not_round_should_have_rounded_shape() {
        let row = 10;
        let row_divisor = 6;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [12, 15].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor);

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_not_round_should_have_same_values() {
        let row = 10;
        let row_divisor = 6;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let padded = pad_round(tensor.clone().into_primitive(), row_divisor, col_divisor);

        let padded = TestTensor::from_primitive(padded).to_data();
        let tensor = tensor.into_data();
        for i in 0..row {
            for j in 0..col {
                assert!(padded.value[i * 15 + j] == tensor.value[i * col + j]);
            }
        }
    }

    #[test]
    fn padding_not_round_should_have_zero_padding() {
        let row = 10;
        let row_divisor = 6;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let padded = pad_round(tensor.clone().into_primitive(), row_divisor, col_divisor);
        let padded = TestTensor::from_primitive(padded).to_data();

        // check right of matrix
        for i in 0..row {
            for j in col..15 {
                assert!(padded.value[i * 15 + j] == 0.0);
            }
        }
        // check below matrix, including bottom right
        for i in row..12 {
            for j in 0..15 {
                assert!(padded.value[i * 15 + j] == 0.0);
            }
        }
    }

    #[test]
    fn padding_works_with_batch() {
        let row = 10;
        let row_divisor = 4;
        let col = 12;
        let col_divisor = 5;
        let tensor = TestTensor::random([2, 3, row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [2, 3, 12, 15].into();

        let padded = pad_round(tensor.into_primitive(), row_divisor, col_divisor);

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn crop_same_shape_should_be_unchanged_shape() {
        let row = 10;
        let col = 12;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [row, col].into();

        let unpadded = crop(tensor.into_primitive(), [row, col].into());

        assert!(unpadded.shape == expected_shape);
    }

    #[test]
    fn crop_same_shape_should_have_unchanged_values() {
        let row = 10;
        let col = 12;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let unpadded = crop(tensor.clone().into_primitive(), [row, col].into());

        let unpadded = TestTensor::from_primitive(unpadded).to_data();
        let tensor = tensor.into_data();
        for i in 0..row {
            for j in 0..col {
                assert!(unpadded.value[i * col + j] == tensor.value[i * col + j]);
            }
        }
    }

    #[test]
    fn crop_should_decrease_shape() {
        let row = 10;
        let keep_rows = 8;
        let col = 12;
        let keep_cols = 10;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [keep_rows, keep_cols].into();

        let unpadded = crop(tensor.into_primitive(), [keep_rows, keep_cols].into());

        assert!(unpadded.shape == expected_shape);
    }

    #[test]
    fn crop_should_keep_same_values() {
        let row = 4;
        let keep_rows = 3;
        let col = 4;
        let keep_cols = 3;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let unpadded = crop(
            tensor.clone().into_primitive(),
            [keep_rows, keep_cols].into(),
        );

        let unpadded = TestTensor::from_primitive(unpadded).to_data();
        let tensor = tensor.into_data();
        println!("{:?}\n {:?}", unpadded, tensor);

        for i in 0..keep_rows {
            for j in 0..keep_cols {
                assert!(unpadded.value[i * keep_cols + j] == tensor.value[i * col + j]);
            }
        }
    }
}
