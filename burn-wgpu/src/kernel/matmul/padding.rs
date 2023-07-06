use burn_tensor::Shape;

use crate::{
    element::WgpuElement,
    kernel::{slice, slice_assign},
    tensor::WgpuTensor,
};

/// 
pub(super) fn pad<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    row_modulo: usize,
    col_modulo: usize,
) -> WgpuTensor<E, D> {
    if tensor.shape.dims[D - 2] % row_modulo == 0 && tensor.shape.dims[D - 1] % col_modulo == 0 {
        return tensor;
    }
    let mut padded_dims = Vec::with_capacity(D);
    let mut ranges = Vec::with_capacity(D);
    for i in 0..D - 2 {
        let batch = tensor.shape.dims[i];
        padded_dims.push(batch);
        ranges.push(0..batch);
    }
    let row = tensor.shape.dims[D - 2];
    let col = tensor.shape.dims[D - 1];
    padded_dims.push(((row - 1) / row_modulo + 1) * row_modulo);
    padded_dims.push(((col - 1) / col_modulo + 1) * col_modulo);
    ranges.push(0..row);
    ranges.push(0..col);

    let shape_out: Shape<D> = padded_dims.into();
    let buffer = tensor
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());

    let padded = WgpuTensor::new(tensor.context.clone(), shape_out, buffer);
    slice_assign::<E, D, D>(padded, ranges.try_into().unwrap(), tensor)
}

/// TODO
pub(super) fn crop<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    keep_rows: usize,
    keep_cols: usize,
) -> WgpuTensor<E, D> {
    if tensor.shape.dims[D - 2] <= keep_rows && tensor.shape.dims[D - 1] <= keep_cols {
        return tensor;
    }
    let mut unpadded_dims = Vec::with_capacity(D);
    let mut ranges = Vec::with_capacity(D);
    for i in 0..D - 2 {
        let batch = tensor.shape.dims[i];
        unpadded_dims.push(batch);
        ranges.push(0..batch);
    }
    unpadded_dims.push(keep_rows);
    unpadded_dims.push(keep_cols);
    ranges.push(0..keep_rows);
    ranges.push(0..keep_cols);

    slice::<E, D, D>(tensor, ranges.try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestTensor;

    #[test]
    fn padding_already_round_should_have_same_shape() {
        let row = 10;
        let row_modulo = 5;
        let col = 12;
        let col_modulo = 3;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [row, col].into();

        let padded = pad(tensor.into_primitive(), row_modulo, col_modulo);

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_already_round_should_have_same_values() {
        let row = 10;
        let row_modulo = 5;
        let col = 12;
        let col_modulo = 3;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let padded = pad(tensor.clone().into_primitive(), row_modulo, col_modulo);

        let padded = TestTensor::from_primitive(padded);
        padded.into_data().assert_approx_eq(&tensor.into_data(), 3);
    }

    #[test]
    fn padding_not_round_should_have_rounded_shape() {
        let row = 10;
        let row_modulo = 6;
        let col = 12;
        let col_modulo = 5;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [12, 15].into();

        let padded = pad(tensor.into_primitive(), row_modulo, col_modulo);

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn padding_not_round_should_have_same_values() {
        let row = 10;
        let row_modulo = 6;
        let col = 12;
        let col_modulo = 5;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let padded = pad(tensor.clone().into_primitive(), row_modulo, col_modulo);

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
        let row_modulo = 6;
        let col = 12;
        let col_modulo = 5;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let padded = pad(tensor.clone().into_primitive(), row_modulo, col_modulo);
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
        let row_modulo = 4;
        let col = 12;
        let col_modulo = 5;
        let tensor = TestTensor::random([2, 3, row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [2, 3, 12, 15].into();

        let padded = pad(tensor.into_primitive(), row_modulo, col_modulo);

        assert!(padded.shape == expected_shape);
    }

    #[test]
    fn crop_same_shape_should_be_unchanged_shape() {
        let row = 10;
        let col = 12;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);
        let expected_shape = [row, col].into();

        let unpadded = crop(tensor.into_primitive(), row, col);

        assert!(unpadded.shape == expected_shape);
    }

    #[test]
    fn crop_same_shape_should_have_unchanged_values() {
        let row = 10;
        let col = 12;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let unpadded = crop(tensor.clone().into_primitive(), row, col);

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

        let unpadded = crop(tensor.into_primitive(), keep_rows, keep_cols);

        assert!(unpadded.shape == expected_shape);
    }

    #[test]
    fn crop_should_keep_same_values() {
        let row = 4;
        let keep_rows = 3;
        let col = 4;
        let keep_cols = 3;
        let tensor = TestTensor::random([row, col], burn_tensor::Distribution::Standard);

        let unpadded = crop(tensor.clone().into_primitive(), keep_rows, keep_cols);

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
