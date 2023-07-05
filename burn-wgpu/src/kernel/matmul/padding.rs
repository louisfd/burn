use burn_tensor::{ops::TensorOps, Shape};

use crate::{element::WgpuElement, ops::BaseOps, tensor::WgpuTensor, GraphicsApi, WgpuBackend};

pub fn pad<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    row_modulo: usize,
    col_modulo: usize,
) -> WgpuTensor<E, D> {
    if tensor.shape.dims[D - 2] % row_modulo == 0 && tensor.shape.dims[D - 1] % col_modulo == 0 {
        return tensor;
    }
    let mut padded_dims = Vec::new();
    let mut ranges = Vec::new();
    for i in 0..D - 2 {
        let batch = tensor.shape.dims[i];
        padded_dims.push(batch);
        ranges.push(0..batch)
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
    WgpuBackend::index_assign(padded, ranges.try_into().unwrap(), tensor)
}
