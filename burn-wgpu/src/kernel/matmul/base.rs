use std::sync::Arc;

use burn_tensor::Shape;

use crate::{context::Context, element::WgpuElement, tensor::WgpuTensor};

pub(super) fn empty_from_context<E: WgpuElement, const D: usize>(
    context: Arc<Context>,
    shape: Shape<D>,
) -> WgpuTensor<E, D> {
    let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(context, shape, buffer)
}
