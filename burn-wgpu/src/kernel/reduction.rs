use super::{build_info, KernelGenerator, KernelSettings};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;
use std::sync::Arc;

kernel_wgsl!(RecursiveSumRaw, "../template/reduction/recursive_sum.wgsl");
kernel_wgsl!(ReductionDimRaw, "../template/reduction/reduce_dim.wgsl");

struct SumDimRaw;

impl KernelGenerator for SumDimRaw {
    type Source = String;

    fn generate() -> Self::Source {
        ReductionDimRaw::generate().replace("ASSIGN", "output[global_id.x] = sum;")
    }
}

struct MeanDimRaw;

impl KernelGenerator for MeanDimRaw {
    type Source = String;

    fn generate() -> Self::Source {
        ReductionDimRaw::generate()
            .replace("ASSIGN", "output[global_id.x] = sum / elem(shape_dim);")
    }
}

pub fn reduction_sum<E: WgpuElement, const D: usize>(input: WgpuTensor<E, D>) -> WgpuTensor<E, 1> {
    const WORKGROUP: usize = 256;

    let mut input_buffer = input.buffer;
    let mut num_invocations =
        f32::ceil(input.shape.num_elements() as f32 / WORKGROUP as f32) as usize;

    let kernel = input
        .context
        .compile::<KernelSettings<RecursiveSumRaw, E, i32, WORKGROUP, 1, 1>>();

    loop {
        let buffer = input
            .context
            .create_buffer(core::mem::size_of::<E>() * num_invocations);
        let workgroup = WorkGroup::new((num_invocations as usize) as u32, 1, 1);

        input
            .context
            .execute(&workgroup, &kernel, &[&input_buffer, &buffer]);

        if num_invocations == 1 {
            return WgpuTensor::new(input.context, Shape::new([1]), Arc::new(buffer));
        }

        input_buffer = Arc::new(buffer);
        num_invocations = f32::ceil(num_invocations as f32 / WORKGROUP as f32) as usize;
    }
}

pub fn reduction_sum_dim<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    reduction_dim::<SumDimRaw, E, D>(input, dim)
}

pub fn reduction_mean_dim<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    reduction_dim::<MeanDimRaw, E, D>(input, dim)
}

fn reduction_dim<K: KernelGenerator, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let buffer = input
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), shape_out, Arc::new(buffer));

    let kernel = input
        .context
        .compile::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&input.buffer, &output.buffer, &info_buffers],
    );

    output
}
