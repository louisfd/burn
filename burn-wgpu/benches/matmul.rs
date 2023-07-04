use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::{
        matmul_mem_coalescing_default, matmul_naive_default, tiling2d_v1, tiling2d_v2, tiling2d_v3,
    },
    run_benchmark, GraphicsApi, WgpuBackend, WgpuDevice,
};

trait MatmulFunction<B: Backend, const D: usize> {
    fn run(lhs: Tensor<B, D>, rhs: Tensor<B, D>) -> Tensor<B, D>;
}

struct MatmulBenchmark<F, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    matmul: PhantomData<F>,
}

impl<F, const D: usize, G> Benchmark<G> for MatmulBenchmark<F, D>
where
    F: MatmulFunction<WgpuBackend<G, f32, i32>, D>,
    G: GraphicsApi,
{
    type Args = (
        Tensor<WgpuBackend<G, f32, i32>, D>,
        Tensor<WgpuBackend<G, f32, i32>, D>,
    );

    fn name(&self) -> String {
        format!(
            "{:?} {:?} x {:?}",
            std::any::type_name::<F>(),
            self.shape_lhs.dims,
            self.shape_rhs.dims
        )
    }

    fn num_samples(&self) -> usize {
        3
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats {
            F::run(lhs.clone(), rhs.clone());
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Standard).to_device(device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Standard).to_device(device);

        (lhs, rhs)
    }
}

struct NaiveMatmul;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D> for NaiveMatmul {
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(matmul_naive_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct MemCoalescingMatmul;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for MemCoalescingMatmul
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(matmul_mem_coalescing_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct Tiling2DMatmulV1;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulV1
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tiling2d_v1::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct Tiling2DMatmulV2;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulV2
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tiling2d_v2::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}
struct Tiling2DMatmulV3;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulV3
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tiling2d_v3::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

fn main() {
    let batch_size = 2;
    let matrix_size = 512;
    // run_benchmark!(MatmulBenchmark::<NaiveMatmul, 3> {
    //     shape_lhs: [batch_size, matrix_size, matrix_size].into(),
    //     shape_rhs: [batch_size, matrix_size, matrix_size].into(),
    //     num_repeats: 10,
    //     matmul: PhantomData::default()
    // });
    // run_benchmark!(MatmulBenchmark::<MemCoalescingMatmul, 3> {
    //     shape_lhs: [batch_size, matrix_size, matrix_size].into(),
    //     shape_rhs: [batch_size, matrix_size, matrix_size].into(),
    //     num_repeats: 10,
    //     matmul: PhantomData::default()
    // });
    // run_benchmark!(MatmulBenchmark::<Tiling2DMatmulV1, 3> {
    //     shape_lhs: [batch_size, matrix_size, matrix_size].into(),
    //     shape_rhs: [batch_size, matrix_size, matrix_size].into(),
    //     num_repeats: 10,
    //     matmul: PhantomData::default()
    // });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulV2, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats: 3,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulV3, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats: 3,
        matmul: PhantomData::default()
    });
}
