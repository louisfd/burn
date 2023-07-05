use std::marker::PhantomData;

use burn_tensor::{backend::Backend, ops::TensorOps, Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::{
        matmul_mem_coalescing_default, matmul_naive_default, tiling2d_continuous_load,
        tiling2d_no_padding, tiling2d_tile_load, tiling2d_tile_load_vectorized_mem_access,
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
        10
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

struct Tiling2DMatmulNoPadding;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulNoPadding
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tiling2d_no_padding::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}
struct Tiling2DMatmulContinuousLoad;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulContinuousLoad
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tiling2d_continuous_load::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}
struct Tiling2DMatmulTileLoad;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulTileLoad
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tiling2d_tile_load::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct Tiling2DMatmulTileLoadVectorizedMemAccess;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulTileLoadVectorizedMemAccess
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(
            tiling2d_tile_load_vectorized_mem_access::matmul_tiling_2d_default(
                lhs.into_primitive(),
                rhs.into_primitive(),
            ),
        )
    }
}

fn main() {
    let num_repeats = 10;
    let batch_size = 10;
    let matrix_size = 1000;
    run_benchmark!(MatmulBenchmark::<NaiveMatmul, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<MemCoalescingMatmul, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulNoPadding, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulContinuousLoad, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulTileLoad, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(
        MatmulBenchmark::<Tiling2DMatmulTileLoadVectorizedMemAccess, 3> {
            shape_lhs: [batch_size, matrix_size, matrix_size].into(),
            shape_rhs: [batch_size, matrix_size, matrix_size].into(),
            num_repeats,
            matmul: PhantomData::default()
        }
    );
}
