use crate::{element::WgpuElement, tensor::WgpuTensor};

use super::tiling2d_v1;

/// Matmul that should be used
pub fn matmul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    tiling2d_v1::matmul_tiling_2d_default(lhs, rhs)
}
