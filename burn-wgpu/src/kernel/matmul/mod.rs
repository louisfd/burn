mod mem_coalescing;
mod naive;
mod tiling2d;

pub use mem_coalescing::*;
pub use naive::*;
pub use tiling2d::*;

/// v1 uses the kernel which is valid on many parameter combinations
/// but slow because it makes many verifications
pub mod tiling2d_v1;
/// v2 assumes B_M % T_M == 0, removing the need for
/// actual_T_M and actual_T_N
pub mod tiling2d_v2;
/// v3 changes the loading from tiles to continuous elements
pub mod tiling2d_v3;
