mod base;
mod mem_coalescing;
mod naive;
mod padding;
mod tiling2d;

pub use mem_coalescing::*;
pub use naive::*;
pub use tiling2d::*;

/// Loading is done in a continuous manner. Assumes shapes divisible by block sizes
pub mod tiling2d_continuous_load;
/// Loading is done in a tile manner. Assumes only B_M % T_M == 0
pub mod tiling2d_no_padding;
/// Loading is done in a tile manner. Assumes shapes divisible by block sizes
pub mod tiling2d_tile_load;
/// Loading is done in a tile manner. lhs is transposed
pub mod tiling2d_tile_load_vectorized_mem_access;
