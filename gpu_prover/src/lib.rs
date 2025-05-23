#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]
#![feature(btree_cursors)]
#![feature(pointer_is_aligned_to)]
#![feature(array_chunks)]

pub mod allocator;
pub mod barycentric;
pub mod blake2s;
pub mod context;
pub mod device_structures;
pub mod field;
pub mod field_bench;
pub mod ntt;
pub mod ops_complex;
pub mod ops_cub;
pub mod ops_simple;
pub mod prover;
pub mod utils;
pub mod witness;

#[cfg(test)]
mod tests;
