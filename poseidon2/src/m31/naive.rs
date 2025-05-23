use core::mem::MaybeUninit;

use super::*;
use field::{Field, Mersenne31Field};

#[unroll::unroll_for_loops]
pub fn poseidon2_compress(input: &[Mersenne31Field; 16]) -> [Mersenne31Field; 8] {
    let working_state = poseidon_permutation_inner(&input);

    #[allow(invalid_value)]
    let mut result: [Mersenne31Field; 8] = unsafe { MaybeUninit::uninit().assume_init() };

    // add initial state, reduce and write back
    for i in 0..8 {
        result[i] = Mersenne31Field::from_u62(working_state[i] + input[i].0 as u64);
    }

    result
}

#[unroll::unroll_for_loops]
pub fn poseidon_permutation(state: &mut [Mersenne31Field; 16]) {
    let working_state = poseidon_permutation_inner(&*state);

    // reduce and write back
    for i in 0..16 {
        state[i] = Mersenne31Field::from_u62(working_state[i]);
    }
}

#[unroll::unroll_for_loops]
#[inline(always)]
fn poseidon_permutation_inner(state: &[Mersenne31Field; 16]) -> [u64; 16] {
    let mut working_state = state.map(|el| el.0 as u64);

    // outer matrix multiplication
    mds_light_permutation_over_u64(&mut working_state);
    // then rounds of round constant/sbox/external matrix
    for i in 0..4 {
        // RF/2
        reduce_apply_rc_and_sbox(&mut working_state, &EXTERNAL_INITIAL_CONSTANTS[i]);
        mds_light_permutation_over_u64(&mut working_state);
    }
    // inner rounds
    for i in 0..14 {
        // RP
        reduce_apply_sbox_inner_round(&mut working_state, INTERNAL_CONSTANTS[i]);
        matmul_internal(&mut working_state);
    }
    for i in 0..4 {
        // RF/2
        reduce_apply_rc_and_sbox(&mut working_state, &EXTERNAL_TERMINAL_CONSTANTS[i]);
        mds_light_permutation_over_u64(&mut working_state);
    }

    working_state
}

/// Multiply a 4-element vector x by:
/// [ 2 3 1 1 ]
/// [ 1 2 3 1 ]
/// [ 1 1 2 3 ]
/// [ 3 1 1 2 ].
///
/// We operate over u64 as we will accumulate the whole matrix multiplication this way.
/// For 16x16 full matrix size assuming initial elements to be 31 bit we will not overflow
/// 62 bits (from which we have simple and efficient reduction)
#[inline(always)]
fn mul_by_4x4_chunk(a: u64, b: u64, c: u64, d: u64) -> (u64, u64, u64, u64) {
    let t01 = a + b;
    let t23 = c + d;
    let t0123 = t01 + t23;
    let t01123 = t0123 + b;
    let t01233 = t0123 + d;
    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].

    let new_d = t01233 + (a << 1); // 3*x[0] + x[1] + x[2] + 2*x[3]
    let new_b = t01123 + (c << 1); // x[0] + 2*x[1] + 3*x[2] + x[3]
    let new_a = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    let new_c = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]

    (new_a, new_b, new_c, new_d)
}

/// NOTE: assumes reduced M31 elements as input (31 bit) to avoid overflows,
/// and outputs unreduced 62 (at worst) bits
pub fn mds_light_permutation_over_u64_and_add_round_constants(
    state: &mut [u64; 16],
    rc: &[u32; 16],
) {
    let x0 = state[0];
    let x1 = state[1];
    let x2 = state[2];
    let x3 = state[3];

    let x4 = state[4];
    let x5 = state[5];
    let x6 = state[6];
    let x7 = state[7];

    let x8 = state[8];
    let x9 = state[9];
    let x10 = state[10];
    let x11 = state[11];

    let x12 = state[12];
    let x13 = state[13];
    let x14 = state[14];
    let x15 = state[15];

    let (t0, t1, t2, t3) = mul_by_4x4_chunk(x0, x1, x2, x3);
    let (t4, t5, t6, t7) = mul_by_4x4_chunk(x4, x5, x6, x7);
    let (t8, t9, t10, t11) = mul_by_4x4_chunk(x8, x9, x10, x11);
    let (t12, t13, t14, t15) = mul_by_4x4_chunk(x12, x13, x14, x15);

    let s0 = t0 + t4 + t8 + t12;
    let s1 = t1 + t5 + t9 + t13;
    let s2 = t2 + t6 + t10 + t14;
    let s3 = t3 + t7 + t11 + t15;

    let x0 = t0 + s0 + rc[0] as u64;
    let x1 = t1 + s1 + rc[1] as u64;
    let x2 = t2 + s2 + rc[2] as u64;
    let x3 = t3 + s3 + rc[3] as u64;

    let x4 = t4 + s0 + rc[4] as u64;
    let x5 = t5 + s1 + rc[5] as u64;
    let x6 = t6 + s2 + rc[6] as u64;
    let x7 = t7 + s3 + rc[7] as u64;

    let x8 = t8 + s0 + rc[8] as u64;
    let x9 = t9 + s1 + rc[9] as u64;
    let x10 = t10 + s2 + rc[10] as u64;
    let x11 = t11 + s3 + rc[11] as u64;

    let x12 = t12 + s0 + rc[12] as u64;
    let x13 = t13 + s1 + rc[13] as u64;
    let x14 = t14 + s2 + rc[14] as u64;
    let x15 = t15 + s3 + rc[15] as u64;

    state[0] = x0;
    state[1] = x1;
    state[2] = x2;
    state[3] = x3;

    state[4] = x4;
    state[5] = x5;
    state[6] = x6;
    state[7] = x7;

    state[8] = x8;
    state[9] = x9;
    state[10] = x10;
    state[11] = x11;

    state[12] = x12;
    state[13] = x13;
    state[14] = x14;
    state[15] = x15;
}

/// NOTE: assumes reduced M31 elements as input (31 bit) to avoid overflows,
/// and outputs unreduced 62 (at worst) bits
pub fn mds_light_permutation_over_u64(state: &mut [u64; 16]) {
    // first we cast it into u64, without any reduction
    let x0 = state[0];
    let x1 = state[1];
    let x2 = state[2];
    let x3 = state[3];

    let x4 = state[4];
    let x5 = state[5];
    let x6 = state[6];
    let x7 = state[7];

    let x8 = state[8];
    let x9 = state[9];
    let x10 = state[10];
    let x11 = state[11];

    let x12 = state[12];
    let x13 = state[13];
    let x14 = state[14];
    let x15 = state[15];

    let (t0, t1, t2, t3) = mul_by_4x4_chunk(x0, x1, x2, x3);
    let (t4, t5, t6, t7) = mul_by_4x4_chunk(x4, x5, x6, x7);
    let (t8, t9, t10, t11) = mul_by_4x4_chunk(x8, x9, x10, x11);
    let (t12, t13, t14, t15) = mul_by_4x4_chunk(x12, x13, x14, x15);

    let s0 = t0 + t4 + t8 + t12;
    let s1 = t1 + t5 + t9 + t13;
    let s2 = t2 + t6 + t10 + t14;
    let s3 = t3 + t7 + t11 + t15;

    let x0 = t0 + s0;
    let x1 = t1 + s1;
    let x2 = t2 + s2;
    let x3 = t3 + s3;

    let x4 = t4 + s0;
    let x5 = t5 + s1;
    let x6 = t6 + s2;
    let x7 = t7 + s3;

    let x8 = t8 + s0;
    let x9 = t9 + s1;
    let x10 = t10 + s2;
    let x11 = t11 + s3;

    let x12 = t12 + s0;
    let x13 = t13 + s1;
    let x14 = t14 + s2;
    let x15 = t15 + s3;

    state[0] = x0;
    state[1] = x1;
    state[2] = x2;
    state[3] = x3;

    state[4] = x4;
    state[5] = x5;
    state[6] = x6;
    state[7] = x7;

    state[8] = x8;
    state[9] = x9;
    state[10] = x10;
    state[11] = x11;

    state[12] = x12;
    state[13] = x13;
    state[14] = x14;
    state[15] = x15;
}

#[unroll::unroll_for_loops]
#[inline(always)]
fn reduce_apply_sbox_inner_round(state: &mut [u64; 16], rc: u32) {
    // only apply round constant and s-xob to the first element
    {
        let el = state[0] + (rc as u64);
        let el = Mersenne31Field::from_u62(el);
        let mut t = el;
        t.square();
        t.square();
        t.mul_assign(&el);
        state[0] = t.0 as u64;
    }
    for i in 1..16 {
        state[i] = Mersenne31Field::from_u62(state[i]).0 as u64;
    }
}

/// NOTE: accepts not yet reduced elements,
/// adds round constants, reduces, and applies non-linearity
#[unroll::unroll_for_loops]
#[inline(always)]
fn reduce_apply_rc_and_sbox(state: &mut [u64; 16], rc: &[u32; 16]) {
    for i in 0..16 {
        let el = state[i] + (rc[i] as u64);
        let el = Mersenne31Field::from_u62(el);
        let mut t = el;
        t.square();
        t.square();
        t.mul_assign(&el);
        state[i] = t.0 as u64;
    }
}

/// Multiply state by the matrix (1 + Diag(V))
///
/// Here V is the vector [-2] + 1 << shifts. This used delayed reduction to be slightly faster.

/// NOTE: takes reduced elements, and outputs unreduced, but not overflowing 62 bits
#[unroll::unroll_for_loops]
#[inline(always)]
pub fn matmul_internal(state: &mut [u64; 16]) {
    // compiler will handle it
    let mut partial_sum = 0u64;
    for i in 1..16 {
        debug_assert!(state[i] <= u32::MAX as u64);
        partial_sum += state[i] as u64;
    }
    debug_assert!(state[0] <= u32::MAX as u64);
    let full_sum = partial_sum + state[0] as u64;

    // first element is a little special
    let mut t = Mersenne31Field(state[0] as u32);
    t.negate();
    state[0] = partial_sum + (t.0 as u64);

    for i in 1..16 {
        state[i] = full_sum + (state[i] << POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS[i - 1]);
    }
}
