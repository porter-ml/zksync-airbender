use crate::bitreverse_enumeration_inplace;
use crate::domain_generator_for_size;
use crate::materialize_powers_parallel_starting_with_one;
use crate::radix_4_step::Radix4;
use crate::FftDirection;
use crate::GoodAllocator;
use field::Field;
use field::Mersenne31Complex;
use field::Mersenne31ComplexVectorized;
use field::Mersenne31ComplexVectorizedInterleaved;
use field::TwoAdicField;
use field::{rotate_90_forward, rotate_90_inversed};
use worker::Worker;

pub fn precompute_twiddles(domain_size: usize) -> (Radix4, Radix4) {
    let fft_inverse = Radix4::new_with_base(domain_size, FftDirection::Inverse);
    let fft_forward = Radix4::new_with_base(domain_size, FftDirection::Forward);
    (fft_forward, fft_inverse)
}

pub fn compute_twiddle(index: usize, fft_len: usize, direction: FftDirection) -> Mersenne31Complex {
    let omega = match direction {
        FftDirection::Forward => domain_generator_for_size::<Mersenne31Complex>(fft_len as u64),
        FftDirection::Inverse => domain_generator_for_size::<Mersenne31Complex>(fft_len as u64)
            .inverse()
            .expect("must always exist for domain generator"),
    };
    omega.pow(index as u32)
}

pub fn rotate_90(value: Mersenne31Complex, direction: FftDirection) -> Mersenne31Complex {
    let mut value = value;
    match direction {
        FftDirection::Forward => Mersenne31Complex {
            c0: value.c1,
            c1: *value.c0.negate(),
        },
        FftDirection::Inverse => Mersenne31Complex {
            c0: *value.c1.negate(),
            c1: value.c0,
        },
    }
}

pub fn rotate_90_vectorized(
    value: Mersenne31ComplexVectorized,
    direction: FftDirection,
) -> Mersenne31ComplexVectorized {
    let mut value = value;
    match direction {
        FftDirection::Forward => Mersenne31ComplexVectorized {
            c0: value.c1,
            c1: *value.c0.negate(),
        },
        FftDirection::Inverse => Mersenne31ComplexVectorized {
            c0: *value.c1.negate(),
            c1: value.c0,
        },
    }
}

pub fn rotate_90_vectorized2(
    value: Mersenne31ComplexVectorizedInterleaved,
    direction: FftDirection,
) -> Mersenne31ComplexVectorizedInterleaved {
    // let mut value = value;
    match direction {
        FftDirection::Forward => rotate_90_forward(value),
        FftDirection::Inverse => rotate_90_inversed(value),
    }
}

fn every_nth_element<E: TwoAdicField, A: GoodAllocator>(values: &mut Vec<E, A>, n: i32) {
    // Retain elements with evenly divisible indexes.
    let mut c = -1;
    values.retain(|_| {
        c += 1;
        return c % n == 0;
    });
}

pub fn precompute_twiddles_for_fft_radix4<
    E: TwoAdicField,
    A: GoodAllocator,
    const INVERSED: bool,
>(
    fft_size: usize,
    worker: &Worker,
) -> Vec<E, A> {
    debug_assert!(fft_size.is_power_of_two());

    let mut omega = domain_generator_for_size::<E>(fft_size as u64);
    if INVERSED {
        omega = omega
            .inverse()
            .expect("must always exist for domain generator");
    }

    let num_powers = fft_size;
    let mut powers = materialize_powers_parallel_starting_with_one(omega, num_powers, &worker);
    bitreverse_enumeration_inplace(&mut powers);
    every_nth_element(&mut powers, 4);

    powers
}
