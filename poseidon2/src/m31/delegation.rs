use super::HASH_SIZE_U32_WORDS;
use ::field::Mersenne31Field;
use core::mem::MaybeUninit;
use non_determinism_source::NonDeterminismSource;

// we need our input to be 16-bit page aligned
#[cfg(all(target_arch = "riscv32", feature = "delegation"))]
#[derive(Clone, Copy, Debug)]
#[repr(align(65536))]
struct Aligner;

// We will align at 64-bit word
#[cfg(not(all(target_arch = "riscv32", feature = "delegation")))]
#[derive(Clone, Copy, Debug)]
#[repr(align(8))]
struct Aligner;

#[cfg(all(target_arch = "riscv32", feature = "delegation"))]
#[inline(always)]
fn csr_trigger_delegation(offset: usize) {
    debug_assert!(offset as u16 == 0);
    unsafe {
        core::arch::asm!(
            "csrrw x0, 0x7c6, {rs}",
            rs = in(reg) offset,
            options(nostack, preserves_flags)
        )
    }
}

// We put 8 elements of the leaf hash/node hash, and single boolean of left/right. Compressor is responsible to
// provide witness internally

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Poseidon2Compressor {
    _aligner: Aligner,
    pub input: [Mersenne31Field; HASH_SIZE_U32_WORDS],
    pub input_is_right: u32,
}

impl Poseidon2Compressor {
    #[allow(invalid_value)]
    #[inline(always)]
    pub fn new() -> Self {
        unsafe {
            let new: Self = MaybeUninit::uninit().assume_init();
            new
        }
    }

    #[inline(always)]
    pub fn get_output(&self) -> [Mersenne31Field; HASH_SIZE_U32_WORDS] {
        let mut result = self.input;
        for dst in result.iter_mut() {
            *dst = Mersenne31Field::from_nonreduced_u32(dst.0);
        }

        result
    }

    #[inline(always)]
    pub unsafe fn provide_witness_and_compress<I: NonDeterminismSource>(
        &mut self,
        input_is_right: bool,
    ) {
        #[cfg(all(target_arch = "riscv32", feature = "delegation"))]
        {
            self.input_is_right = input_is_right as u32;
            let _ = csr_trigger_delegation(self.input.as_ptr().addr());
        }

        #[cfg(not(all(target_arch = "riscv32", feature = "delegation")))]
        {
            use crate::m31::poseidon2_compress;
            use field::PrimeField;

            #[allow(invalid_value)]
            let mut state: [Mersenne31Field; 16] = MaybeUninit::uninit().assume_init();

            let offset = if input_is_right { 8 } else { 0 };
            for i in 0..8 {
                state[i + offset] = self.input[i];
            }
            let offset = if input_is_right { 0 } else { 8 };
            for i in 0..8 {
                let witness_value = Mersenne31Field(I::read_reduced_field_element(
                    Mersenne31Field::CHARACTERISTICS as u32,
                ));
                state[i + offset] = witness_value;
            }
            self.input = poseidon2_compress(&state);
        }
    }
}
