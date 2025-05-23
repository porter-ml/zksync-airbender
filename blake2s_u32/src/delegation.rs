use super::*;

#[cfg(all(target_arch = "riscv32", not(feature = "delegation")))]
compile_error!("feature `delegation` must be activated on RISC-V architecture to use this module");

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
            "csrrw x0, 0x7c1, {rs}",
            rs = in(reg) offset,
            options(nostack, preserves_flags)
        )
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Blake2sState {
    _aligner: Aligner,
    pub state: [u32; BLAKE2S_STATE_WIDTH_IN_U32_WORDS],
    pub state_word_12: u32, // product of `t` and IV
    pub state_word_14: u32, // product of last round/not and IV
    pub input_buffer: [u32; BLAKE2S_BLOCK_SIZE_U32_WORDS],
    pub t: u32, // we limit ourselves to <4Gb inputs
}

impl Blake2sState {
    pub const SUPPORT_SPEC_SINGLE_ROUND: bool = false;

    #[unroll::unroll_for_loops]
    #[inline(always)]
    pub unsafe fn spec_run_single_round_into_destination<const REDUCED_ROUNDS: bool>(
        &mut self,
        block_len: usize,
        dst: *mut [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
    ) {
        let t = (block_len * core::mem::size_of::<u32>()) as u32;

        #[cfg(all(target_arch = "riscv32", feature = "delegation"))]
        {
            self.state_word_12 = t ^ IV[4];
            self.state_word_14 = 0xffffffffu32 ^ IV[6];

            let _ = csr_trigger_delegation(self.state.as_ptr().addr());

            for i in 0..8 {
                dst.as_mut_unchecked()[i] = self.state[i];
            }
        }

        #[cfg(not(all(target_arch = "riscv32", feature = "delegation")))]
        {
            let mut extended_state = [
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                self.state[6],
                self.state[7],
                IV[0],
                IV[1],
                IV[2],
                IV[3],
                t ^ IV[4],
                IV[5],
                0xffffffff ^ IV[6],
                IV[7],
            ];

            if REDUCED_ROUNDS {
                round_function_reduced_rounds(&mut extended_state, &self.input_buffer);
            } else {
                round_function_full_rounds(&mut extended_state, &self.input_buffer);
            }

            for i in 0..8 {
                dst.as_mut_unchecked()[i] =
                    CONFIGURED_IV[i] ^ extended_state[i] ^ extended_state[i + 8];
            }
        }
    }

    pub const fn new() -> Self {
        Self {
            _aligner: Aligner,
            state: CONFIGURED_IV,
            state_word_12: 0u32, // product of `t` and IV
            state_word_14: 0u32, // product of last round/not and IV
            input_buffer: [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS],
            t: 0,
        }
    }

    #[inline(always)]
    pub fn read_state_for_output(&self) -> [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS] {
        self.state
    }

    #[inline(always)]
    pub fn read_state_for_output_ref(&self) -> &[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS] {
        &self.state
    }

    #[unroll::unroll_for_loops]
    #[inline(always)]
    pub fn reset(&mut self) {
        for i in 0..8 {
            self.state[i] = CONFIGURED_IV[i];
        }
        // self.state = CONFIGURED_IV;
        self.t = 0;
    }

    /// caller must fill the buffer (do not forget to zero-pad),
    /// and then specify the parameters of the input block
    #[inline(always)]
    pub unsafe fn run_round_function<const REDUCED_ROUNDS: bool>(
        &mut self,
        input_size_words: usize,
        last_round: bool,
    ) {
        self.run_round_function_with_byte_len::<REDUCED_ROUNDS>(
            input_size_words * core::mem::size_of::<u32>(),
            last_round,
        );
    }

    #[inline]
    #[unroll::unroll_for_loops]
    pub unsafe fn run_round_function_with_byte_len<const REDUCED_ROUNDS: bool>(
        &mut self,
        input_size_bytes: usize,
        last_round: bool,
    ) {
        self.t += input_size_bytes as u32;

        #[cfg(all(target_arch = "riscv32", feature = "delegation"))]
        {
            self.state_word_12 = self.t ^ IV[4];
            self.state_word_14 = (0xffffffff * last_round as u32) ^ IV[6];

            let _ = csr_trigger_delegation(self.state.as_ptr().addr());
        }

        #[cfg(not(all(target_arch = "riscv32", feature = "delegation")))]
        {
            let mut extended_state = [
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                self.state[6],
                self.state[7],
                IV[0],
                IV[1],
                IV[2],
                IV[3],
                self.t ^ IV[4],
                IV[5],
                (0xffffffff * last_round as u32) ^ IV[6],
                IV[7],
            ];

            if REDUCED_ROUNDS {
                round_function_reduced_rounds(&mut extended_state, &self.input_buffer);
            } else {
                round_function_full_rounds(&mut extended_state, &self.input_buffer);
            }

            for i in 0..8 {
                self.state[i] ^= extended_state[i];
                self.state[i] ^= extended_state[i + 8];
            }
        }
    }

    #[inline(always)]
    #[unroll::unroll_for_loops]
    pub fn absorb<const REDUCED_ROUNDS: bool>(
        &mut self,
        message_block: &[u32; BLAKE2S_BLOCK_SIZE_U32_WORDS],
    ) {
        self.t += BLAKE2S_BLOCK_SIZE_BYTES as u32;

        #[cfg(all(target_arch = "riscv32", feature = "delegation"))]
        {
            self.state_word_12 = self.t ^ IV[4];
            self.state_word_14 = IV[6];

            unsafe {
                core::ptr::copy_nonoverlapping(
                    message_block.as_ptr(),
                    self.input_buffer.as_mut_ptr(),
                    BLAKE2S_BLOCK_SIZE_U32_WORDS,
                );

                let _ = csr_trigger_delegation(self.state.as_ptr().addr());
            }
        }

        #[cfg(not(all(target_arch = "riscv32", feature = "delegation")))]
        {
            let mut extended_state = [
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                self.state[6],
                self.state[7],
                IV[0],
                IV[1],
                IV[2],
                IV[3],
                self.t ^ IV[4],
                IV[5],
                IV[6],
                IV[7],
            ];

            if REDUCED_ROUNDS {
                round_function_reduced_rounds(&mut extended_state, message_block);
            } else {
                round_function_full_rounds(&mut extended_state, message_block);
            }

            for i in 0..8 {
                self.state[i] ^= extended_state[i];
                self.state[i] ^= extended_state[i + 8];
            }
        }
    }

    #[inline(always)]
    #[unroll::unroll_for_loops]
    pub fn absorb_final_block<const REDUCED_ROUNDS: bool>(
        &mut self,
        message_block: &[u32; BLAKE2S_BLOCK_SIZE_U32_WORDS],
        block_len: usize,
        dst: &mut [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
    ) {
        self.t += (block_len * core::mem::size_of::<u32>()) as u32;

        #[cfg(all(target_arch = "riscv32", feature = "delegation"))]
        {
            self.state_word_12 = self.t ^ IV[4];
            self.state_word_14 = 0xffffffff ^ IV[6];

            unsafe {
                core::ptr::copy_nonoverlapping(
                    message_block.as_ptr(),
                    self.input_buffer.as_mut_ptr(),
                    BLAKE2S_BLOCK_SIZE_U32_WORDS,
                );

                let _ = csr_trigger_delegation(self.state.as_ptr().addr());

                core::ptr::copy_nonoverlapping(
                    self.state.as_ptr(),
                    dst.as_mut_ptr(),
                    BLAKE2S_DIGEST_SIZE_U32_WORDS,
                );
            }
        }

        #[cfg(not(all(target_arch = "riscv32", feature = "delegation")))]
        {
            let mut extended_state = [
                self.state[0],
                self.state[1],
                self.state[2],
                self.state[3],
                self.state[4],
                self.state[5],
                self.state[6],
                self.state[7],
                IV[0],
                IV[1],
                IV[2],
                IV[3],
                self.t ^ IV[4],
                IV[5],
                0xffffffff ^ IV[6],
                IV[7],
            ];

            if REDUCED_ROUNDS {
                round_function_reduced_rounds(&mut extended_state, message_block);
            } else {
                round_function_full_rounds(&mut extended_state, message_block);
            }

            *dst = self.state;

            for i in 0..8 {
                dst[i] ^= extended_state[i];
                dst[i] ^= extended_state[i + 8];
            }
        }
    }

    #[inline(always)]
    #[unroll::unroll_for_loops]
    pub fn compress_two_to_one<const REDUCED_ROUNDS: bool>(
        message_block: &[u32; BLAKE2S_BLOCK_SIZE_U32_WORDS],
        dst: &mut [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
    ) {
        #[cfg(all(target_arch = "riscv32", feature = "delegation"))]
        {
            let mut this = Self::new();
            this.absorb_final_block::<REDUCED_ROUNDS>(message_block, BLAKE2S_BLOCK_SIZE_BYTES, dst);
        }

        #[cfg(not(all(target_arch = "riscv32", feature = "delegation")))]
        {
            let mut extended_state = [
                CONFIGURED_IV[0],
                CONFIGURED_IV[1],
                CONFIGURED_IV[2],
                CONFIGURED_IV[3],
                CONFIGURED_IV[4],
                CONFIGURED_IV[5],
                CONFIGURED_IV[6],
                CONFIGURED_IV[7],
                IV[0],
                IV[1],
                IV[2],
                IV[3],
                (BLAKE2S_BLOCK_SIZE_BYTES as u32) ^ IV[4],
                IV[5],
                0xffffffff ^ IV[6],
                IV[7],
            ];

            if REDUCED_ROUNDS {
                round_function_reduced_rounds(&mut extended_state, message_block);
            } else {
                round_function_full_rounds(&mut extended_state, message_block);
            }

            *dst = CONFIGURED_IV;

            for i in 0..8 {
                dst[i] ^= extended_state[i];
                dst[i] ^= extended_state[i + 8];
            }
        }
    }
}
