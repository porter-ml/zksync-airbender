use super::*;

pub const INITIAL_EXTENDED_STATE: [u32; BLAKE2S_EXTENDED_STATE_WIDTH_IN_U32_WORDS] = const {
    let mut result = [0u32; BLAKE2S_EXTENDED_STATE_WIDTH_IN_U32_WORDS];
    let mut i = 0;
    while i < 8 {
        result[i] = CONFIGURED_IV[i];
        i += 1;
    }

    result
};

#[derive(Clone, Copy, Debug)]
pub struct Blake2sState {
    extended_state: [u32; BLAKE2S_EXTENDED_STATE_WIDTH_IN_U32_WORDS],
    t: u32, // we limit ourselves to <4Gb inputs
}

impl Blake2sState {
    pub const fn new() -> Self {
        Self {
            extended_state: INITIAL_EXTENDED_STATE,
            t: 0,
        }
    }

    #[inline(always)]
    #[unroll::unroll_for_loops]
    pub fn absorb<const REDUCED_ROUNDS: bool>(
        &mut self,
        message_block: &[u32; BLAKE2S_BLOCK_SIZE_U32_WORDS],
    ) {
        let h = [
            self.extended_state[0],
            self.extended_state[1],
            self.extended_state[2],
            self.extended_state[3],
            self.extended_state[4],
            self.extended_state[5],
            self.extended_state[6],
            self.extended_state[7],
        ];
        for i in 0..8 {
            self.extended_state[i + 8] = IV[i];
        }
        self.t += BLAKE2S_BLOCK_SIZE_BYTES as u32;
        self.extended_state[12] ^= self.t;
        // 13th element is not touched
        // 14th element is not touched

        if REDUCED_ROUNDS {
            round_function_reduced_rounds(&mut self.extended_state, message_block);
        } else {
            round_function_full_rounds(&mut self.extended_state, message_block);
        }

        for i in 0..8 {
            self.extended_state[i] ^= h[i];
            self.extended_state[i] ^= self.extended_state[i + 8];
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
        *dst = [
            self.extended_state[0],
            self.extended_state[1],
            self.extended_state[2],
            self.extended_state[3],
            self.extended_state[4],
            self.extended_state[5],
            self.extended_state[6],
            self.extended_state[7],
        ];
        for i in 0..8 {
            self.extended_state[i + 8] = IV[i];
        }
        self.t += (block_len * core::mem::size_of::<u32>()) as u32;
        self.extended_state[12] ^= self.t;
        // 13th element is not touched
        self.extended_state[14] ^= 0xffffffff;

        if REDUCED_ROUNDS {
            round_function_reduced_rounds(&mut self.extended_state, message_block);
        } else {
            round_function_full_rounds(&mut self.extended_state, message_block);
        }

        for i in 0..8 {
            dst[i] ^= self.extended_state[i];
            dst[i] ^= self.extended_state[i + 8];
        }
    }

    #[inline(always)]
    #[unroll::unroll_for_loops]
    pub fn compress_two_to_one<const REDUCED_ROUNDS: bool>(
        message_block: &[u32; BLAKE2S_BLOCK_SIZE_U32_WORDS],
        dst: &mut [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
    ) {
        let mut this = Self::new();

        *dst = [
            this.extended_state[0],
            this.extended_state[1],
            this.extended_state[2],
            this.extended_state[3],
            this.extended_state[4],
            this.extended_state[5],
            this.extended_state[6],
            this.extended_state[7],
        ];
        for i in 0..8 {
            this.extended_state[i + 8] = IV[i];
        }
        this.t += BLAKE2S_BLOCK_SIZE_BYTES as u32;
        this.extended_state[12] ^= this.t;
        // 13th element is not touched
        this.extended_state[14] ^= 0xffffffff;

        if REDUCED_ROUNDS {
            round_function_reduced_rounds(&mut this.extended_state, message_block);
        } else {
            round_function_full_rounds(&mut this.extended_state, message_block);
        }

        for i in 0..8 {
            dst[i] ^= this.extended_state[i];
            dst[i] ^= this.extended_state[i + 8];
        }
    }
}
