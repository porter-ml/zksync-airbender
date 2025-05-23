use super::*;
use worker::Worker;

const BLAKE2S_NO_RESULT: u64 = u64::MAX;
const BLAKE2S_ROUNDS_PER_INVOCAITON: usize = 1 << 16u32;

impl Blake2sTranscript {
    pub fn search_pow(seed: &Seed, pow_bits: u32, worker: &Worker) -> (Seed, u64) {
        assert!(pow_bits <= 32);

        let initial_state = [
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
            IV[4] ^ (((BLAKE2S_DIGEST_SIZE_U32_WORDS + 2) * core::mem::size_of::<u32>()) as u32),
            IV[5],
            IV[6] ^ 0xffffffff,
            IV[7],
        ];

        let mut base_input = [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS];
        base_input[..BLAKE2S_DIGEST_SIZE_U32_WORDS].copy_from_slice(&seed.0);

        if pow_bits <= BLAKE2S_ROUNDS_PER_INVOCAITON.trailing_zeros() {
            // serial case
            let mut input = base_input;
            for challenge in 0u64..(BLAKE2S_NO_RESULT - 1) {
                // we expect somewhat "good" hash distribution

                // write LE
                input[BLAKE2S_DIGEST_SIZE_U32_WORDS] = challenge as u32;
                input[BLAKE2S_DIGEST_SIZE_U32_WORDS + 1] = (challenge >> 32) as u32;
                let mut state = initial_state;
                if USE_REDUCED_BLAKE2_ROUNDS {
                    round_function_reduced_rounds(&mut state, &input);
                } else {
                    round_function_full_rounds(&mut state, &input);
                }
                let word_to_test = CONFIGURED_IV[0] ^ state[0] ^ state[8];

                if word_to_test <= (0xffffffff >> pow_bits) {
                    let mut output = CONFIGURED_IV;

                    for i in 0..8 {
                        output[i] ^= state[i];
                        output[i] ^= state[i + 8];
                    }

                    return (Seed(output), challenge);
                }
            }
        }

        use std::sync::atomic::AtomicU64;
        use std::sync::atomic::Ordering;

        let result = std::sync::Arc::new(AtomicU64::new(BLAKE2S_NO_RESULT));

        let pow_rounds_per_invocation = BLAKE2S_ROUNDS_PER_INVOCAITON as u64;
        // it's good to parallelize
        let num_workers = worker.num_cores as u64;
        worker.scope(usize::MAX, |scope, _| {
            for worker_idx in 0..num_workers {
                let mut input = base_input;
                let result = std::sync::Arc::clone(&result);
                Worker::smart_spawn(scope, worker_idx == num_workers - 1, move |_| {
                    for i in 0..((BLAKE2S_NO_RESULT - 1) / num_workers / pow_rounds_per_invocation)
                    {
                        let base = (worker_idx + i * num_workers) * pow_rounds_per_invocation;
                        let current_flag = result.load(Ordering::Relaxed);
                        if current_flag == BLAKE2S_NO_RESULT {
                            for j in 0..pow_rounds_per_invocation {
                                let challenge_u64 = base + j;
                                // write LE
                                input[BLAKE2S_DIGEST_SIZE_U32_WORDS] = challenge_u64 as u32;
                                input[BLAKE2S_DIGEST_SIZE_U32_WORDS + 1] =
                                    (challenge_u64 >> 32) as u32;
                                let mut state = initial_state;
                                if USE_REDUCED_BLAKE2_ROUNDS {
                                    round_function_reduced_rounds(&mut state, &input);
                                } else {
                                    round_function_full_rounds(&mut state, &input);
                                }
                                let word_to_test = CONFIGURED_IV[0] ^ state[0] ^ state[8];

                                if word_to_test <= (0xffffffff >> pow_bits) {
                                    let _ = result.compare_exchange(
                                        BLAKE2S_NO_RESULT,
                                        challenge_u64,
                                        Ordering::Acquire,
                                        Ordering::Relaxed,
                                    );

                                    break;
                                }
                            }
                        } else {
                            break;
                        }
                    }
                })
            }
        });

        let challenge_u64 = result.load(Ordering::SeqCst);

        // we should just recompute
        let mut new_seed = *seed;
        Self::verify_pow(&mut new_seed, challenge_u64, pow_bits);

        (new_seed, challenge_u64)
    }
}
