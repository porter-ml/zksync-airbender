use ::field::*;
use blake2s_u32::*;
use fft::bitreverse_enumeration_inplace;
use fft::GoodAllocator;
use std::alloc::Global;
use trace_holder::*;
use worker::Worker;



#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Blake2sU32MerkleTreeCap {
    pub cap: Vec<[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS]>,
}

#[derive(Clone, Debug)]
pub struct Blake2sU32MerkleTreeWithCap<A: GoodAllocator = Global> {
    pub cap_size: usize,
    pub leaf_hashes: Vec<[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS], A>,
    pub node_hashes_enumerated_from_leafs: Vec<Vec<[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS], A>>,
}

impl<A: GoodAllocator> Blake2sU32MerkleTreeWithCap<A> {
    pub fn construct_for_coset<const N: usize>(
        trace: &RowMajorTrace<Mersenne31Field, N, A>,
        cap_size: usize,
        bitreverse: bool,
        worker: &Worker,
    ) -> Self {
        debug_assert!(cap_size > 0);
        debug_assert!(cap_size.is_power_of_two());

        #[cfg(feature = "timing_logs")]
        let now = std::time::Instant::now();

        let tree_size = trace.len();
        debug_assert!(tree_size.is_power_of_two());
        let tree_depth = tree_size.trailing_zeros();
        let layers_to_skip = cap_size.trailing_zeros();
        #[cfg(feature = "timing_logs")]
        let elements_per_leaf = trace.width();
        debug_assert!(tree_size >= cap_size);

        // simplest job ever - compute by layers with parallelism
        // To prevent to complex parallelism we will work over each individual coset

        let mut leaf_hashes = Vec::with_capacity_in(tree_size, A::default());

        unsafe {
            worker.scope(tree_size, |scope, geometry| {
                let mut dst = &mut leaf_hashes.spare_capacity_mut()[..tree_size];
                for i in 0..geometry.len() {
                    let chunk_size = geometry.get_chunk_size(i);
                    let chunk_start = geometry.get_chunk_start_pos(i);

                    let range = chunk_start..(chunk_start + chunk_size);
                    let mut trace_view = trace.row_view(range.clone());
                    let (dst_chunk, rest) = dst.split_at_mut_unchecked(chunk_size);
                    dst = rest;

                    scope.spawn(move |_| {
                        let mut dst_ptr = dst_chunk.as_mut_ptr();
                        let mut hasher = Blake2sState::new();
                        for _i in 0..chunk_size {
                            hasher.reset();
                            let trace_view_row = trace_view.current_row();
                            let only_full_rounds =
                                trace_view_row.len() % BLAKE2S_BLOCK_SIZE_U32_WORDS == 0;
                            let num_full_roudns =
                                trace_view_row.len() / BLAKE2S_BLOCK_SIZE_U32_WORDS;
                            let mut array_chunks =
                                trace_view_row.array_chunks::<BLAKE2S_BLOCK_SIZE_U32_WORDS>();

                            let write_into = (&mut *dst_ptr).assume_init_mut();
                            for i in 0..num_full_roudns {
                                let last_round = i == num_full_roudns - 1;
                                let chunk = array_chunks.next().unwrap_unchecked();

                                let block = chunk.map(|el| el.to_reduced_u32());

                                if last_round && only_full_rounds {
                                    hasher.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                                        &block,
                                        BLAKE2S_BLOCK_SIZE_U32_WORDS,
                                        write_into,
                                    );
                                } else {
                                    hasher.absorb::<USE_REDUCED_BLAKE2_ROUNDS>(&block);
                                }
                            }

                            if only_full_rounds == false {
                                let remainder = array_chunks.remainder();
                                let mut block = [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS];
                                let len = remainder.len();
                                for i in 0..len {
                                    block[i] = remainder[i].to_reduced_u32();
                                }
                                hasher.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                                    &block, len, write_into,
                                );
                            }

                            dst_ptr = dst_ptr.add(1);
                            trace_view.advance_row();
                        }
                    });
                }

                assert!(dst.is_empty());
            });

            leaf_hashes.set_len(tree_size)
        };

        #[cfg(feature = "timing_logs")]
        println!(
            "Merkle tree of size 2^{} leaf hashes taken {:?} for {} elements per leaf",
            tree_size.trailing_zeros(),
            now.elapsed(),
            elements_per_leaf,
        );

        let num_layers_to_construct = tree_depth - layers_to_skip;

        if bitreverse {
            bitreverse_enumeration_inplace(&mut leaf_hashes);
        }

        Self::continue_from_leaf_hashes(leaf_hashes, num_layers_to_construct, cap_size, worker)
    }

    pub fn construct_separate_for_coset<const N: usize>(
        trace: &RowMajorTrace<Mersenne31Field, N, A>,
        separators: &[usize],
        cap_size: usize,
        bitreverse: bool,
        worker: &Worker,
    ) -> Vec<Self> {
        debug_assert!(cap_size > 0);
        debug_assert!(cap_size.is_power_of_two());

        assert!(
            *separators
                .last()
                .expect("Should contain at least one separator")
                <= trace.width(),
            "Separator is out of bounds"
        );
        for idx in 0..separators.len() - 1 {
            assert!(
                separators[idx] < separators[idx + 1],
                "Separators are not sorted"
            );
        }

        #[cfg(feature = "timing_logs")]
        let now = std::time::Instant::now();

        let tree_size = trace.len();
        debug_assert!(tree_size.is_power_of_two());
        let tree_depth = tree_size.trailing_zeros();
        let layers_to_skip = cap_size.trailing_zeros();
        #[cfg(feature = "timing_logs")]
        let elements_per_leaf = trace.width();
        debug_assert!(tree_size >= cap_size);

        // simplest job ever - compute by layers with parallelism
        // To prevent to complex parallelism we will work over each individual coset

        let mut chunk_widths = vec![separators[0]];
        for i in 0..separators.len() - 1 {
            chunk_widths.push(separators[i + 1] - separators[i]);
        }

        let mut leaf_hashes: Vec<_> = (0..separators.len())
            .map(|_| Vec::with_capacity_in(tree_size, A::default()))
            .collect();

        unsafe {
            worker.scope(tree_size, |scope, geometry| {
                let mut dst: Vec<_> = leaf_hashes
                    .iter_mut()
                    .map(|lh| &mut lh.spare_capacity_mut()[..tree_size])
                    .collect();
                for i in 0..geometry.len() {
                    let chunk_size = geometry.get_chunk_size(i);
                    let chunk_start = geometry.get_chunk_start_pos(i);

                    let range = chunk_start..(chunk_start + chunk_size);
                    let mut trace_view = trace.row_view(range.clone());
                    let chunk_widths_clone = chunk_widths.clone();

                    let mut rest_chunks = vec![];
                    let mut dst_chunks = vec![];

                    dst.into_iter().for_each(|dst| {
                        let (dst_chunk, rest) = dst.split_at_mut_unchecked(chunk_size);
                        dst_chunks.push(dst_chunk);
                        rest_chunks.push(rest);
                    });
                    dst = rest_chunks;

                    scope.spawn(move |_| {
                        let mut dst_ptrs: Vec<_> =
                            dst_chunks.iter_mut().map(|dst| dst.as_mut_ptr()).collect();
                        let mut hasher = Blake2sState::new();
                        for _i in 0..chunk_size {
                            let mut trace_view_row = trace_view.current_row();
                            for j in 0..dst_ptrs.len() {
                                hasher.reset();

                                let (cur_trace_view_row, rest) =
                                    trace_view_row.split_at_mut_unchecked(chunk_widths_clone[j]);
                                trace_view_row = rest;

                                let only_full_rounds =
                                    chunk_widths_clone[j] % BLAKE2S_BLOCK_SIZE_U32_WORDS == 0;
                                let num_full_roudns =
                                    chunk_widths_clone[j] / BLAKE2S_BLOCK_SIZE_U32_WORDS;
                                let mut array_chunks = cur_trace_view_row
                                    .array_chunks::<BLAKE2S_BLOCK_SIZE_U32_WORDS>(
                                );

                                let write_into = (&mut *dst_ptrs[j]).assume_init_mut();
                                for i in 0..num_full_roudns {
                                    let last_round = i == num_full_roudns - 1;
                                    let chunk = array_chunks.next().unwrap_unchecked();

                                    let block = chunk.map(|el| el.to_reduced_u32());

                                    if last_round && only_full_rounds {
                                        hasher.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                                            &block,
                                            BLAKE2S_BLOCK_SIZE_U32_WORDS,
                                            write_into,
                                        );
                                    } else {
                                        hasher.absorb::<USE_REDUCED_BLAKE2_ROUNDS>(&block);
                                    }
                                }

                                if only_full_rounds == false {
                                    let remainder = array_chunks.remainder();
                                    let mut block = [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS];
                                    let len = remainder.len();
                                    for i in 0..len {
                                        block[i] = remainder[i].to_reduced_u32();
                                    }
                                    hasher.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                                        &block, len, write_into,
                                    );
                                }
                            }

                            dst_ptrs
                                .iter_mut()
                                .for_each(|dst_ptr| *dst_ptr = dst_ptr.add(1));
                            trace_view.advance_row();
                        }
                    });
                }

                assert!(dst.iter().all(|d| d.is_empty()));
            });

            leaf_hashes.iter_mut().for_each(|lh| lh.set_len(tree_size));
        };

        #[cfg(feature = "timing_logs")]
        println!(
            "Merkle tree of size 2^{} leaf hashes taken {:?} for {} elements per leaf",
            tree_size.trailing_zeros(),
            now.elapsed(),
            elements_per_leaf,
        );

        let num_layers_to_construct = tree_depth - layers_to_skip;

        if bitreverse {
            for mut lh in leaf_hashes.iter_mut() {
                bitreverse_enumeration_inplace(&mut lh);
            }
        }

        leaf_hashes
            .into_iter()
            .map(|lh| {
                Self::continue_from_leaf_hashes(lh, num_layers_to_construct, cap_size, worker)
            })
            .collect()
    }

    pub fn construct_for_column_major_coset(
        trace: &ColumnMajorTrace<Mersenne31Quartic, A>,
        combine_by: usize,
        cap_size: usize,
        bitreverse: bool,
        worker: &Worker,
    ) -> Self {
        assert_eq!(
            trace.width(),
            1,
            "we only support it for narrow traces for now"
        );
        assert!(combine_by.is_power_of_two());
        assert_eq!(trace.len() % combine_by, 0);

        debug_assert!(cap_size > 0);
        debug_assert!(cap_size.is_power_of_two());

        #[cfg(feature = "timing_logs")]
        let now = std::time::Instant::now();

        let tree_size = trace.len() / combine_by;
        assert!(tree_size.is_power_of_two());
        let tree_depth = tree_size.trailing_zeros();
        let layers_to_skip = cap_size.trailing_zeros();
        #[cfg(feature = "timing_logs")]
        let elements_per_leaf = trace.width();
        assert!(tree_size >= cap_size);
        let leaf_width_in_field_elements = combine_by * trace.width() * 4;

        let num_full_roudns = leaf_width_in_field_elements / BLAKE2S_BLOCK_SIZE_U32_WORDS;
        let remainder = leaf_width_in_field_elements % BLAKE2S_BLOCK_SIZE_U32_WORDS;
        let only_full_rounds = remainder == 0;

        // simplest job ever - compute by layers with parallelism
        // To prevent to complex parallelism we will work over each individual coset

        let mut leaf_hashes = Vec::with_capacity_in(tree_size, A::default());
        let source_column = trace.columns_iter().next().unwrap();

        unsafe {
            worker.scope(tree_size, |scope, geometry| {
                let mut dst = &mut leaf_hashes.spare_capacity_mut()[..tree_size];
                for i in 0..geometry.len() {
                    let chunk_size = geometry.get_chunk_size(i);
                    let chunk_start = geometry.get_chunk_start_pos(i);

                    let _dst_range = chunk_start..(chunk_start + chunk_size);
                    let src_range =
                        chunk_start * combine_by..(chunk_start + chunk_size) * combine_by;
                    let (dst_chunk, rest) = dst.split_at_mut_unchecked(chunk_size);
                    dst = rest;

                    scope.spawn(move |_| {
                        let mut dst_ptr = dst_chunk.as_mut_ptr();
                        let source_chunk = &source_column[src_range];
                        assert_eq!(source_chunk.len(), chunk_size * combine_by);
                        let mut src_ptr = source_chunk.as_ptr();
                        let mut hasher = Blake2sState::new();
                        for _i in 0..chunk_size {
                            hasher.reset();
                            let src_chunk = core::slice::from_raw_parts(
                                src_ptr.cast::<Mersenne31Field>(),
                                combine_by * 4,
                            );
                            let mut array_chunks =
                                src_chunk.array_chunks::<BLAKE2S_BLOCK_SIZE_U32_WORDS>();
                            debug_assert_eq!(src_chunk.len(), leaf_width_in_field_elements);

                            let write_into = (&mut *dst_ptr).assume_init_mut();
                            for i in 0..num_full_roudns {
                                let last_round = i == num_full_roudns - 1;
                                let chunk = array_chunks.next().unwrap_unchecked();

                                let block = chunk.map(|el| el.to_reduced_u32());

                                if last_round && only_full_rounds {
                                    hasher.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                                        &block,
                                        BLAKE2S_BLOCK_SIZE_U32_WORDS,
                                        write_into,
                                    );
                                } else {
                                    hasher.absorb::<USE_REDUCED_BLAKE2_ROUNDS>(&block);
                                }
                            }

                            if only_full_rounds == false {
                                let remainder = array_chunks.remainder();
                                let mut block = [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS];
                                let len = remainder.len();
                                for i in 0..len {
                                    block[i] = remainder[i].to_reduced_u32();
                                }
                                hasher.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                                    &block, len, write_into,
                                );
                            }

                            dst_ptr = dst_ptr.add(1);
                            src_ptr = src_ptr.add(combine_by);
                        }
                    });
                }

                assert!(dst.is_empty());
            });

            leaf_hashes.set_len(tree_size)
        };

        #[cfg(feature = "timing_logs")]
        println!(
            "Merkle tree of size 2^{} leaf hashes taken {:?} for {} elements per leaf",
            tree_size.trailing_zeros(),
            now.elapsed(),
            elements_per_leaf,
        );

        let num_layers_to_construct = tree_depth - layers_to_skip;

        if bitreverse {
            bitreverse_enumeration_inplace(&mut leaf_hashes);
        }

        Self::continue_from_leaf_hashes(leaf_hashes, num_layers_to_construct, cap_size, worker)
    }

    fn continue_from_leaf_hashes(
        leaf_hashes: Vec<[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS], A>,
        num_layers_to_construct: u32,
        cap_size: usize,
        worker: &Worker,
    ) -> Self {
        if num_layers_to_construct == 0 {
            #[cfg(feature = "debug_logs")]
            println!("Do not need to construct nodes, can use leaf hashes directly to form a cap");
            assert_eq!(cap_size, leaf_hashes.len());
            return Self {
                cap_size,
                leaf_hashes,
                node_hashes_enumerated_from_leafs: Vec::new(),
            };
        }

        #[cfg(feature = "timing_logs")]
        let now = std::time::Instant::now();
        assert!(num_layers_to_construct > 0);

        let mut previous = &leaf_hashes[..];
        let mut node_hashes_enumerated_from_leafs =
            Vec::with_capacity(num_layers_to_construct as usize);
        for _ in 0..num_layers_to_construct {
            let next_layer_len = previous.len() / 2;
            debug_assert!(next_layer_len > 0);
            debug_assert!(next_layer_len.is_power_of_two());
            let mut new_layer_node_hashes: Vec<[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS], A> =
                Vec::with_capacity_in(next_layer_len, A::default());

            unsafe {
                worker.scope(next_layer_len, |scope, geometry| {
                    let mut dst = &mut new_layer_node_hashes.spare_capacity_mut()[..next_layer_len];
                    let mut src = previous;
                    for i in 0..geometry.len() {
                        let chunk_size = geometry.get_chunk_size(i);

                        let (dst_chunk, rest) = dst.split_at_mut_unchecked(chunk_size);
                        dst = rest;
                        let (src_chunk, rest) = src.split_at_unchecked(chunk_size * 2);
                        src = rest;

                        scope.spawn(move |_| {
                            let mut dst_ptr = dst_chunk.as_mut_ptr();
                            // easier to use pointers
                            let mut src_ptr = src_chunk
                                .as_ptr()
                                .cast::<[u32; BLAKE2S_BLOCK_SIZE_U32_WORDS]>();
                            for _i in 0..chunk_size {
                                let read_from = &*src_ptr;
                                let write_into = (&mut *dst_ptr).assume_init_mut();
                                Blake2sState::compress_two_to_one::<USE_REDUCED_BLAKE2_ROUNDS>(
                                    read_from, write_into,
                                );

                                src_ptr = src_ptr.add(1);
                                dst_ptr = dst_ptr.add(1);
                            }
                        });
                    }
                });

                new_layer_node_hashes.set_len(next_layer_len)
            };

            node_hashes_enumerated_from_leafs.push(new_layer_node_hashes);
            previous = node_hashes_enumerated_from_leafs.last().unwrap();
        }

        debug_assert_eq!(previous.len(), cap_size);

        #[cfg(feature = "timing_logs")]
        println!(
            "Nodes construction of size 2^{} taken {:?}",
            leaf_hashes.len().trailing_zeros(),
            now.elapsed()
        );

        Self {
            cap_size,
            leaf_hashes,
            node_hashes_enumerated_from_leafs,
        }
    }

    pub fn get_cap_ref(&self) -> &[[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS]] {
        let output = if let Some(cap) = self.node_hashes_enumerated_from_leafs.last() {
            cap.as_slice()
        } else {
            &self.leaf_hashes
        };

        output
    }

    pub fn get_cap(&self) -> Blake2sU32MerkleTreeCap {
        let output = if let Some(cap) = self.node_hashes_enumerated_from_leafs.last() {
            let mut result = Vec::new();
            result.extend_from_slice(cap);

            result
        } else {
            let mut result = Vec::new();
            result.extend_from_slice(&self.leaf_hashes);

            result
        };

        Blake2sU32MerkleTreeCap { cap: output }
    }

    pub fn get_proof<C: GoodAllocator>(
        &self,
        idx: usize,
    ) -> (
        [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
        Vec<[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS], C>,
    ) {
        let depth = self.node_hashes_enumerated_from_leafs.len(); // we do not need the element of the cap
        let mut result = Vec::with_capacity_in(depth, C::default());
        let mut idx = idx;
        let this_el_leaf_hash = self.leaf_hashes[idx];
        for i in 0..depth {
            let pair_idx = idx ^ 1;
            let proof_element = if i == 0 {
                self.leaf_hashes[pair_idx]
            } else {
                self.node_hashes_enumerated_from_leafs[i - 1][pair_idx]
            };

            result.push(proof_element);
            idx >>= 1;
        }

        (this_el_leaf_hash, result)
    }

    pub fn verify_proof_over_cap(
        _proof: &[[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS]],
        _cap: &[[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS]],
        _leaf_hash: [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
        _idx: usize,
    ) -> bool {
        todo!();

        // let mut idx = idx;
        // let mut current = leaf_hash;
        // for proof_el in proof.iter() {
        //     if idx & 1 == 0 {
        //         current = H::hash_into_node(&current, proof_el, 0);
        //     } else {
        //         current = H::hash_into_node(proof_el, &current, 0);
        //     }

        //     idx >>= 1;
        // }

        // let cap_el = &cap[idx];
        // H::normalize_output(&mut current);

        // cap_el == &current
    }

    pub fn dump_caps(caps: &[Self]) -> Vec<Blake2sU32MerkleTreeCap> {
        let mut result = Vec::with_capacity(caps.len());
        for el in caps.iter() {
            result.push(el.get_cap());
        }

        result
    }
}
