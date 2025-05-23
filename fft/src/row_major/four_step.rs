use super::*;

pub fn parallel_row_major_full_line_quasi_six_step_ifft<const N: usize, A: Allocator + Clone>(
    trace_columns: &mut RowMajorTrace<Mersenne31Field, N, A>,
    precomputations: &[Mersenne31Complex],
    worker: &Worker,
) {
    assert!(N >= CACHE_LINE_MULTIPLE);
    let trace_len = trace_columns.len();
    assert!(trace_len.is_power_of_two());

    if trace_len == 1 {
        return;
    }

    // we want to have multiple stages, each using all many threads as possible,
    // and each thread working on radix-2 (for now), working on the full row via unrolled loop over some number
    // of elements each time, and using some prefetch

    if trace_len >= 16 {
        assert_eq!(precomputations.len() * 2, trace_len);
    }

    let n = trace_len;

    let log_n = n.trailing_zeros();
    let row_width = trace_columns.width();
    let row_offset = trace_columns.padded_width;

    // we will do small-sized full row FFT, but we assume that all data must be in L1 at that time
    let mut num_rows_to_fit_into_l1 =
        L1_CACHE_SIZE / (core::mem::size_of::<Mersenne31Field>() * trace_columns.padded_width);
    if num_rows_to_fit_into_l1.is_power_of_two() == false {
        num_rows_to_fit_into_l1 = num_rows_to_fit_into_l1.next_power_of_two() / 2;
    }

    let single_stage_size = num_rows_to_fit_into_l1.trailing_zeros();

    let mut work_size = log_n;
    let mut fft_plan = vec![];
    while work_size > 0 {
        let unit_size = if single_stage_size <= work_size {
            single_stage_size
        } else {
            work_size
        };

        work_size -= unit_size;
        fft_plan.push(unit_size);
    }

    let barriers: Vec<_> = (0..fft_plan.len())
        .map(|_| Barrier::new(worker.num_cores))
        .collect();

    let unroll_stage_offset = 64;
    let num_unrolled_cycles = row_width / unroll_stage_offset;
    // we will go formally out of bounds for logically defined row, but why not, if it helps performance
    let padded_remainder = row_offset % unroll_stage_offset;
    let num_remainder_passes = padded_remainder / CACHE_LINE_MULTIPLE;

    let barriers_ref = &barriers;
    let fft_plan_ref = &fft_plan;

    let trace_start_ptr = PointerWrapper(trace_columns.ptr);

    let num_stages = fft_plan.len();
    let num_cores = worker.num_cores;

    unsafe {
        worker.scope(1, |scope, _| {
            for core_idx in 0..num_cores {
                Worker::smart_spawn(scope, core_idx == num_cores - 1, move |_| {
                    let ptr = trace_start_ptr;

                    for stage in 0..num_stages {
                        let stage_size = fft_plan_ref[stage];
                        let fft_size = 1 << stage_size;
                        debug_assert_eq!(trace_len % fft_size, 0);
                        let num_independent_ffts = trace_len / fft_size;

                        let geometry =
                            Worker::get_geometry_for_num_cores(num_cores, num_independent_ffts);
                        // here we enumerate over M units, each of those being of size `fft_size`, and get chunk over M
                        let chunk_start = geometry.get_chunk_start_pos(core_idx);
                        let chunk_size = geometry.get_chunk_size(core_idx);
                        let num_pairs = fft_size / 2;

                        let mut subwork_idx = chunk_start;

                        // here we just do many small independent FFT, and then instead of doing physical memory transpose as in six-step, we just remunerate indexes
                        for _ in 0..chunk_size {
                            debug_assert!(subwork_idx < num_independent_ffts);
                            // we have FFT "works" of some size (like 128 for M1),
                            // and we need to determine indexes in our holder that will correspond to that FFT
                            // without performing actually transpose/swaps/etc in memory

                            // Here we do FFT of size `fft_size`

                            // We do radix 2 for now
                            // We will model remappign later

                            let mut j = 0;
                            let mut idx = subwork_idx * fft_size;
                            while j < num_pairs {
                                // it's a remap
                                let distance = num_pairs;
                                let (u_absolute_index, v_absolute_index) = (idx, idx + distance);

                                // it's an FFT routine itself
                                let mut u_ptr = ptr.0.add(row_offset * u_absolute_index);
                                let mut v_ptr = ptr.0.add(row_offset * v_absolute_index);

                                #[cfg(target_arch = "aarch64")]
                                {
                                    prefetch_next_line(u_ptr);
                                    prefetch_next_line(v_ptr);
                                    prefetch_next_line(u_ptr.add(CACHE_LINE_MULTIPLE));
                                    prefetch_next_line(v_ptr.add(CACHE_LINE_MULTIPLE));
                                }

                                for cycle in 0..num_unrolled_cycles {
                                    let _last_cycle = cycle == num_unrolled_cycles - 1;
                                    // prefetch next while we work here
                                    let u_ptr_next = u_ptr.add(unroll_stage_offset);
                                    let v_ptr_next = v_ptr.add(unroll_stage_offset);

                                    #[cfg(target_arch = "aarch64")]
                                    {
                                        debug_assert_eq!(
                                            CACHE_LINE_MULTIPLE * 2,
                                            unroll_stage_offset
                                        );
                                        prefetch_next_line(u_ptr_next);
                                        prefetch_next_line(u_ptr_next);
                                        if _last_cycle == false {
                                            prefetch_next_line(u_ptr_next.add(CACHE_LINE_MULTIPLE));
                                            prefetch_next_line(v_ptr_next.add(CACHE_LINE_MULTIPLE));
                                        }
                                    }

                                    // to have it in front of us here
                                    debug_assert_eq!(unroll_stage_offset, 64);

                                    seq!(N in 0..64 {
                                        let u_el_ptr = u_ptr.add(N);
                                        let v_el_ptr = v_ptr.add(N);

                                        let u = u_el_ptr.read();
                                        let v = v_el_ptr.read();

                                        let mut add_res = u;
                                        let mut sub_res = u;
                                        add_res.add_assign(&v);
                                        sub_res.sub_assign(&v);

                                        u_el_ptr.write(add_res);
                                        v_el_ptr.write(sub_res);
                                    });

                                    u_ptr = u_ptr_next;
                                    v_ptr = v_ptr_next;
                                }

                                if padded_remainder != 0 {
                                    debug_assert_eq!(padded_remainder % CACHE_LINE_MULTIPLE, 0);

                                    #[cfg(target_arch = "aarch64")]
                                    {
                                        debug_assert_eq!(
                                            CACHE_LINE_MULTIPLE * 2,
                                            unroll_stage_offset
                                        );
                                        debug_assert_eq!(CACHE_LINE_MULTIPLE, 32);
                                        debug_assert_eq!(num_remainder_passes, 1);

                                        seq!(N in 0..32 {
                                            let u_el_ptr = u_ptr.add(N);
                                            let v_el_ptr = v_ptr.add(N);

                                            let u = u_el_ptr.read();
                                            let v = v_el_ptr.read();

                                            let mut add_res = u;
                                            let mut sub_res = u;
                                            add_res.add_assign(&v);
                                            sub_res.sub_assign(&v);

                                            u_el_ptr.write(add_res);
                                            v_el_ptr.write(sub_res);
                                        });
                                    }

                                    #[cfg(not(target_arch = "aarch64"))]
                                    {
                                        debug_assert!(num_remainder_passes <= 3);

                                        debug_assert_eq!(
                                            CACHE_LINE_MULTIPLE * 4,
                                            unroll_stage_offset
                                        );
                                        debug_assert_eq!(CACHE_LINE_MULTIPLE, 16);
                                        for _ in 0..num_remainder_passes {
                                            let u_ptr_next = u_ptr.add(CACHE_LINE_MULTIPLE);
                                            let v_ptr_next = v_ptr.add(CACHE_LINE_MULTIPLE);
                                            prefetch_next_line(u_ptr_next);
                                            prefetch_next_line(u_ptr_next);

                                            seq!(N in 0..16 {
                                                let u_el_ptr = u_ptr.add(N);
                                                let v_el_ptr = v_ptr.add(N);

                                                let u = u_el_ptr.read();
                                                let v = v_el_ptr.read();

                                                let mut add_res = u;
                                                let mut sub_res = u;
                                                add_res.add_assign(&v);
                                                sub_res.sub_assign(&v);

                                                u_el_ptr.write(add_res);
                                                v_el_ptr.write(sub_res);
                                            });

                                            u_ptr = u_ptr_next;
                                            v_ptr = v_ptr_next;
                                        }
                                    }
                                }

                                j += 1;
                                idx += 1;
                            }

                            subwork_idx += 1;
                        }

                        // our work of M ffts of isize `fft size is complete`
                        barriers_ref[stage].wait();

                        // now we should multiply by twiddles, but we should merge it with next stage
                    }
                });
            }
        });
    }
}
