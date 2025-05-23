use super::*;
use crate::machine::machine_configurations::minimal_state::MinimalStateRegistersInMemory;
use crate::machine::ops::*;

pub(crate) fn writeback_no_exception<
    F: PrimeField,
    CS: Circuit<F>,
    const ASSUME_TRUSTED_CODE: bool,
    const PERFORM_DELEGATION: bool,
>(
    cs: &mut CS,
    opcodes_are_in_rom: bool,
    flags_source: BasicFlagsSource,
    rd: Constraint<F>,
    update_rd: Constraint<F>,
    rs2_mem_query: RS2ShuffleRamQueryCandidate<F>,
    mem_load_query: ShuffleRamMemQuery,
    mem_store_query: ShuffleRamMemQuery,
    mut memory_queries: Vec<ShuffleRamMemQuery>,
    application_results: Vec<CommonDiffs<F>>,
    default_next_pc: Register<F>,
    opt_ctx: &OptimizationContext<F, CS>,
) -> MinimalStateRegistersInMemory<F> {
    // merge rs2 access and memory load op query
    merge_rs2_and_memload_access_optimized(
        cs,
        opcodes_are_in_rom,
        &flags_source,
        rs2_mem_query,
        mem_load_query,
        &mut memory_queries,
    );

    if ASSUME_TRUSTED_CODE {
        // we can NOT have exceptions coming from the reachable code itself

        // NOTE: if we have the code that is "sound", then any broken invariant will either be:
        // - caught in the opcode, so circuit is unsatisfiable
        // - logical "panic" from the code will finish the program, that will be either UNIMP, or just
        // panic handler that will logically finish the program
        if application_results.iter().all(|el| el.trapped.is_none()) {
            // there are no traps in opcodes support,
            // we can just apply state updates

            // we do not care about predicating state updates below, because if trap happens it's already unsatisfiable circuit

            let new_reg_val = CommonDiffs::select_final_rd_value(cs, &application_results);

            let should_update_reg = update_rd;

            let rd_store_timestamp = if opcodes_are_in_rom {
                RD_STORE_LOCAL_TIMESTAMP
            } else {
                RD_STORE_LOCAL_TIMESTAMP + 1
            };
            assert_eq!(rd_store_timestamp, mem_store_query.local_timestamp_in_cycle);

            let execute_mem_family = flags_source.get_major_flag(MEMORY_COMMON_OP_KEY);
            // NOTE: here we use operation bound flat not in the "branch" of the operation, but
            // it works because we immediately predicate it
            let should_write_mem_if_mem =
                flags_source.get_minor_flag(MEMORY_COMMON_OP_KEY, MEMORY_WRITE_COMMON_OP_KEY);
            let should_write_mem = Boolean::and(&execute_mem_family, &should_write_mem_if_mem, cs);

            let query = update_register_op_as_shuffle_ram_optimized(
                cs,
                rd_store_timestamp,
                rd,
                new_reg_val,
                should_update_reg,
                mem_store_query,
                should_write_mem,
            );

            memory_queries.push(query);

            let new_pc =
                CommonDiffs::select_final_pc_value(cs, &application_results, default_next_pc);

            let final_state = MinimalStateRegistersInMemory { pc: new_pc };

            assert_eq!(memory_queries.len(), 3);

            for mem_query in memory_queries.into_iter() {
                cs.add_shuffle_ram_query(mem_query);
            }

            cs.set_log(&opt_ctx, "EXECUTOR");
            cs.view_log(if PERFORM_DELEGATION {
                "ISA_WITH_DELEGATION"
            } else {
                "ISA_WITHOUT_DELEGATION"
            });

            final_state
        } else {
            todo!()
        }
    } else {
        unreachable!()
    }
}
