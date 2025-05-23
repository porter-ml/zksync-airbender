use super::*;

pub(crate) fn base_isa_state_transition<
    F: PrimeField,
    CS: Circuit<F>,
    const ASSUME_TRUSTED_CODE: bool,
    const OUTPUT_EXACT_EXCEPTIONS: bool,
    const PERFORM_DELEGATION: bool,
>(
    cs: &mut CS,
    opcodes_are_in_rom: bool,
    decode_table_splitting: [usize; 2],
    boolean_keys: DecoderOutputExtraKeysHolder,
) -> (
    MinimalStateRegistersInMemory<F>,
    MinimalStateRegistersInMemory<F>,
) {
    let splitting = decode_table_splitting;

    let mut memory_queries = vec![];

    let initial_state = MinimalStateRegistersInMemory::<F>::initialize(cs);

    // now apply decoding and all the opcodes.
    // Note that we use custom decoder here

    let next_pc = calculate_pc_next_no_overflows(cs, initial_state.pc);
    let pc = *initial_state.get_pc();

    let next_opcode = read_opcode_from_rom(cs, pc);

    // there is one small thing here - if we use CSR processing that doesn't have matching over CSR index,
    // then we must handle UNIMP instruction here that is csrrw x0, cycle, x0
    // Also ROM is padded with UNIMP

    if ASSUME_TRUSTED_CODE {
        if PERFORM_DELEGATION {
            // there will be mtaching over CSR index in the corresponding path, and we do not support "cycle" csr, so we will fail

            // Do nothing
        } else {
            assert_no_unimp(cs, next_opcode);
        }
    } else {
        unimplemented!()
    }

    if let Some(opcode) = next_opcode.get_value_unsigned(cs) {
        println!("Opcode = 0x{:08x}", opcode);
    }

    use crate::machine::decoder::DecoderInput;

    let decoder_input = DecoderInput {
        instruction: next_opcode,
    };
    let (invalid_opcode, base_decoder_output, opcode_format_bits, other_bits) =
        BaseDecoder::decode::<F, CS>(&decoder_input, cs, splitting);

    if ASSUME_TRUSTED_CODE {
        // if opcode is invalid - it's unsatisfiable
        cs.add_constraint_allow_explicit_linear_prevent_optimizations(Constraint::<F>::from(
            invalid_opcode,
        ));
    } else {
        unimplemented!()
    }

    let flags_source = BasicFlagsSource::new(boolean_keys, other_bits);

    let (src1_reg, rs1_mem_query) =
        get_register_op_as_shuffle_ram(cs, base_decoder_output.rs1, opcodes_are_in_rom, true);
    // even though not all the opcodes read first register, we still consider it unconditional
    memory_queries.push(rs1_mem_query);

    let (src2_reg, rs2_mem_query) =
        get_register_op_as_shuffle_ram(cs, base_decoder_output.rs2, opcodes_are_in_rom, false);
    // we will merge rs2 with mem LOAD opcode queries below

    let (src1, src2, update_rd) = BaseDecoder::select_src1_and_src2_values(
        cs,
        &opcode_format_bits,
        src1_reg,
        base_decoder_output.imm,
        src2_reg,
    );

    let mut opt_ctx = OptimizationContext::<F, CS>::new();

    let src1 = RegisterDecompositionWithSign::parse_reg(cs, src1);
    let src2 = RegisterDecompositionWithSign::parse_reg(cs, src2);

    let decoder_output = BasicDecodingResultWithSigns {
        pc_next: next_pc,
        src1,
        src2,
        imm: base_decoder_output.imm,
        funct3: base_decoder_output.funct3,
        funct12: base_decoder_output.funct12,
    };

    cs.set_log(&opt_ctx, "DECODER");
    let mut application_results = Vec::<CommonDiffs<F>>::with_capacity(32);

    let application_result = AddOp::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "ADD");

    let application_result = SubOp::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "SUB");

    let application_result = LuiOp::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "LUI");

    let application_result = AuiPc::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "AUIPC");

    let application_result = BinaryOp::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "BINARY");

    let application_result = MulOp::<true>::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "MUL");

    let application_result =
        DivRemOp::<true>::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
            cs,
            &initial_state,
            &decoder_output,
            &flags_source,
            &mut opt_ctx,
        );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "DIVREM");

    let application_result =
        ConditionalOp::<true>::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
            cs,
            &initial_state,
            &decoder_output,
            &flags_source,
            &mut opt_ctx,
        );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "CONDITIONAL");

    let application_result =
        ShiftOp::<true, false>::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
            cs,
            &initial_state,
            &decoder_output,
            &flags_source,
            &mut opt_ctx,
        );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "SHIFT_SRA_ROT");

    let application_result = JumpOp::apply::<_, ASSUME_TRUSTED_CODE, OUTPUT_EXACT_EXCEPTIONS>(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "JUMP");

    let mut mem_op_queries = vec![];
    let application_result = MemoryOp::<true, true>::apply_with_mem_access::<
        _,
        ASSUME_TRUSTED_CODE,
        OUTPUT_EXACT_EXCEPTIONS,
    >(
        cs,
        &initial_state,
        &decoder_output,
        &flags_source,
        &mut opt_ctx,
        &mut mem_op_queries,
    );
    application_results.push(application_result);
    cs.set_log(&opt_ctx, "MEMORY");

    assert_eq!(mem_op_queries.len(), 2);
    let mem_load_query = mem_op_queries[0];
    let mem_store_query = mem_op_queries[1];

    if PERFORM_DELEGATION == false {
        // CSR operation must be hand implemented for most of the machines, even though we can declare support of it in the opcode
        let application_result = apply_non_determinism_csr_only_assuming_no_unimp::<
            _,
            _,
            _,
            _,
            _,
            _,
            false,
            false,
            false,
            ASSUME_TRUSTED_CODE,
            OUTPUT_EXACT_EXCEPTIONS,
        >(
            cs,
            &initial_state,
            &decoder_output,
            &flags_source,
            &mut opt_ctx,
        );
        application_results.push(application_result);
    } else {
        let application_result = apply_csr_with_delegation::<
            _,
            _,
            _,
            _,
            _,
            _,
            false,
            false,
            false,
            ASSUME_TRUSTED_CODE,
            OUTPUT_EXACT_EXCEPTIONS,
        >(
            cs,
            &initial_state,
            &decoder_output,
            &flags_source,
            &mut opt_ctx,
        );
        application_results.push(application_result);
    };
    cs.set_log(&opt_ctx, "CSR");

    // finish with optimizer, as we do not have any "branching" below
    opt_ctx.enforce_all(cs);
    // drop(opt_ctx);

    // now it's time to merge state

    assert!(OUTPUT_EXACT_EXCEPTIONS == false);

    // merge rs2 access and memory load op query
    merge_rs2_and_memload_access(
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

            let query = update_register_op_as_shuffle_ram(
                cs,
                rd_store_timestamp,
                base_decoder_output.rd,
                new_reg_val,
                should_update_reg,
                mem_store_query,
                should_write_mem,
            );

            memory_queries.push(query);

            let new_pc = CommonDiffs::select_final_pc_value(cs, &application_results);

            let final_state = MinimalStateRegistersInMemory { pc: new_pc };

            assert_eq!(memory_queries.len(), 3);

            for mem_query in memory_queries.into_iter() {
                cs.add_shuffle_ram_query(mem_query);
            }

            cs.set_log(&opt_ctx, "EXECUTOR");
            cs.view_log(if PERFORM_DELEGATION {
                "FULL_ISA_WITH_DELEGATION"
            } else {
                "FULL_ISA"
            });
            (initial_state, final_state)
        } else {
            todo!()
        }
    } else {
        // trap handling
        let trap_in_instruction = {
            let mut exec_flags = Vec::with_capacity(application_results.len());
            let mut trap_flags = Vec::with_capacity(application_results.len());

            for el in application_results.iter() {
                let Some(trapped) = el.trapped else {
                    continue;
                };
                let exec_flag = el.exec_flag;

                trap_flags.push(trapped);
                exec_flags.push(exec_flag);
            }

            let trap_in_instruction =
                Boolean::choose_from_orthogonal_flags::<F, CS>(cs, &exec_flags, &trap_flags);

            trap_in_instruction
        };

        if trap_in_instruction.get_value(cs).unwrap_or(false) {
            println!("Trapped from instruction");
        }

        // if trap happened - it's unsatisfiable
        cs.add_constraint_allow_explicit_linear_prevent_optimizations(Constraint::<F>::from(
            trap_in_instruction,
        ));

        cs.require_invariant(
            trap_in_instruction.get_variable().unwrap(),
            Invariant::Substituted((Placeholder::Trapped, 0)),
        );

        unreachable!()
    }
}
