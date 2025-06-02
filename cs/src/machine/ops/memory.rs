use super::*;

pub const MEMORY_COMMON_OP_KEY: DecoderMajorInstructionFamilyKey =
    DecoderMajorInstructionFamilyKey("LW/SW/LH/LHU/SH/LB/LBU/SB");
pub const MEMORY_WRITE_COMMON_OP_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("SW/SH/SB");
pub const ACCESS_BYTE_OP_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("LB/LBU/SB");
pub const ACCESS_HALF_WORD_OP_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("LH/LHU/SH");
pub const ACCESS_WORD_OP_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("LW/SW");
pub const SIGN_EXTEND_ON_LOAD_OP_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("LB/LW");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryOp<const SUPPORT_SIGNED: bool, const SUPPORT_LESS_THAN_WORD: bool>;

impl<const SUPPORT_SIGNED: bool, const SUPPORT_LESS_THAN_WORD: bool> DecodableMachineOp
    for MemoryOp<SUPPORT_SIGNED, SUPPORT_LESS_THAN_WORD>
{
    fn define_decoder_subspace(
        &self,
        opcode: u8,
        func3: u8,
        func7: u8,
    ) -> Result<
        (
            InstructionType,
            DecoderMajorInstructionFamilyKey,
            &'static [DecoderInstructionVariantsKey],
        ),
        (),
    > {
        let params = match (opcode, func3, func7) {
            (OPERATION_LOAD, 0b000, _) if SUPPORT_SIGNED & SUPPORT_LESS_THAN_WORD => {
                // LB
                (
                    InstructionType::IType,
                    MEMORY_COMMON_OP_KEY,
                    &[ACCESS_BYTE_OP_KEY, SIGN_EXTEND_ON_LOAD_OP_KEY][..],
                )
            }
            (OPERATION_LOAD, 0b001, _) if SUPPORT_SIGNED & SUPPORT_LESS_THAN_WORD => {
                // LH
                (
                    InstructionType::IType,
                    MEMORY_COMMON_OP_KEY,
                    &[ACCESS_HALF_WORD_OP_KEY, SIGN_EXTEND_ON_LOAD_OP_KEY][..],
                )
            }
            (OPERATION_LOAD, 0b010, _) => {
                // LW
                if SUPPORT_LESS_THAN_WORD {
                    (
                        InstructionType::IType,
                        MEMORY_COMMON_OP_KEY,
                        &[ACCESS_WORD_OP_KEY][..],
                    )
                } else {
                    (InstructionType::IType, MEMORY_COMMON_OP_KEY, &[][..])
                }
            }
            (OPERATION_LOAD, 0b100, _) if SUPPORT_LESS_THAN_WORD => {
                // LBU
                (
                    InstructionType::IType,
                    MEMORY_COMMON_OP_KEY,
                    &[ACCESS_BYTE_OP_KEY][..],
                )
            }
            (OPERATION_LOAD, 0b101, _) if SUPPORT_LESS_THAN_WORD => {
                // LHU
                (
                    InstructionType::IType,
                    MEMORY_COMMON_OP_KEY,
                    &[ACCESS_HALF_WORD_OP_KEY][..],
                )
            }
            (OPERATION_STORE, 0b000, _) if SUPPORT_LESS_THAN_WORD => {
                // SB
                (
                    InstructionType::SType,
                    MEMORY_COMMON_OP_KEY,
                    &[ACCESS_BYTE_OP_KEY, MEMORY_WRITE_COMMON_OP_KEY][..],
                )
            }
            (OPERATION_STORE, 0b001, _) if SUPPORT_LESS_THAN_WORD => {
                // SH
                (
                    InstructionType::SType,
                    MEMORY_COMMON_OP_KEY,
                    &[ACCESS_HALF_WORD_OP_KEY, MEMORY_WRITE_COMMON_OP_KEY][..],
                )
            }
            (OPERATION_STORE, 0b010, _) => {
                // SW
                if SUPPORT_LESS_THAN_WORD {
                    (
                        InstructionType::SType,
                        MEMORY_COMMON_OP_KEY,
                        &[ACCESS_WORD_OP_KEY, MEMORY_WRITE_COMMON_OP_KEY][..],
                    )
                } else {
                    (
                        InstructionType::SType,
                        MEMORY_COMMON_OP_KEY,
                        &[MEMORY_WRITE_COMMON_OP_KEY][..],
                    )
                }
            }
            _ => return Err(()),
        };

        Ok(params)
    }
}

impl<
        F: PrimeField,
        ST: BaseMachineState<F>,
        RS: RegisterValueSource<F>,
        DE: DecoderOutputSource<F, RS>,
        BS: IndexableBooleanSet,
        const SUPPORT_SIGNED: bool,
        const SUPPORT_LESS_THAN_WORD: bool,
    > MachineOp<F, ST, RS, DE, BS> for MemoryOp<SUPPORT_SIGNED, SUPPORT_LESS_THAN_WORD>
{
    fn define_used_tables() -> Vec<TableType> {
        if SUPPORT_SIGNED {
            vec![
                TableType::MemoryOffsetGetBits,
                TableType::U16GetSignAndHighByte,
                TableType::U16SelectByteAndGetByteSign,
            ]
        } else {
            vec![TableType::MemoryOffsetGetBits]
        }
    }

    fn apply<
        CS: Circuit<F>,
        const ASSUME_TRUSTED_CODE: bool,
        const OUTPUT_EXACT_EXCEPTIONS: bool,
    >(
        _cs: &mut CS,
        _machine_state: &ST,
        _inputs: &DE,
        _boolean_set: &BS,
        _opt_ctx: &mut OptimizationContext<F, CS>,
    ) -> CommonDiffs<F> {
        panic!("use `apply_with_mem_access` for this opcode")
    }

    fn apply_with_mem_access<
        CS: Circuit<F>,
        const ASSUME_TRUSTED_CODE: bool,
        const OUTPUT_EXACT_EXCEPTIONS: bool,
    >(
        cs: &mut CS,
        _machine_state: &ST,
        inputs: &DE,
        boolean_set: &BS,
        opt_ctx: &mut OptimizationContext<F, CS>,
        memory_queries: &mut Vec<ShuffleRamMemQuery>,
    ) -> CommonDiffs<F> {
        opt_ctx.reset_indexers();

        const NUM_MEMORY_INSNS: usize = 3;

        const MEM_LOAD_LOCAL_TIMESTAMP: usize = RS2_LOAD_LOCAL_TIMESTAMP; // same as RS2
        const MEM_STORE_LOCAL_TIMESTAMP: usize = RD_STORE_LOCAL_TIMESTAMP; // same as rd

        let mem_load_timestamp = if ST::opcodes_are_in_rom() {
            MEM_LOAD_LOCAL_TIMESTAMP
        } else {
            MEM_LOAD_LOCAL_TIMESTAMP + 1
        };

        let mem_store_timestamp = if ST::opcodes_are_in_rom() {
            MEM_STORE_LOCAL_TIMESTAMP
        } else {
            MEM_STORE_LOCAL_TIMESTAMP + 1
        };

        let execute_family = boolean_set.get_major_flag(MEMORY_COMMON_OP_KEY);
        let should_write_mem = Boolean::and(
            &execute_family,
            &boolean_set.get_minor_flag(MEMORY_COMMON_OP_KEY, MEMORY_WRITE_COMMON_OP_KEY),
            cs,
        );

        let src1 = inputs.get_rs1_or_equivalent();
        let src2 = inputs.get_rs2_or_equivalent();

        if execute_family.get_value(cs).unwrap_or(false) {
            println!("MEMORY");
            println!("Address = {:?}", src1.get_register().get_value_unsigned(cs));
        }

        if SUPPORT_LESS_THAN_WORD == true {
            // this is common for FAMILY of memory instructions

            let src1 = src1.get_register();
            let imm = inputs.get_imm();

            let (unaligned_address, _of_flag) =
                opt_ctx.append_add_relation(src1, imm, execute_family, cs);

            let [bit_0, bit_1] = opt_ctx.append_lookup_relation(
                cs,
                &[unaligned_address.0[0].get_variable()],
                TableType::MemoryOffsetGetBits.to_num(),
                execute_family,
            );
            let aligned_address_low_constraint = {
                Constraint::from(unaligned_address.0[0].get_variable())
                    - (Term::from(bit_1) * Term::from(2))
                    - Term::from(bit_0)
            };

            // NOTE: we do NOT cast presumable bits to booleans, as it's under conditional assignment of lookup

            // NOTE: whether it's read or write, we will always read from src1 + imm.
            // Also from the modification above we know that we always read from 0 mod 4, so we are fine
            // also with the structure of our ROM table
            let (source, mem_load_query, address_is_in_ram_range) =
                read_from_shuffle_ram_or_bytecode_no_decomposition_with_ctx(
                    cs,
                    mem_load_timestamp,
                    aligned_address_low_constraint,
                    unaligned_address.0[1],
                    opt_ctx,
                    execute_family,
                );

            // we can not write into ROM
            if ASSUME_TRUSTED_CODE {
                // NOTE: `should_write_mem` always conditioned over execution of the opcode itself
                cs.add_constraint(
                    should_write_mem.get_terms()
                        * (Term::from(1) - Term::from(address_is_in_ram_range)),
                );
            } else {
                // we should trap maybe
                todo!()
            }

            // if we will do STORE, then it'll be the value
            let val_to_store = src2.get_register_with_decomposition();

            let mem_read_words: [_; 2] =
                std::array::from_fn(|idx: usize| Term::from(source.0[idx]));
            let word_to_store = val_to_store.u16_limbs[0];
            let byte_to_store = val_to_store.u8_decomposition[0];

            let shift_8 = Term::from(1 << 8);

            use crate::devices::aux_data::*;

            // let mut aux_data_arr = Vec::with_capacity(NUM_MEMORY_INSNS);
            let mut aux_data_arr_reduced = Vec::with_capacity(NUM_MEMORY_INSNS);

            // Below we will additionally predicate every branch for optimization context
            let opt_ctx_indexers = opt_ctx.save_indexers();

            let exec_word = Boolean::and(
                &execute_family,
                &boolean_set.get_minor_flag(MEMORY_COMMON_OP_KEY, ACCESS_WORD_OP_KEY),
                cs,
            );
            let exec_half_word = Boolean::and(
                &execute_family,
                &boolean_set.get_minor_flag(MEMORY_COMMON_OP_KEY, ACCESS_HALF_WORD_OP_KEY),
                cs,
            );
            let exec_byte = Boolean::and(
                &execute_family,
                &boolean_set.get_minor_flag(MEMORY_COMMON_OP_KEY, ACCESS_BYTE_OP_KEY),
                cs,
            );

            // work with 3 different access types
            {
                // WORD
                opt_ctx.restore_indexers(opt_ctx_indexers);

                let rd = [
                    Constraint::<F>::from(source.0[0]),
                    Constraint::<F>::from(source.0[1]),
                ];
                let mem_dst = Register([val_to_store.u16_limbs[0], val_to_store.u16_limbs[1]]);

                if ASSUME_TRUSTED_CODE {
                    // unprovable if we do not have proper alignment
                    cs.add_constraint(
                        (Term::from(bit_0) + Term::from(bit_1)) * exec_word.get_terms(),
                    );

                    let data = AuxDataArrReduced::new(exec_word, mem_dst, rd);
                    aux_data_arr_reduced.push(data);
                } else {
                    todo!();

                    // let trapped =
                    //     Boolean::Is(cs.add_variable_from_constraint(Term::from(1) - offset_0_flag));

                    // let data = AuxDataArr::new(exec_word, trapped, mem_dst, rd);
                    // aux_data_arr.push(data);
                }
            }

            let read_selected_u16 = Num::<F>::Var(cs.add_variable_from_constraint(
                (Term::from(1) - Term::from(bit_1)) * mem_read_words[0]
                    + (Term::from(bit_1) * mem_read_words[1]),
            ));

            {
                // HALF WORD
                opt_ctx.restore_indexers(opt_ctx_indexers);

                let rd = if SUPPORT_SIGNED {
                    let should_sign_extend_mem = boolean_set
                        .get_minor_flag(MEMORY_COMMON_OP_KEY, SIGN_EXTEND_ON_LOAD_OP_KEY);
                    let [sign_if_u16, _high_byte] = opt_ctx.append_lookup_relation(
                        cs,
                        &[read_selected_u16.get_variable()],
                        TableType::U16GetSignAndHighByte.to_num(),
                        exec_half_word,
                    );
                    let do_sign_extension = cs.add_variable_from_constraint(
                        Term::from(sign_if_u16)
                            * Term::from(should_sign_extend_mem.get_variable().unwrap()),
                    );
                    let rd = [
                        Constraint::<F>::from(read_selected_u16),
                        Constraint::<F>::from(Term::from(do_sign_extension) * Term::from(0xffff)),
                    ];

                    rd
                } else {
                    let rd = [
                        Constraint::<F>::from(read_selected_u16),
                        Constraint::<F>::empty(),
                    ];

                    rd
                };

                let read_selected_u16 = Term::from(read_selected_u16);
                // overwrite - we need to update a value that we read from memory with the corresponding word.
                // Note that we selected word based on bit_1 already, so here we need to use the same bit
                // and update either low or high part

                // If bit_1 is false, then we both:
                // - selected `read_selected_word` as mem_read_words[0] above
                // - will use it in combination for low word
                let mem_dst_low = Num::Var(cs.add_variable_from_constraint(
                    (Term::from(word_to_store) - read_selected_u16)
                        * (Term::from(1) - Term::from(bit_1))
                        + mem_read_words[0],
                ));
                // same applies for top half
                let mem_dst_high = Num::Var(cs.add_variable_from_constraint(
                    (Term::from(word_to_store) - read_selected_u16) * Term::from(bit_1)
                        + mem_read_words[1],
                ));
                let mem_dst = Register([mem_dst_low, mem_dst_high]);

                if ASSUME_TRUSTED_CODE {
                    // unprovable if we do not have proper alignment
                    cs.add_constraint(Term::from(bit_0) * exec_half_word.get_terms());

                    let data = AuxDataArrReduced::new(exec_half_word, mem_dst, rd);
                    aux_data_arr_reduced.push(data);
                } else {
                    todo!();

                    // let trapped =
                    //     Boolean::Is(cs.add_variable_from_constraint(offset_1_flag + offset_3_flag));

                    // let data = AuxDataArr::new(exec_half_word, trapped, mem_dst, rd);
                    // aux_data_arr.push(data);
                }
            }

            {
                // BYTE
                opt_ctx.restore_indexers(opt_ctx_indexers);

                // we already have word selected above, so we can get byte and sign at once here,
                // by performing lookup over linear combination of word + 1 bit

                let constraint = Term::from(read_selected_u16)
                    + Term::from((F::from_u64_unchecked(1 << 16), bit_0));
                let [read_selected_byte, sign_if_u8] = opt_ctx
                    .append_lookup_relation_from_linear_terms(
                        cs,
                        &[constraint],
                        TableType::U16SelectByteAndGetByteSign.to_num(),
                        exec_byte,
                    );

                let rd = if SUPPORT_SIGNED {
                    let should_sign_extend_mem = boolean_set
                        .get_minor_flag(MEMORY_COMMON_OP_KEY, SIGN_EXTEND_ON_LOAD_OP_KEY);

                    let do_sign_extension = cs.add_variable_from_constraint(
                        Term::from(sign_if_u8)
                            * Term::from(should_sign_extend_mem.get_variable().unwrap()),
                    );

                    let rd = [
                        Term::from(do_sign_extension) * Term::from(0xff << 8)
                            + Term::from(read_selected_byte),
                        Term::from(do_sign_extension) * Term::from(0xffff),
                    ];

                    rd
                } else {
                    let rd = [
                        Constraint::<F>::from(read_selected_byte),
                        Constraint::<F>::empty(),
                    ];

                    rd
                };

                let read_selected_byte = Term::from(read_selected_byte);
                // we selected word above based on bit_1, and our selected byte also matches with the byte at the proper position
                // for the corresponding word, so we just need to make a constraint to properly update it

                let byte_difference_shifted = cs.add_variable_from_constraint(
                    (Term::from(byte_to_store) - read_selected_byte)
                        * (shift_8 * Term::from(bit_0) + (Term::from(1) - Term::from(bit_0))), // we should shift it by either 8 or 0 bits
                );

                // We only "roll over" the difference based on correct bit_1
                let mem_dst_low = Num::Var(cs.add_variable_from_constraint(
                    ((Term::from(1) - Term::from(bit_1)) * Term::from(byte_difference_shifted))
                        + mem_read_words[0],
                ));
                let mem_dst_high = Num::Var(cs.add_variable_from_constraint(
                    Term::from(bit_1) * Term::from(byte_difference_shifted) + mem_read_words[1],
                ));
                let mem_dst = Register([mem_dst_low, mem_dst_high]);

                if ASSUME_TRUSTED_CODE {
                    let data = AuxDataArrReduced::new(exec_byte, mem_dst, rd);
                    aux_data_arr_reduced.push(data);
                } else {
                    todo!();

                    // let trapped = Boolean::Constant(false);
                    // let data = AuxDataArr::new(exec_byte, trapped, mem_dst, rd);
                    // aux_data_arr.push(data);
                }
            }

            // NOTE on load into x0: even though the result is discarded, we still go
            // through all the same exceptions, as dictated by the spec

            if ASSUME_TRUSTED_CODE {
                // assert!(aux_data_arr.is_empty());

                let (value_to_store, rd) = AuxDataArrReduced::choose_from_orthogonal_variants::<CS>(
                    cs,
                    &aux_data_arr_reduced,
                );
                // we add read query if we LOAD
                memory_queries.push(mem_load_query);

                let execute_read = execute_family;
                let execute_write = should_write_mem;

                // and we form join query if we STORE
                let mut mem_store_query = mem_load_query;
                mem_store_query.local_timestamp_in_cycle = mem_store_timestamp;
                mem_store_query.write_value = std::array::from_fn(|i| {
                    cs.choose(
                        execute_write,
                        value_to_store.0[i],
                        Num::Var(mem_store_query.read_value[i]),
                    )
                    .get_variable()
                });

                memory_queries.push(mem_store_query);

                if execute_family.get_value(cs).unwrap_or(false) {
                    dbg!(execute_read.get_value(cs));
                    dbg!(should_write_mem.get_value(cs));
                    dbg!(execute_write.get_value(cs));
                    dbg!(src1.get_value_unsigned(cs));
                    dbg!(src2.get_register().get_value_unsigned(cs));
                    dbg!(rd.get_value_unsigned(cs));
                }

                let returned_value = [
                    Constraint::<F>::from(rd.0[0].get_variable()),
                    Constraint::<F>::from(rd.0[1].get_variable()),
                ];

                CommonDiffs {
                    exec_flag: execute_family,
                    trapped: None,
                    trap_reason: None,
                    rd_value: Some(returned_value),
                    new_pc_value: NextPcValue::Default,
                }
            } else {
                // we trap if misaligned access that can happen in untrusted code

                todo!();

                // assert!(aux_data_arr_reduced.is_empty());

                // let (misallgned_access, value_to_store, rd) =
                //     AuxDataArr::choose_from_orthogonal_variants::<CS>(cs, &aux_data_arr);

                // let trapped = misallgned_access;
                // let execute_read = Boolean::multi_and(&[execute_family, trapped.toggle()], cs);

                // todo!();

                // // we add read query if we LOAD
                // memory_queries.push(mem_load_query);
                // // and we form join query if we STORE
                // // let mut mem_store_query = mem_load_query;
                // // mem_store_query.local_timestamp_in_cycle = mem_store_timestamp;
                // // mem_store_query.write_value = std::array::from_fn(|i| {
                // //     cs.choose(address_is_in_ram_range, value_to_store.0[i], mem_store_query.read_value[i]).get_variable()
                // // });

                // // there are three possibilities for trap_reason:
                // // 1) LoadAddressMisaligned - in case of misaligned read
                // // 2) StoreOrAMOAddressMisaligned - in case of missaligned write
                // // trap_reason = misaligned_access * (mem_write)

                // let trap_reason = if OUTPUT_EXACT_EXCEPTIONS {
                //     let trap_reason = cs.choose(
                //         should_write_mem,
                //         Num::Constant(F::from_u64_unchecked(
                //             TrapReason::StoreOrAMOAddressMisaligned as u64,
                //         )),
                //         Num::Constant(F::from_u64_unchecked(
                //             TrapReason::LoadAddressMisaligned as u64,
                //         )),
                //     );

                //     Some(trap_reason)
                // } else {
                //     None
                // };

                // let execute_write = Boolean::and(&execute_read, &should_write_mem, cs);

                // if execute_family.get_value(cs).unwrap_or(false) {
                //     println!("MEMORY");
                //     dbg!(execute_read.get_value(cs));
                //     dbg!(should_write_mem.get_value(cs));
                //     dbg!(execute_write.get_value(cs));
                //     dbg!(src1.get_value_unsigned(cs));
                //     dbg!(src2.get_register().get_value_unsigned(cs));
                //     dbg!(rd.get_value_unsigned(cs));
                // }

                // CommonDiffs {
                //     exec_flag: execute_family,
                //     trapped: Some(trapped),
                //     trap_reason: trap_reason,
                //     rd_value: Some(RegisterLikeDiff::Register(rd)),
                //     new_pc_value: Some(RegisterLikeDiff::Register(pc_next)),
                // }
            }
        } else {
            // support only SW/LW, and so we assume code is trusted
            assert!(ASSUME_TRUSTED_CODE);

            // this is common for FAMILY of memory instructions

            let src1 = src1.get_register();
            let imm = inputs.get_imm();

            let (unaligned_address, _of_flag) =
                opt_ctx.append_add_relation(src1, imm, execute_family, cs);

            let [bit_0, bit_1] = opt_ctx.append_lookup_relation(
                cs,
                &[unaligned_address.0[0].get_variable()],
                TableType::MemoryOffsetGetBits.to_num(),
                execute_family,
            );
            let aligned_address_low_constraint = {
                Constraint::from(unaligned_address.0[0].get_variable())
                    - (Term::from(bit_1) * Term::from(2))
                    - Term::from(bit_0)
            };

            // NOTE: in a form that we use the "share" lookup we can not use Boolean::or here (that uses custom witness generation)
            // that assumes that some values are booleans. Instead we evaluate it as generic constraint

            // 1 - b + ab
            // res = 1 - (1 - a)(1-b) = a + b - ab
            let is_unaligned = cs.add_variable_from_constraint(
                Constraint::from(bit_0) + Term::from(bit_1) - Term::from(bit_0) * Term::from(bit_1),
            );

            // unprovable if unaligned
            cs.add_constraint(Term::from(is_unaligned) * execute_family.get_terms());

            // NOTE: whether it's read or write, we will always read from src1 + imm
            let (source, mem_load_query, address_is_in_ram_range) =
                read_from_shuffle_ram_or_bytecode_no_decomposition_with_ctx(
                    cs,
                    mem_load_timestamp,
                    aligned_address_low_constraint,
                    unaligned_address.0[1],
                    opt_ctx,
                    execute_family,
                );

            // NOTE: `should_write_mem` always conditioned over execution of the opcode itself
            cs.add_constraint(
                should_write_mem.get_terms()
                    * (Term::from(1) - Term::from(address_is_in_ram_range)),
            );

            // if we will do STORE, then it'll be the value
            let val_to_store = src2.get_register();

            let returned_value = [
                Constraint::<F>::from(source.0[0].get_variable()),
                Constraint::<F>::from(source.0[1].get_variable()),
            ];

            let value_to_store = Register([val_to_store.0[0], val_to_store.0[1]]);

            // we add read query if we LOAD
            memory_queries.push(mem_load_query);

            let execute_read = execute_family;
            let execute_write = should_write_mem;

            // and we form join query if we STORE
            let mut mem_store_query = mem_load_query;
            mem_store_query.local_timestamp_in_cycle = mem_store_timestamp;
            mem_store_query.write_value = std::array::from_fn(|i| {
                cs.choose(
                    execute_write,
                    value_to_store.0[i],
                    Num::Var(mem_store_query.read_value[i]),
                )
                .get_variable()
            });

            memory_queries.push(mem_store_query);

            if execute_family.get_value(cs).unwrap_or(false) {
                println!("MEMORY");
                dbg!(execute_read.get_value(cs));
                dbg!(should_write_mem.get_value(cs));
                dbg!(execute_write.get_value(cs));
                dbg!(src1.get_value_unsigned(cs));
                dbg!(src2.get_register().get_value_unsigned(cs));
                // dbg!(rd.get_value_unsigned(cs));
            }

            CommonDiffs {
                exec_flag: execute_family,
                trapped: None,
                trap_reason: None,
                rd_value: Some(returned_value),
                new_pc_value: NextPcValue::Default,
            }
        }
    }
}
