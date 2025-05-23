use super::*;

pub const SHIFT_COMMON_OP_KEY: DecoderMajorInstructionFamilyKey =
    DecoderMajorInstructionFamilyKey("SHIFT_COMMON_KEY");
// by default - all shifts are left shifts
// pub const SHIFT_LEFT_KEY: DecoderInstructionVariantsKey = DecoderInstructionVariantsKey("SLL/SLLI/ROL");
pub const SHIFT_RIGHT_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("SRL/SRLI/ROR/RORI");
pub const SHIFT_CYCLIC_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("ROR/RORI/ROL");
pub const SHIFT_RIGHT_ALGEBRAIC_KEY: DecoderInstructionVariantsKey =
    DecoderInstructionVariantsKey("SRA");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShiftOp<const SUPPORT_SRA: bool, const SUPPORT_ROT: bool>;

impl<const SUPPORT_SRA: bool, const SUPPORT_ROT: bool> DecodableMachineOp
    for ShiftOp<SUPPORT_SRA, SUPPORT_ROT>
{
    fn define_decoder_subspace(
        &self,
        opcode: u8,
        func3: u8,
        func7: u8,
    ) -> Result<
        (
            InstructionOperandSelectionData,
            InstructionType,
            DecoderMajorInstructionFamilyKey,
            &'static [DecoderInstructionVariantsKey],
        ),
        (),
    > {
        let params = match (opcode, func3, func7) {
            (OPERATION_OP_IMM, 0b001, 0) => {
                // SLLI
                (
                    BASE_I_TYPE_AUX_DATA,
                    InstructionType::IType,
                    SHIFT_COMMON_OP_KEY,
                    &[][..],
                )
            }
            (OPERATION_OP_IMM, 0b101, 0) => {
                // SRLI
                (
                    BASE_I_TYPE_AUX_DATA,
                    InstructionType::IType,
                    SHIFT_COMMON_OP_KEY,
                    &[SHIFT_RIGHT_KEY][..],
                )
            }
            (OPERATION_OP_IMM, 0b101, 0b010_0000) if SUPPORT_SRA => {
                // SRAI
                (
                    BASE_I_TYPE_AUX_DATA,
                    InstructionType::IType,
                    SHIFT_COMMON_OP_KEY,
                    &[SHIFT_RIGHT_KEY, SHIFT_RIGHT_ALGEBRAIC_KEY][..],
                )
            }
            (OPERATION_OP_IMM, 0b101, 0b011_0000) if SUPPORT_ROT => {
                // RORI
                (
                    BASE_I_TYPE_AUX_DATA,
                    InstructionType::IType,
                    SHIFT_COMMON_OP_KEY,
                    &[SHIFT_RIGHT_KEY, SHIFT_CYCLIC_KEY][..],
                )
            }
            (OPERATION_OP, 0b001, 0) => {
                // SLL
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    SHIFT_COMMON_OP_KEY,
                    &[][..],
                )
            }
            (OPERATION_OP, 0b101, 0) => {
                // SRL
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    SHIFT_COMMON_OP_KEY,
                    &[SHIFT_RIGHT_KEY][..],
                )
            }
            (OPERATION_OP, 0b101, 0b010_0000) if SUPPORT_SRA => {
                // SRA
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    SHIFT_COMMON_OP_KEY,
                    &[SHIFT_RIGHT_KEY, SHIFT_RIGHT_ALGEBRAIC_KEY][..],
                )
            }
            (OPERATION_OP, 0b001, 0b011_0000) if SUPPORT_ROT => {
                // ROL
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    SHIFT_COMMON_OP_KEY,
                    &[SHIFT_CYCLIC_KEY][..],
                )
            }
            (OPERATION_OP, 0b101, 0b011_0000) if SUPPORT_ROT => {
                // ROR
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    SHIFT_COMMON_OP_KEY,
                    &[SHIFT_RIGHT_KEY, SHIFT_CYCLIC_KEY][..],
                )
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
        const SUPPORT_SRA: bool,
        const SUPPORT_ROT: bool,
    > MachineOp<F, ST, RS, DE, BS> for ShiftOp<SUPPORT_SRA, SUPPORT_ROT>
{
    fn define_used_tables() -> Vec<TableType> {
        vec![TableType::ShiftImplementation]
        // if SUPPORT_SRA {
        //     vec![TableType::PowersOf2, TableType::SRASignFiller]
        // } else {
        //     vec![TableType::PowersOf2]
        // }
    }

    fn apply<
        CS: Circuit<F>,
        const ASSUME_TRUSTED_CODE: bool,
        const OUTPUT_EXACT_EXCEPTIONS: bool,
    >(
        cs: &mut CS,
        _machine_state: &ST,
        inputs: &DE,
        boolean_set: &BS,
        opt_ctx: &mut OptimizationContext<F, CS>,
    ) -> CommonDiffs<F> {
        // this is common for FAMILY of shift instructions
        opt_ctx.reset_indexers();
        let exec_flag = boolean_set.get_major_flag(SHIFT_COMMON_OP_KEY);
        let is_right_shift = boolean_set.get_minor_flag(SHIFT_COMMON_OP_KEY, SHIFT_RIGHT_KEY);

        const SHIFT_MASK: u64 = (1 << 5) - 1;

        let src1 = inputs.get_rs1_or_equivalent();
        let src2 = inputs.get_rs2_or_equivalent();

        let input = src1.get_register_with_decomposition().u8_decomposition;
        let shift_amount_low_byte = src2.get_register_with_decomposition().u8_decomposition[0];

        let [encoded_shift_amount] = opt_ctx.append_lookup_relation_from_linear_terms::<2, 1>(
            cs,
            &[
                Constraint::from(shift_amount_low_byte.get_variable()),
                Constraint::from(SHIFT_MASK),
            ],
            TableType::And.to_num(),
            exec_flag,
        );

        // we will do a little of brute force and ask a table for contributions

        if exec_flag.get_value(cs).unwrap_or(false) {
            println!("SHIFT OPCODE");
            dbg!(src1.get_register().get_value_unsigned(cs));
            dbg!(src2.get_register().get_value_unsigned(cs));
            dbg!(cs.get_value(encoded_shift_amount));
            if is_right_shift.get_value(cs).unwrap() {
                if SUPPORT_SRA {
                    if boolean_set
                        .get_minor_flag(SHIFT_COMMON_OP_KEY, SHIFT_RIGHT_ALGEBRAIC_KEY)
                        .get_value(cs)
                        .unwrap()
                    {
                        println!("SRA");
                    } else {
                        println!("SRL");
                    }
                } else {
                    println!("SRL");
                }
            } else {
                println!("SLL");
            }
        }

        use crate::tables::*;

        let rd_parts = if SUPPORT_ROT == false {
            let results_per_input_byte_words: [_; 4] = std::array::from_fn(|byte_idx| {
                let input_byte = input[byte_idx].get_variable();
                let byte_idx = byte_idx as u64;
                let mut input_constraint = Constraint::empty();
                let mut shift = 0;
                // byte index
                input_constraint += Term::<F>::from((1u64 << shift) * byte_idx);
                shift += NUM_BYTE_INDEX_BITS;
                // shift type
                input_constraint += Term::<F>::from((
                    F::from_u64_unchecked(1u64 << shift),
                    is_right_shift.get_variable().unwrap(),
                ));
                assert_eq!(0, LEFT_OR_RIGHT_BIT_INDEX);
                shift += 1;
                if SUPPORT_SRA {
                    let sra_flag =
                        boolean_set.get_minor_flag(SHIFT_COMMON_OP_KEY, SHIFT_RIGHT_ALGEBRAIC_KEY);
                    input_constraint += Term::<F>::from((
                        F::from_u64_unchecked(1u64 << shift),
                        sra_flag.get_variable().unwrap(),
                    ));
                    assert_eq!(1, ARITHMETIC_SHIFT_INDEX);
                }
                shift += 1;
                assert_eq!(shift, NUM_BYTE_INDEX_BITS + NUM_SHIFT_TYPE_BITS);
                // shift amount
                input_constraint +=
                    Term::<F>::from((F::from_u64_unchecked(1u64 << shift), encoded_shift_amount));
                shift += NUM_SHIFT_AMOUNT_BITS;
                // byte itself
                input_constraint +=
                    Term::<F>::from((F::from_u64_unchecked(1u64 << shift), input_byte));

                // perform lookup
                let [low, high] = opt_ctx.append_lookup_relation_from_linear_terms::<1, 2>(
                    cs,
                    &[input_constraint],
                    TableType::ShiftImplementation.to_num(),
                    exec_flag,
                );

                [low, high]
            });

            results_per_input_byte_words
        } else {
            todo!();

            // let cyclic_flag = boolean_set.get_minor_flag(SHIFT_COMMON_OP_KEY, SHIFT_CYCLIC_KEY);
            // let result_if_shift_left = low;
            // let result_if_shift_right = high;

            // // unfortunately we can not select with single combination of degree 2 because we have cyclic right,
            // // so we need extra variable
            // let is_left_cyclic = Boolean::and(&cyclic_flag, &is_right_shift.toggle(), cs);
            // let is_right_cyclic = Boolean::and(&cyclic_flag, &is_right_shift, cs);

            // let extra_if_rotate_left = high;
            // let extra_if_rotate_right = low;

            // let combined: [Num<F>; REGISTER_SIZE] = std::array::from_fn(|i| {
            //     let logical_left = result_if_shift_left.0[i];
            //     let logical_right = result_if_shift_right.0[i];
            //     let extra_rot_left = extra_if_rotate_left.0[i];
            //     let extra_rot_right = extra_if_rotate_right.0[i];

            //     let selected = cs.add_variable_from_constraint(
            //         Term::from(logical_left) * (Term::from(1) - Term::from(is_right_shift)) // SLL
            //         + Term::from(logical_right) * Term::from(is_right_shift) // SLR
            //         + Term::from(extra_rot_left) * Term::from(is_left_cyclic) // ROL
            //         + Term::from(extra_rot_right) * Term::from(is_right_cyclic), // ROT
            //     );

            //     Num::Var(selected)
            // });

            // let rd = Register(combined);

            // CommonDiffs {
            //     exec_flag,
            //     trapped: None,
            //     trap_reason: None,
            //     rd_value: Some(RegisterLikeDiff::Register(rd)),
            //     new_pc_value: Some(RegisterLikeDiff::Register(pc_next)),
            // }
        };

        // now merge all the contributions

        let returned_value = std::array::from_fn(|word_idx| {
            let mut result_word = Constraint::empty();
            for i in 0..4 {
                result_word += Term::from(rd_parts[i][word_idx]);
            }

            result_word
        });

        CommonDiffs {
            exec_flag,
            trapped: None,
            trap_reason: None,
            rd_value: Some(returned_value),
            new_pc_value: NextPcValue::Default,
        }
    }
}
