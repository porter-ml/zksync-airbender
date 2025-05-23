use super::*;

pub const BINOP_COMMON_OP_KEY: DecoderMajorInstructionFamilyKey =
    DecoderMajorInstructionFamilyKey("BINOP_COMMON_KEY");
pub const AND_OP_KEY: DecoderInstructionVariantsKey = DecoderInstructionVariantsKey("AND/ANDI");
pub const OR_OP_KEY: DecoderInstructionVariantsKey = DecoderInstructionVariantsKey("OR/ORI");
pub const XOR_OP_KEY: DecoderInstructionVariantsKey = DecoderInstructionVariantsKey("XOR/XORI");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryOp;

impl DecodableMachineOp for BinaryOp {
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
            (OPERATION_OP, 0b111, 0b000_0000) => {
                // AND
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    BINOP_COMMON_OP_KEY,
                    &[AND_OP_KEY][..],
                )
            }
            (OPERATION_OP_IMM, 0b111, _) => {
                // ANDI
                (
                    BASE_I_TYPE_AUX_DATA,
                    InstructionType::IType,
                    BINOP_COMMON_OP_KEY,
                    &[AND_OP_KEY][..],
                )
            }
            (OPERATION_OP, 0b110, 0b000_0000) => {
                // OR
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    BINOP_COMMON_OP_KEY,
                    &[OR_OP_KEY][..],
                )
            }
            (OPERATION_OP_IMM, 0b110, _) => {
                // ORI
                (
                    BASE_I_TYPE_AUX_DATA,
                    InstructionType::IType,
                    BINOP_COMMON_OP_KEY,
                    &[OR_OP_KEY][..],
                )
            }
            (OPERATION_OP, 0b100, 0b000_0000) => {
                // XOR
                (
                    BASE_R_TYPE_AUX_DATA,
                    InstructionType::RType,
                    BINOP_COMMON_OP_KEY,
                    &[XOR_OP_KEY][..],
                )
            }
            (OPERATION_OP_IMM, 0b100, _) => {
                // ANDI
                (
                    BASE_I_TYPE_AUX_DATA,
                    InstructionType::IType,
                    BINOP_COMMON_OP_KEY,
                    &[XOR_OP_KEY][..],
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
    > MachineOp<F, ST, RS, DE, BS> for BinaryOp
{
    fn define_used_tables() -> Vec<TableType> {
        vec![TableType::Xor, TableType::Or, TableType::And]
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
        opt_ctx.reset_indexers();
        let exec_flag = boolean_set.get_major_flag(BINOP_COMMON_OP_KEY);

        // decoder will place immediate into SRC2 to account for ADD/ADDI and similar variations
        let src1 = inputs.get_rs1_or_equivalent();
        let src2 = inputs.get_rs2_or_equivalent();

        let funct3 = inputs.funct3();

        let src1_decomposition = src1.get_register_with_decomposition().u8_decomposition;
        let src2_decomposition = src2.get_register_with_decomposition().u8_decomposition;

        // NB: we don't need explicit range checks here: the correctness will be enforced by the call
        // to binary table - first and second tables are costrainted to be 8-bits long
        let mut res_chunks = vec![];
        let iter = itertools::multizip((src1_decomposition.iter(), src2_decomposition.iter()));

        for (left_in, right_in) in iter {
            let [out] = opt_ctx.append_lookup_relation::<2, 1>(
                cs,
                &[left_in.get_variable(), right_in.get_variable()],
                funct3,
                exec_flag,
            );
            res_chunks.push(out);
        }

        if exec_flag.get_value(cs).unwrap_or(false) {
            println!("BINOP");
            dbg!(src1.get_register().get_value_unsigned(cs));
            dbg!(src2.get_register().get_value_unsigned(cs));
            dbg!(cs.get_value(funct3.get_variable()));
            // dbg!(rd.get_value_unsigned(cs));
        }

        let returned_value = [
            Constraint::<F>::from(
                Term::from(res_chunks[0])
                    + Term::from((F::from_u64_unchecked(1 << 8), res_chunks[1])),
            ),
            Constraint::<F>::from(
                Term::from(res_chunks[2])
                    + Term::from((F::from_u64_unchecked(1 << 8), res_chunks[3])),
            ),
        ];

        CommonDiffs {
            exec_flag,
            trapped: None,
            trap_reason: None,
            rd_value: Some(returned_value),
            new_pc_value: NextPcValue::Default,
        }
    }
}
