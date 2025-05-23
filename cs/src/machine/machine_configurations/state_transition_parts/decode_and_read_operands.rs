use super::*;
use crate::machine::decoder::decode_optimized_must_handle_csr::OptimizedDecoderOutput;

pub(crate) fn optimized_decode_and_read_reg_operands<
    F: PrimeField,
    CS: Circuit<F>,
    const ASSUME_TRUSTED_CODE: bool,
    const PERFORM_DELEGATION: bool,
    const ROM_ADDRESS_SPACE_SECOND_WORD_BITS: usize,
>(
    cs: &mut CS,
    pc: Register<F>,
    opcodes_are_in_rom: bool,
    decode_table_splitting: [usize; 2],
) -> (
    Vec<ShuffleRamMemQuery>,
    RS2ShuffleRamQueryCandidate<F>,
    Register<F>,
    Register<F>,
    Constraint<F>,
    OptimizedDecoderOutput<F>,
    Vec<Boolean>,
) {
    let next_opcode = read_opcode_from_rom::<F, CS, ROM_ADDRESS_SPACE_SECOND_WORD_BITS>(cs, pc);

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

    use crate::machine::decoder::decode_optimized_must_handle_csr::*;
    use crate::machine::decoder::DecoderInput;

    let decoder_input = DecoderInput {
        instruction: next_opcode,
    };
    let (invalid_opcode, raw_decoder_output, opcode_format_bits, other_bits) =
        OptimizedDecoder::decode::<F, CS>(&decoder_input, cs, decode_table_splitting);

    if ASSUME_TRUSTED_CODE {
        // if opcode is invalid - it's unsatisfiable
        cs.add_constraint_allow_explicit_linear_prevent_optimizations(Constraint::<F>::from(
            invalid_opcode,
        ));
    } else {
        unimplemented!()
    }

    let mut memory_queries = vec![];

    let (src1_reg, rs1_mem_query) =
        get_rs1_as_shuffle_ram(cs, raw_decoder_output.rs1, opcodes_are_in_rom);
    // even though not all the opcodes read first register, we still consider it unconditional
    memory_queries.push(rs1_mem_query);

    // for src2 we will drag some data along

    let (src2_reg, rs2_mem_query) =
        prepare_rs2_op_as_shuffle_ram(cs, raw_decoder_output.rs2.clone(), opcodes_are_in_rom);
    // we will merge rs2 with mem LOAD opcode queries below

    let (src1, src2, update_rd) = OptimizedDecoder::select_src1_and_src2_values(
        cs,
        &opcode_format_bits,
        src1_reg,
        raw_decoder_output.imm,
        src2_reg,
    );

    (
        memory_queries,
        rs2_mem_query,
        src1,
        src2,
        update_rd,
        raw_decoder_output,
        other_bits,
    )
}
