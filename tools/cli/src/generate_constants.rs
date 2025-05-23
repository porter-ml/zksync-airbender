use crate::{vk::generate_params_for_binary, Machine};
use blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
use execution_utils::{
    compute_chain_encoding, final_recursion_layer_verifier_vk,
    recursion_layer_no_delegation_verifier_vk, recursion_layer_verifier_vk,
    universal_circuit_no_delegation_verifier_vk, universal_circuit_verifier_vk,
};

pub fn generate_constants_for_binary(bin: &String, universal_verifier: &bool, recompute: &bool) {
    let base_layer_bin = std::fs::read(bin).expect("Failed to read base layer binary file");

    let (end_params, aux_values) = if *universal_verifier {
        if *recompute {
            generate_params_and_register_values(
                &base_layer_bin,
                execution_utils::UNIVERSAL_CIRCUIT_VERIFIER,
                execution_utils::UNIVERSAL_CIRCUIT_VERIFIER,
                execution_utils::UNIVERSAL_CIRCUIT_NO_DELEGATION_VERIFIER,
                execution_utils::UNIVERSAL_CIRCUIT_NO_DELEGATION_VERIFIER,
            )
        } else {
            let base_params = generate_params_for_binary(&base_layer_bin, Machine::Standard);

            let aux_values = compute_chain_encoding(vec![
                [0u32; 8],
                base_params,
                universal_circuit_verifier_vk().params,
                universal_circuit_no_delegation_verifier_vk().params,
            ]);

            (
                universal_circuit_no_delegation_verifier_vk().params,
                aux_values,
            )
        }
    } else {
        if *recompute {
            generate_params_and_register_values(
                &base_layer_bin,
                execution_utils::BASE_LAYER_VERIFIER,
                execution_utils::RECURSION_LAYER_VERIFIER,
                execution_utils::RECURSION_LAYER_NO_DELEGATION_VERIFIER,
                execution_utils::FINAL_RECURSION_LAYER_VERIFIER,
            )
        } else {
            let base_params = generate_params_for_binary(&base_layer_bin, Machine::Standard);

            let aux_values = compute_chain_encoding(vec![
                [0u32; 8],
                base_params,
                recursion_layer_verifier_vk().params,
                recursion_layer_no_delegation_verifier_vk().params,
                final_recursion_layer_verifier_vk().params,
            ]);

            (final_recursion_layer_verifier_vk().params, aux_values)
        }
    };

    println!("End params: {:?}", end_params);
    println!("Aux values: {:?}", aux_values);
}

pub fn generate_params_and_register_values(
    base_layer_bin: &[u8],
    first_recursion_layer_bin: &[u8],
    next_recursion_layer_bin: &[u8],
    first_final_recursion_bin: &[u8],
    next_final_recursion_bin: &[u8],
) -> (
    [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
    [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
) {
    let end_params = generate_params_for_binary(next_final_recursion_bin, Machine::ReducedFinal);

    let aux_registers_values = compute_commitment_for_chain_of_programs(
        base_layer_bin,
        first_recursion_layer_bin,
        next_recursion_layer_bin,
        first_final_recursion_bin,
    );
    (end_params, aux_registers_values)
}

/// blake(
///     blake(
///         blake([0u32; 8] || base_program_end_params)
///         || first_recursion_step_end_params)
///     || next_recursion_step_end_params
/// )
fn compute_commitment_for_chain_of_programs(
    base_layer_bin: &[u8],
    first_recursion_layer_bin: &[u8],
    next_recursion_layer_bin: &[u8],
    first_final_recursion_bin: &[u8],
) -> [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS] {
    let base_layer_end_params = generate_params_for_binary(base_layer_bin, Machine::Standard);

    let first_recursion_layer_end_params =
        generate_params_for_binary(first_recursion_layer_bin, Machine::Reduced);

    let next_recursion_layer_end_params =
        generate_params_for_binary(next_recursion_layer_bin, Machine::Reduced);

    let first_final_recursion_end_params =
        generate_params_for_binary(first_final_recursion_bin, Machine::ReducedFinal);

    compute_chain_encoding(vec![
        [0u32; BLAKE2S_DIGEST_SIZE_U32_WORDS],
        base_layer_end_params,
        first_recursion_layer_end_params,
        next_recursion_layer_end_params,
        first_final_recursion_end_params,
    ])
}
