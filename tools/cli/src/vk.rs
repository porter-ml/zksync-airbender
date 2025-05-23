use std::alloc::Global;

use crate::Machine;
use execution_utils::VerificationKey;
use sha3::{Digest, Keccak256};
use worker::Worker;

pub fn generate_vk(bin_path: &String, machine: &Option<Machine>, output: &Option<String>) {
    let binary = std::fs::read(bin_path).expect("Failed to read binary file");

    let mut hasher = Keccak256::new();
    hasher.update(&binary);
    let hash = hasher.finalize();
    let bytecode_hash_hex = format!("{:x}", hash);
    let params = generate_params_for_binary(&binary, machine.clone().unwrap_or(Machine::Standard));

    let params_hex = params
        .iter()
        .map(|p| format!("{:08x}", p))
        .collect::<Vec<_>>()
        .join("");

    let vk = VerificationKey {
        machine_type: format!("{:?}", machine.clone().unwrap_or(Machine::Standard)),
        bytecode_hash_hex,
        params,
        params_hex,
    };

    println!("Verification key generated: {:?}", vk);

    if let Some(output) = output {
        let json = serde_json::to_string_pretty(&vk)
            .expect("Failed to serialize verification key to JSON");
        std::fs::write(output, json).expect("Failed to write verification key to output file");
        println!("Verification key written to {}", output);
    }
}

pub fn generate_params_for_binary(bin: &[u8], machine: Machine) -> [u32; 8] {
    let worker = Worker::new_with_num_threads(8);

    let expected_final_pc = execution_utils::find_binary_exit_point(&bin);
    let binary: Vec<u32> = execution_utils::get_padded_binary(&bin);
    match machine {
        Machine::Standard => execution_utils::compute_end_parameters(
            expected_final_pc,
            &setups::get_main_riscv_circuit_setup::<Global, Global>(&binary, &worker),
        ),
        Machine::Reduced => execution_utils::compute_end_parameters(
            expected_final_pc,
            &setups::get_reduced_riscv_circuit_setup::<Global, Global>(&binary, &worker),
        ),

        Machine::ReducedFinal => execution_utils::compute_end_parameters(
            expected_final_pc,
            &setups::get_final_reduced_riscv_circuit_setup::<Global, Global>(&binary, &worker),
        ),
    }
}
