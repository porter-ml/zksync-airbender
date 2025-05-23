use super::*;

pub struct DelegationProcessorDescription {
    pub delegation_type: u32,
    pub num_requests_per_circuit: usize,
    pub trace_len: usize,
    pub table_driver: TableDriver<Mersenne31Field>,
    pub compiled_circuit: CompiledCircuitArtifact<Mersenne31Field>,
}

pub struct DelegationWorkForType<A: Allocator + Clone> {
    pub delegation_type: u16,
    pub num_requests_per_circuit: usize,
    pub trace_len: usize,
    pub compiled_circuit: CompiledCircuitArtifact<Mersenne31Field>,
    pub table_driver: TableDriver<Mersenne31Field>,
    pub work_units: Vec<DelegationProcessorWitness<A>>,
}

pub struct MemoryOnlyDelegationWorkForType<A: Allocator + Clone> {
    pub delegation_type: u16,
    pub num_requests_per_circuit: usize,
    pub trace_len: usize,
    pub work_units: Vec<DelegationProcessorMemoryOnlyWitness<A>>,
}

#[derive(Clone, Debug)]
pub struct DelegationProcessorWitness<A: Allocator + Clone> {
    pub witness: WitnessEvaluationData<DEFAULT_TRACE_PADDING_MULTIPLE, A>,
}

#[derive(Clone, Debug)]
pub struct DelegationProcessorMemoryOnlyWitness<A: Allocator + Clone> {
    pub witness: DelegationMemoryOnlyWitnessEvaluationData<DEFAULT_TRACE_PADDING_MULTIPLE, A>,
}

pub fn check_satisfied<const N: usize, A: GoodAllocator>(
    compiled_machine: &CompiledCircuitArtifact<Mersenne31Field>,
    exec_table: &RowMajorTrace<Mersenne31Field, N, A>,
    num_witness_columns: usize,
) -> bool {
    assert!(exec_table.len().is_power_of_two());
    assert!(exec_table.width() >= num_witness_columns);
    let num_cycles = exec_table.len() - 1;
    let mut exec_table_view = exec_table.row_view(0..num_cycles);
    for row in 0..num_cycles {
        let (witness_row, memory_row) =
            unsafe { exec_table_view.current_row_split(num_witness_columns) };

        let row_satisfied = check_satisfied_row(compiled_machine, &*witness_row, &*memory_row, row);

        if row_satisfied == false {
            println!("Unsatisfied at row {}", row);
            return false;
        }

        exec_table_view.advance_row();
    }

    true
}

pub fn check_satisfied_row(
    compiled_machine: &CompiledCircuitArtifact<Mersenne31Field>,
    witness_row: &[Mersenne31Field],
    memory_row: &[Mersenne31Field],
    absolute_row_idx: usize,
) -> bool {
    // we only check constraints and not tables
    for constraint in compiled_machine.degree_1_constraints.iter() {
        let eval_result = constraint.evaluate_at_row_on_main_domain(witness_row, memory_row);
        if eval_result != Mersenne31Field::ZERO {
            println!(
                "Unsatisfied at row {}, linear constraint {:?}",
                absolute_row_idx, constraint
            );
            for (_, a) in constraint.linear_terms.iter() {
                println!("{:?} = {:?}", a, read_value(*a, witness_row, memory_row));
            }
            return false;
        }
    }

    for constraint in compiled_machine.degree_2_constraints.iter() {
        let eval_result = constraint.evaluate_at_row_on_main_domain(witness_row, memory_row);
        if eval_result != Mersenne31Field::ZERO {
            println!(
                "Unsatisfied at row {}, quadratic constraint {:?}",
                absolute_row_idx, constraint
            );
            for (_, a, b) in constraint.quadratic_terms.iter() {
                println!("{:?} = {:?}", a, read_value(*a, witness_row, memory_row));
                println!("{:?} = {:?}", b, read_value(*b, witness_row, memory_row));
            }
            for (_, a) in constraint.linear_terms.iter() {
                println!("{:?} = {:?}", a, read_value(*a, witness_row, memory_row));
            }
            return false;
        }
    }

    true
}
