use std::collections::BTreeMap;

use super::*;
use crate::devices::aux_data::PcWrapper;

#[derive(Clone, Copy, Debug)]
pub struct MinimalStateRegistersInMemory<F: PrimeField> {
    pub pc: Register<F>,
}

impl<F: PrimeField> AbstractMachineState<F> for MinimalStateRegistersInMemory<F> {
    fn set_size() -> usize {
        Register::<F>::set_size()
    }

    fn append_into_variables_set(&self, dst: &mut Vec<Variable>) {
        self.pc.append_into_variables_set(dst);
    }
}

impl<F: PrimeField> MinimalStateRegistersInMemory<F> {
    pub fn initialize<CS: Circuit<F>>(circuit: &mut CS) -> Self {
        // this will link to witness inputs
        let pc = PcWrapper::<F>::initialize(circuit);

        Self { pc: pc.pc }
    }
}

impl<F: PrimeField> BaseMachineState<F> for MinimalStateRegistersInMemory<F> {
    fn opcodes_are_in_rom() -> bool {
        true
    }

    fn get_pc(&self) -> &Register<F> {
        &self.pc
    }
    fn get_pc_mut(&mut self) -> &mut Register<F> {
        &mut self.pc
    }

    fn csr_use_props() -> CSRUseProperties {
        CSRUseProperties {
            standard_csrs: vec![],
            allow_non_determinism_csr: true,
            support_mstatus: false,
        }
    }

    fn all_csrs(&self) -> BTreeMap<u16, Register<F>> {
        BTreeMap::new()
    }
}
