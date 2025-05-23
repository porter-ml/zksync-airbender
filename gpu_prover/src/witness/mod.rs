mod column;
mod layout;
pub mod memory_delegation;
pub mod memory_main;
pub(crate) mod multiplicities;
mod option;
mod placeholder;
mod ram_access;
pub mod trace_delegation;
pub mod trace_main;
pub mod witness_delegation;
pub mod witness_main;

use crate::witness::trace_delegation::DelegationCircuitType;
use crate::witness::trace_main::MainCircuitType;
use field::Mersenne31Field;

type BF = Mersenne31Field;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum CircuitType {
    Main(MainCircuitType),
    Delegation(DelegationCircuitType),
}

impl CircuitType {
    pub fn from_delegation_type(delegation_type: u16) -> Self {
        Self::Delegation(delegation_type.into())
    }

    pub fn as_main(&self) -> Option<MainCircuitType> {
        match self {
            CircuitType::Main(circuit_type) => Some(*circuit_type),
            _ => None,
        }
    }

    pub fn as_delegation(&self) -> Option<DelegationCircuitType> {
        match self {
            CircuitType::Delegation(circuit_type) => Some(*circuit_type),
            _ => None,
        }
    }
}
