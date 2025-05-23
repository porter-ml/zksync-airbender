use prover::{
    fft::GoodAllocator,
    risc_v_simulator::cycle::{IMStandardIsaConfig, IWithoutByteAccessIsaConfigWithDelegation},
};
use setups::{DelegationCircuitPrecomputations, MainCircuitPrecomputations};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::{collections::hash_map::DefaultHasher, sync::Arc};

#[derive(Default)]
pub struct SetupCache<A: GoodAllocator, B: GoodAllocator> {
    pub main_circuit_setup:
        HashMap<u64, Arc<MainCircuitPrecomputations<IMStandardIsaConfig, A, B>>>,
    pub reduced_circuit_setup: HashMap<
        u64,
        Arc<MainCircuitPrecomputations<IWithoutByteAccessIsaConfigWithDelegation, A, B>>,
    >,
    pub delegations: Arc<Vec<(u32, DelegationCircuitPrecomputations<A, B>)>>,
}

impl<A: GoodAllocator, B: GoodAllocator> SetupCache<A, B> {
    pub fn get_or_create_main_circuit(
        &mut self,
        bytecode: &Vec<u32>,
    ) -> &Arc<MainCircuitPrecomputations<IMStandardIsaConfig, A, B>> {
        let mut hasher = DefaultHasher::new();
        bytecode.hash(&mut hasher);
        let hash = hasher.finish();

        self.main_circuit_setup.entry(hash).or_insert_with(|| {
            let worker = worker::Worker::new_with_num_threads(8);
            // Compute the setup here
            Arc::new(setups::get_main_riscv_circuit_setup(&bytecode, &worker))
        })
    }
    pub fn get_or_create_reduced_circuit(
        &mut self,
        bytecode: &Vec<u32>,
    ) -> &Arc<MainCircuitPrecomputations<IWithoutByteAccessIsaConfigWithDelegation, A, B>> {
        let mut hasher = DefaultHasher::new();
        bytecode.hash(&mut hasher);
        let hash = hasher.finish();

        self.reduced_circuit_setup.entry(hash).or_insert_with(|| {
            let worker = worker::Worker::new_with_num_threads(8);
            // Compute the setup here
            Arc::new(setups::get_reduced_riscv_circuit_setup(&bytecode, &worker))
        })
    }

    pub fn get_or_create_delegations(
        &mut self,
    ) -> &Arc<Vec<(u32, DelegationCircuitPrecomputations<A, B>)>> {
        if self.delegations.is_empty() {
            let worker = worker::Worker::new_with_num_threads(8);
            // Compute the setup here
            self.delegations = Arc::new(setups::all_delegation_circuits_precomputations(&worker));
        }
        &self.delegations
    }
}
