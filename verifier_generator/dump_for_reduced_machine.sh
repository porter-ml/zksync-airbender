#!/bin/sh
cargo test -- test::generate_reduced_machine --exact
rustfmt src/generated.rs
rustfmt src/generated_inlined_verifier.rs
cp src/generated.rs ../verifier/src/generated/circuit_layout.rs
cp src/generated_inlined_verifier.rs ../verifier/src/generated/quotient.rs
