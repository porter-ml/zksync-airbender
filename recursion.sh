#!/bin/bash
set -e

# Basic fibonacci recursion

mkdir -p output/
mkdir -p output/basic
mkdir -p output/recursion_0
mkdir -p output/recursion_1
mkdir -p output/recursion_2
mkdir -p output/final_1
mkdir -p output/final_2


# refresh the verifier (with recursion_verifier feature flag).
(cd tools/verifier && cargo objcopy --release  -- -O binary app.bin)
# This will be the verifier that we'll use for the final verification (that cannot use any delegation).
(cd tools/verifier && CARGO_TARGET_DIR=target/no_delegation  cargo objcopy --release --features individual_circuit_no_delegation --no-default-features  -- -O binary app_no_delegation.bin)

# First level of proofs
cargo run -p cli --profile cli prove --bin examples/basic_fibonacci/app.bin --output-dir output/basic


## First recursion level

# Flatten
cargo run -p cli --profile cli flatten-all --input-metadata output/basic/metadata.json --output-file output/basic/proof_flatten0.json

# Run (to check)
cargo run -p cli --profile cli -- run --bin tools/verifier/app.bin --input-file  output/basic/proof_flatten0.json --cycles 10000000 --machine reduced --expected-results 144,0

# Prove
cargo run -p cli --profile cli -- prove --machine reduced --bin tools/verifier/app.bin --input-file  output/basic/proof_flatten0.json --output-dir output/recursion_0 --prev-metadata output/basic/metadata.json

# Verify
cargo run -p cli --profile cli verify --proof output/recursion_0/reduced_proof_0.json

# Verify all 
cargo run -p cli --profile cli verify-all --metadata output/recursion_0/metadata.json 


## Second recursion

# Flatten
cargo run -p cli --profile cli flatten-all --input-metadata output/recursion_0/metadata.json --output-file output/recursion_0/proof_flatten0.json

# Run (to check)
cargo run -p cli --profile cli -- run --bin tools/verifier/app.bin --input-file  output/recursion_0/proof_flatten0.json --cycles 32000000 --machine reduced --expected-results 144,0

# Prove
cargo run -p cli --profile cli -- prove --machine reduced --bin tools/verifier/app.bin --input-file  output/recursion_0/proof_flatten0.json --output-dir output/recursion_1 --prev-metadata output/recursion_0/metadata.json

# Verify
cargo run -p cli --profile cli verify --proof output/recursion_1/reduced_proof_0.json

# Verify all
cargo run -p cli --profile cli verify-all --metadata output/recursion_1/metadata.json 

## Third recursion

# Flatten
cargo run -p cli --profile cli flatten-all --input-metadata output/recursion_1/metadata.json --output-file output/recursion_1/proof_flatten1.json

# Run (to check) - currently around 28M cycles
cargo run -p cli --profile cli -- run --bin tools/verifier/app.bin --input-file  output/recursion_1/proof_flatten1.json --cycles 64000000 --machine reduced --expected-results 144,0

# Prove (this is what fails -- due to RAM)
# previous metadata must be 'recursion_0' - as we don't keep updating the recursion hash chain.
cargo run -p cli --profile cli -- prove --machine reduced --bin tools/verifier/app.bin --input-file  output/recursion_1/proof_flatten1.json --output-dir output/recursion_2 --prev-metadata output/recursion_0/metadata.json

cargo run -p cli --profile cli flatten-all --input-metadata output/recursion_2/metadata.json --output-file output/recursion_2/proof_flatten2.json


## First 'final'

# Run (but use final machine, and no-delegation verifier)
cargo run -p cli --profile cli -- run --bin tools/verifier/app_no_delegation.bin --input-file  output/recursion_2/proof_flatten2.json --cycles 64000000 --machine reduced-final --expected-results 144,0

cargo run -p cli --profile cli -- prove --machine reduced-final --bin tools/verifier/app_no_delegation.bin --input-file  output/recursion_2/proof_flatten2.json --output-dir output/final_1 --prev-metadata output/recursion_0/metadata.json

cargo run -p cli --profile cli flatten-all --input-metadata output/final_1/metadata.json --output-file output/final_1/proof_flatten_final_1.json


## Second 'final'

cargo run -p cli --profile cli -- run --bin tools/verifier/app_no_delegation.bin --input-file  output/final_1/proof_flatten_final_1.json --cycles 64000000 --machine reduced-final --expected-results 144,0

cargo run -p cli --profile cli -- prove --machine reduced-final --bin tools/verifier/app_no_delegation.bin --input-file  output/final_1/proof_flatten_final_1.json --output-dir output/final_2 --prev-metadata output/final_1/metadata.json
