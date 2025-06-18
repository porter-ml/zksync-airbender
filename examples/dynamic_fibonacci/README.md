# ZK prover example

Dynamic fibonacci reads a number `n` (in hex) from an input file and computes the n-th fibonacci number.

The example input.txt sets `n=002dc6c0` (3 million iterations). You can try it with tools/cli as shown below.

## Example commands (from tools/cli directory)

Trace the execution and get cycle count:

```
cargo run --profile cli run --bin ../../examples/dynamic_fibonacci/app.bin --input-file ../../examples/dynamic_fibonacci/input.txt --cycles 40000000
```
`--cycles 40000000` tells the prover to trace and prove up to 40m RiscV cycles, enough to accommodate input.txt's default 3000000 iterations.

Prove (with recursion):

```
cargo run --release -p cli --no-default-features --features gpu prove --bin ../../examples/dynamic_fibonacci/app.bin  --cycles 40000000 --input-file ../../examples/dynamic_fibonacci/input.txt --output-dir /tmp --gpu --until final-recursion
```
