# ZK prover example

Dynamic fibonacci, is reading a value `n` from the input, and computes the n-th fibonacci number.

input.txt contains an example input (15), that can be used with the tools/cli.

For example (from tools/cli dir):

```
cargo run --profile cli run --bin ../../examples/dynamic_fibonacci/app.bin --input_file ../../examples/dynamic_fibonacci/input.txt
```