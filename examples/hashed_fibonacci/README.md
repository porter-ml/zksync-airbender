# ZK prover example

Hashed fibonacci, is reading values `n`  and `h` from the input, and computes the n-th fibonacci number % 10000 and then it applies Blake hash `h` times.

This example is used to show how you can use delegation circuits (in this example - blake for hashing).

input.txt contains an example input (15), that can be used with the tools/cli.

For example (from tools/cli dir):

```
cargo run --profile cli run --bin ../../examples/hashed_fibonacci/app.bin --input_file ../../examples/hashed_fibonacci/input.txt
```