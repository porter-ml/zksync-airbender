# ZK prover example

Simple fibonacci, that computes the 10th fibonacci number and stores it in the output register.

Does not use any Oracles (inputs, outputs).

## Building

one time setup:

```
rustup target add riscv32i-unknown-none-elf
rustup component add llvm-tools-preview
```

After each change:
```
cargo build
cargo objcopy  -- -O binary app.bin
```

## Proving
Please use `tools/cli` binary.