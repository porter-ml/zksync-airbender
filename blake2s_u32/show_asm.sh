#!/bin/sh
cargo asm -p blake2s_u32 --lib -C target_feature=+zbb,+m --rust --target=riscv32i-unknown-none-elf blake2s_u32::Blake2sState::absorb_reduced_rounds