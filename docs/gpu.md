# GPU proving



You can use GPU, to greatly improve proving speed.


```shell
cargo run -p cli --release --no-default-features --features gpu prove --bin prover/app.bin --output-dir /tmp/foo --gpu
```

You must compile with 'gpu' feature flag (so that gpu libraries are linked), and you must pass '--gpu' parameter.

Current issues:
* It works only on basic & recursion level - final proofs are still done on CPU (as they require 150GB of RAM).