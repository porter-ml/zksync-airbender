# Running prover end to end

You can either run the proof for some ethereum transaction (using anvil-zksync) or prove execution of a custom binary.

To run prover end to end (from riscV binary to a SNARK), you will need 4 pieces:
* anvil-zksync (if you want to prove some transactions)
* cli from this repo (tools/cli)
* cli from the zkos_wrapper repo
* cli from era-boojum-validator-cli repo

## Preparing binary & data

### Custom code

If you want to prove some custom riscV code, first check if it works (for example running a hashed fibonacci)

```shell
cargo run --release -p cli run --bin examples/hashed_fibonacci/app.bin --input-file examples/hashed_fibonacci/input.txt
```

Remember the final register outputs, as you should compare them with the ones from step 3.


### Anvil-zksync

If you want to prove some ethereum transaction:

* you can either use the precompiled binary from `examples/zksync_os/app.bin`
* OR checkout zksync-os repo, and compile the zksync_os binary

```shell
cd zksync-os/zksync_os && ./dump_bin.sh
```

* checkout anvil-zksync repo **with boojumos-dev** branch

```shell
cargo run -- --use-boojum --boojum-bin-path ../zksync-os/zksync_os/app.bin
```

This will start anvil-zksync with boojum  on port 8011. Then you can use `cast` to create some transactions:

```shell
cast send -r http://localhost:8011 0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65 --value 100ether --private-key   0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 --gas-limit 10000000
```

## Proving 

There are 4 options:

* if you do/don't have GPU
* if you do/don't have 150GB of RAM.


### Cheap option (no GPU, no RAM).

If you don't have RAM, you'll have to stop on 'final-recursion' step.

If you run your custom code:

```shell
cargo run --release -p cli prove --bin examples/hashed_fibonacci/app.bin --input-file examples/hashed_fibonacci/input.txt  --until final-recursion --tmp-dir /tmp
```

If you run anvil-zksync (and want to prove first batch):

```shell
cargo run --release -p cli  prove --bin ../zksync-os/zksync_os/app.bin  --input-rpc http://localhost:8011 --input-batch 1 --output-dir /tmp  --until final-recursion
```

### GPU

If you have gpu, you can compile with `--features gpu` flag, and then pass `--gpu` - to make proving go a lot faster:


```shell
cargo run --release -p cli --features gpu prove --bin ../zksync-os/zksync_os/app.bin  --input-rpc http://localhost:8011 --input-batch 1 --output-dir /tmp --gpu --until final-recursion
```


### If you have more than 150GB of RAM

In such case, you can pass `--until final-proof` - to compute the single proof.


```shell
cargo run --release -p cli prove --bin examples/hashed_fibonacci/app.bin --input-file examples/hashed_fibonacci/input.txt --output-dir /tmp --until final-proof --tmp-dir /tmp
```

Where 'bin' is your riscV binary, and input-file (optional) is any input data that your binary consumes.

After a while, you'll end up with a single 'final' file in the output dir, called `final_program_proof.json`

## Wrapping the riscV into SNARK

This step works only if you have over 150GB of RAM, and did the `--until final-prove` before:

You need to get the zkos-wrapper repo, and run:

```
cargo run --release -- --input /tmp/final_program_proof.json --input-binary ../zksync-airbender/examples/hashed_fibonacci/app.bin   --output-dir /tmp
```

Make sure that you pass the same input-binary that you used during proving (if not, you'll get a failed assert quickly).

This step, will wrap your boojum 2 prover proof, first into original boojum (together with compression), and then finally into a single SNARK.

### verify the snark

For this step, please use the tool from `era-boojum-validator-cli` repo:

```
cargo run -- verify-snark-boojum-os /tmp/snark_proof.json /tmp/snark_vk.json
```

This tool will verify that the proof and verification key matches.

### Generating verification keys

The code above is using 'fake' CRS - for production use cases, you should pass `--trusted-setup-file` during ZKsyncOS wrapper.

You can also generate verification key for snark by running (from zkos_wrapper repo):

```shell
cargo run --release generate-vk --input-binary ../zksync-airbender/examples/hashed_fibonacci/app.bin --output-dir /tmp --trusted-setup-file crs/setup.key
```