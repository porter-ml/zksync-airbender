# ZKSmith

Stand-alone prover/watcher for anvil zksync.

First - start anvil-zksync with boojum os **must be from boojumos-dev branch**

```
RUST_BACKTRACE=1 cargo run -- --use-boojum --boojum-bin-path ../../zksync-airbender/examples/zksync_os/app.bin
```

Then run ZKSmith

```shell
cargo run -- --anvil-url http://localhost:8011 --zksync-os-bin-path ../../examples/zksync_os/app.bin
```

Go to http://localhost:3030 - and see the progress.