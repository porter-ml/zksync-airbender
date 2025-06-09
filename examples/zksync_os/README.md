# ZKsyncOS

This directory contains a binary built from ZKsyncOS repo.

In the future, we should have a proper release pipeline there, so this will no longer be needed.


Binary built from 607e52db0e10b4bbce9cd163169c15f59a075a5a (tag: 0.0.1 + fix from PR-40) (June 6).


To rebuild:
* checkout zksync-os repo
* cd zksync_os && ./dump_bin.sh
* cp app.bin zksync-airbender/examples/zksync_os/app.bin

And in this repo:

```
cargo run --release -p cli generate-vk --bin app.bin --machine standard --output app.vk.json
```

You can also run:
```
cargo run --release -p cli generate-constants app.bin --universal-verifier
```

to see the final verification keys:

```
End params: [450952736, 266097338, 57980520, 3983253845, 3619491068, 640210741, 3659638418, 635428486]
Aux values: [1395365915, 2489455614, 2042893390, 2587610735, 4249872676, 4254236201, 368137084, 3216712393]
```