# Rust COS Chooser

This is the code for the 2026 Fall semester OMR course for the Mathematical Sciences master. The most important part of the code is the COS pricer, which uses the characteristic function for a distribution to evaluate the expected value of a payoff on a stock with that characteristic function. 

There is code for the COS method, as well as some other ways to price financial instruments. These other methods are only really used to verify results from applying the COS method.

Some helper functions are in `util.rs`. 

# Requires

Make sure a recent version of `rust` and `cargo` has been installed on your system. You can find those [here](https://rust-lang.org/).

# Running

To run the main `price_options.rs` file (the default binary), simply call
```sh
$ cargo run
```

To run any of the other binaries (`show-gbm-paths`, `show-merton-paths`, etc.), use the `--bin` flag followed by the binary (as specified in `Cargo.toml`). For example:
```sh
$ cargo run --bin show-gbm-paths
```

If you want to run the fastest possible version of a binary, make sure to pass the `--release` flag to cargo. For example:
```sh
$ cargo run --release --bin show-gbm-paths
```