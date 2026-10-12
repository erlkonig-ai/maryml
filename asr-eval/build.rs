fn main() {
    // MLS on HuggingFace ships Ogg/Opus; decode with the system libopus
    // (runtime library only, no dev headers on sky, so link by path).
    println!("cargo:rustc-link-arg=/lib/aarch64-linux-gnu/libopus.so.0");
}
