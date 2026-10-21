fn main() {
    // On macOS a Python extension module is dlopen'ed at import time and the
    // Python C-API symbols are resolved dynamically, so the linker must allow
    // the resulting undefined symbols. maturin already passes this; emitting it
    // here makes a plain `cargo build` of the cdylib work as well.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
