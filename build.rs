//! Compiles the HIP kernels with hipcc and links them, plus the HIP runtime.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let src = "kernels/kernels.cpp";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=FCL_KERNEL_FLAGS");

    let obj = out.join("kernels.o");
    let status = Command::new(format!("{rocm}/bin/hipcc"))
        .args(["-O3", "-std=c++17", "--offload-arch=gfx1201", "-fPIC"])
        .args(std::env::var("FCL_KERNEL_FLAGS").unwrap_or_default().split_whitespace())
        .args(["-c", src, "-o"])
        .arg(&obj)
        .status()
        .expect("run hipcc");
    assert!(status.success(), "hipcc failed");

    let lib = out.join("libfcl_kernels.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar").arg("rcs").arg(&lib).arg(&obj).status().expect("run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=fcl_kernels");
    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");
}
