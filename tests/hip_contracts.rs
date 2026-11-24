//! Compile-only checks: these never allocate GPU memory or call HIP.
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_PROBE: AtomicUsize = AtomicUsize::new(0);

fn compile_probe(body: &str) -> std::process::Output {
    let dir = std::env::temp_dir().join(format!(
        "fcl-hip-contract-{}-{}",
        std::process::id(),
        NEXT_PROBE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let hip = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hip.rs");
    let source = format!(
        "#![allow(dead_code)]\n#[path = {:?}] mod hip;\n{body}\n",
        hip
    );
    let path = dir.join("probe.rs");
    std::fs::write(&path, source).unwrap();
    let output = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "--crate-type=lib", "--emit=metadata"])
        .arg(&path)
        .arg("--out-dir")
        .arg(&dir)
        .output()
        .unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    output
}

#[test]
fn asynchronous_copies_require_unsafe_at_call_sites() {
    for body in [
        "fn check(b: &hip::DevBuf<i32>, s: hip::Stream) { b.upload(&[1], s); }",
        "fn check(b: &hip::DevBuf<i32>, s: hip::Stream) { b.download(&mut [0], s); }",
        "fn check() { hip::upload_raw(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null_mut()); }",
        "fn check() { hip::download_raw(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null_mut()); }",
    ] {
        let output = compile_probe(body);
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "unsafe copy compiled as safe: {body}"
        );
        assert!(diagnostic.contains("E0133"), "{diagnostic}");
    }
}

#[test]
fn ownership_bearing_types_cannot_be_downloaded() {
    let output = compile_probe(
        "fn check(b: &hip::DevBuf<String>, s: hip::Stream) { unsafe { b.download(&mut [String::new()], s); } }",
    );
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(diagnostic.contains("DeviceScalar"), "{diagnostic}");
}

#[test]
fn scalar_copies_compile_with_explicit_unsafe() {
    let output = compile_probe(
        "fn check(b: &hip::DevBuf<i32>, s: hip::Stream) { unsafe { b.upload(&[1], s); b.download(&mut [0], s); } }",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
