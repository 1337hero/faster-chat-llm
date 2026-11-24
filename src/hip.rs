//! Thin HIP runtime bindings and the C ABI of `kernels/kernels.cpp`.
//!
//! Everything here is asynchronous on a stream unless the name says otherwise. Device memory is
//! addressed by raw pointers because kernels on one card are handed buffers of many shapes; the
//! `DevBuf` wrapper only owns the allocation and remembers its length.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::marker::PhantomData;

pub type Stream = *mut c_void;
pub type Event = *mut c_void;

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;
const DEVICE_TO_DEVICE: c_int = 3;

pub const MAX_GROUPS: usize = 4;
pub const BMAX: usize = 16;
/// Decode attention segment length in cache slots, and floats per segment partial (see kernels.cpp).
pub const ATTN_SEG_SLOTS: usize = 16 * 16;
pub const ATTN_PART: usize = 16 * 256 + 32;
/// Prefill attention sweeps the cache in stretches of this many slots: K and V of one stretch
/// (2 KiB per slot per rank at TP2) stay inside the card's cache while every row tile uses them.
pub const ATTN_SWEEP_SLOTS: usize = 2048;

/// Mirrors `PkGroups` in kernels.cpp.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Groups {
    pub n: i32,
    pub start: [i32; MAX_GROUPS],
    pub len: [i32; MAX_GROUPS],
    pub slot: [i32; MAX_GROUPS],
    pub cur: [i32; MAX_GROUPS],
    pub hist_off: [i32; MAX_GROUPS],
    pub nacc: [i32; MAX_GROUPS],
    pub commit: [i32; MAX_GROUPS],
    pub pos0: [i32; MAX_GROUPS],
    pub kv_dir: [i32; MAX_GROUPS],
    pub kv_base: [i64; MAX_GROUPS],
}

/// Mirrors `PkColRanges`: column selection in 32-element blocks.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ColRanges {
    pub n: i32,
    pub start: [i32; 4],
    pub len: [i32; 4],
}

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipInit(flags: u32) -> c_int;
    fn hipGetDeviceCount(n: *mut c_int) -> c_int;
    fn hipSetDevice(dev: c_int) -> c_int;
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    fn hipHostFree(ptr: *mut c_void) -> c_int;
    fn hipMemcpyAsync(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int, s: Stream) -> c_int;
    fn hipMemsetAsync(dst: *mut c_void, value: c_int, size: usize, s: Stream) -> c_int;
    fn hipMemcpyPeerAsync(dst: *mut c_void, dst_dev: c_int, src: *const c_void, src_dev: c_int, size: usize, s: Stream) -> c_int;
    fn hipStreamCreate(s: *mut Stream) -> c_int;
    fn hipStreamSynchronize(s: Stream) -> c_int;
    fn hipEventCreate(e: *mut Event) -> c_int;
    fn hipEventRecord(e: Event, s: Stream) -> c_int;
    fn hipEventSynchronize(e: Event) -> c_int;
    fn hipStreamWaitEvent(s: Stream, e: Event, flags: u32) -> c_int;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn hipGetErrorString(e: c_int) -> *const c_char;
}

unsafe extern "C" {
    pub fn pk_wave_size(s: Stream, out: *mut i32);
    pub fn pk_unpack_q8(s: Stream, raw: *const u8, qs: *mut i8, d: *mut u16, n_rows: i64, nb_src: i32, nb_dst: i32, dst_row0: i64, cr: *const ColRanges);
    pub fn pk_embed(s: Stream, qs: *const i8, d: *const u16, tokens: *const i32, h: *mut f32, dim: i32, b: i32);
    pub fn pk_norm_quant(s: Stream, h: *mut f32, p0: *const f32, p1: *const f32, w: *const f32, eps: f32, dim: i32, b: i32, xq: *mut i8, xd: *mut f32, xs: i32, e_off: i32, xf: *mut f32);
    pub fn pk_gemm(s: Stream, w: *const i8, wd: *const u16, xq: *const i8, xd: *const f32, xs: i32, b_off: i32, y: *mut f32, rows: i64, cols: i32, b: i32, y_stride: i32);
    pub fn pk_ffn_act(s: Stream, y: *const f32, f: i32, b: i32, xq: *mut i8, xd: *mut f32, xs: i32);
    pub fn pk_gdn_conv(s: Stream, yin: *const f32, nin: i32, mix: *mut f32, qkv: *mut f32, r: i32, conv_w: *const f32, dt_bias: *const f32, ssm_a: *const f32, hk: i32, hv: i32, eps: f32, mix_s: *mut f32, qkv_s: *mut f32, b: i32, g: *const Groups);
    pub fn pk_gdn_step(s: Stream, state: *mut f32, qkv: *const f32, r: i32, yin: *const f32, nin: i32, norm_w: *const f32, hk: i32, hv: i32, eps: f32, xq: *mut i8, xd: *mut f32, xs: i32, qkv_s: *const f32, g: *const Groups);
    pub fn pk_attn_prep(s: Stream, yin: *const f32, nin: i32, hq: i32, hkv: i32, u0: i32, kv_off: i32, q_norm_w: *const f32, k_norm_w: *const f32, eps: f32, theta_scale: f32, n_rot: i32, qr: *mut u16, kc: *mut u16, vc: *mut u16, b: i32, g: *const Groups);
    pub fn pk_attn(s: Stream, qr: *const u16, yin: *const f32, nin: i32, hq: i32, hkv: i32, kc: *const u16, vc: *const u16, seg_tiles: i32, part: *mut f32, tiles: i32, segs: i32, xq: *mut i8, xd: *mut f32, xs: i32, b: i32, g: *const Groups);
    pub fn pk_attn_swept(s: Stream, qr: *const u16, yin: *const f32, nin: i32, hq: i32, hkv: i32, kc: *const u16, vc: *const u16, seg_tiles: i32, seg_first: i32, seg_last: i32, part: *mut f32, tiles: i32, xq: *mut i8, xd: *mut f32, xs: i32, b: i32, g: *const Groups);
    pub fn pk_xchg(s: Stream, own: *const f32, host_out: *mut f32, my_flags: *mut i64, host_in: *const f32, peer_flags: *const i64, peer: *mut f32, n: i32, seq: i64, err: *mut i32);
    pub fn pk_pick(s: Stream, own: *const i32, peer: *const i32, own_v0: i32, peer_v0: i32, tokens: *mut i32, hist: *mut i32, b: i32);
    pub fn pk_argmax(s: Stream, logits: *const f32, n: i64, stride: i64, b: i32, out: *mut i32);
}

#[track_caller]
fn ck(e: c_int) {
    if e != 0 {
        let msg = unsafe { CStr::from_ptr(hipGetErrorString(e)) }.to_string_lossy().into_owned();
        panic!("HIP error {e}: {msg}");
    }
}

pub fn init() -> usize {
    let mut n = 0;
    unsafe {
        ck(hipInit(0));
        ck(hipGetDeviceCount(&mut n));
    }
    n as usize
}

pub fn set_device(dev: usize) {
    unsafe { ck(hipSetDevice(dev as c_int)) }
}

pub fn stream_create() -> Stream {
    let mut s = std::ptr::null_mut();
    unsafe { ck(hipStreamCreate(&mut s)) };
    s
}

pub fn event_create() -> Event {
    let mut e = std::ptr::null_mut();
    unsafe { ck(hipEventCreate(&mut e)) };
    e
}

/// Mark the current tail of stream `s`; `event_wait` returns once everything before it is done.
pub fn event_record(e: Event, s: Stream) {
    unsafe { ck(hipEventRecord(e, s)) }
}

/// Make everything queued on `s` from now on wait for the point `e` marks (on any stream).
pub fn stream_wait_event(s: Stream, e: Event) {
    unsafe { ck(hipStreamWaitEvent(s, e, 0)) }
}

pub fn event_wait(e: Event) {
    unsafe { ck(hipEventSynchronize(e)) }
}

pub fn sync(s: Stream) {
    unsafe { ck(hipStreamSynchronize(s)) }
}

/// (free, total) bytes on the current device.
pub fn mem_info() -> (usize, usize) {
    let (mut f, mut t) = (0, 0);
    unsafe { ck(hipMemGetInfo(&mut f, &mut t)) };
    (f, t)
}

/// Device allocation of `len` elements of `T` on the device current at construction.
pub struct DevBuf<T> {
    ptr: *mut T,
    len: usize,
    _t: PhantomData<T>,
}

impl<T> DevBuf<T> {
    pub fn new(len: usize) -> Self {
        let mut p = std::ptr::null_mut();
        unsafe { ck(hipMalloc(&mut p, (len * size_of::<T>()).max(16))) };
        Self { ptr: p as *mut T, len, _t: PhantomData }
    }

    pub fn zeroed(len: usize, s: Stream) -> Self {
        let b = Self::new(len);
        b.zero(s);
        b
    }

    pub fn from_slice(data: &[T], s: Stream) -> Self {
        let b = Self::new(data.len());
        b.upload(data, s);
        sync(s);
        b
    }

    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn zero(&self, s: Stream) {
        unsafe { ck(hipMemsetAsync(self.ptr as *mut c_void, 0, self.len * size_of::<T>(), s)) }
    }

    /// Async copy; `data` must stay alive until the stream is synchronised.
    pub fn upload(&self, data: &[T], s: Stream) {
        assert!(data.len() <= self.len);
        unsafe { ck(hipMemcpyAsync(self.ptr as *mut c_void, data.as_ptr() as *const c_void, size_of_val(data), HOST_TO_DEVICE, s)) }
    }

    /// Async copy; `out` must stay alive and untouched until the stream is synchronised.
    pub fn download(&self, out: &mut [T], s: Stream) {
        assert!(out.len() <= self.len);
        unsafe { ck(hipMemcpyAsync(out.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, size_of_val(out), DEVICE_TO_HOST, s)) }
    }
}

impl<T> Drop for DevBuf<T> {
    fn drop(&mut self) {
        unsafe { hipFree(self.ptr as *mut c_void) };
    }
}

/// Copy `n` elements from one card's buffer to another's, queued on `s` (the source's stream).
///
/// # Safety
/// Both ranges must lie inside live allocations of the named devices.
pub unsafe fn copy_peer<T>(dst: *mut T, dst_dev: usize, src: *const T, src_dev: usize, n: usize, s: Stream) {
    unsafe { ck(hipMemcpyPeerAsync(dst as *mut c_void, dst_dev as c_int, src as *const c_void, src_dev as c_int, n * size_of::<T>(), s)) }
}

/// Page-locked host staging memory.
pub struct Pinned {
    ptr: *mut u8,
    len: usize,
}

/// hipHostMallocCoherent: kernels on every card see each other's writes to this memory.
const HOST_COHERENT: u32 = 0x4000_0000;

impl Pinned {
    pub fn new(len: usize) -> Self {
        let mut p = std::ptr::null_mut();
        unsafe { ck(hipHostMalloc(&mut p, len, 0)) };
        Self { ptr: p as *mut u8, len }
    }

    /// Zeroed memory that kernels on all cards may read and write directly (atomics only).
    pub fn shared(len: usize) -> Self {
        let mut p = std::ptr::null_mut();
        unsafe {
            ck(hipHostMalloc(&mut p, len, HOST_COHERENT));
            std::ptr::write_bytes(p as *mut u8, 0, len);
        }
        Self { ptr: p as *mut u8, len }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { hipHostFree(self.ptr as *mut c_void) };
    }
}

/// Raw host→device copy from pinned memory, queued on `s`.
pub fn upload_raw(dst: *mut u8, src: *const u8, bytes: usize, s: Stream) {
    unsafe { ck(hipMemcpyAsync(dst as *mut c_void, src as *const c_void, bytes, HOST_TO_DEVICE, s)) }
}

/// Raw device→host copy, queued on `s`.
///
/// # Safety
/// `dst` must be valid for `bytes` until the stream is synchronised.
pub unsafe fn download_raw(dst: *mut u8, src: *const u8, bytes: usize, s: Stream) {
    unsafe { ck(hipMemcpyAsync(dst as *mut c_void, src as *const c_void, bytes, DEVICE_TO_HOST, s)) }
}

/// Zero `bytes` of device memory, queued on `s`.
///
/// # Safety
/// `dst..dst+bytes` must lie inside one device allocation.
pub unsafe fn zero_range(dst: *mut c_void, bytes: usize, s: Stream) {
    unsafe { ck(hipMemsetAsync(dst, 0, bytes, s)) }
}

/// Same-device copy of `n` elements, queued on `s`.
///
/// # Safety
/// Both ranges must lie inside live allocations of the stream's device.
pub unsafe fn copy_dev<T>(dst: *mut T, src: *const T, n: usize, s: Stream) {
    unsafe { ck(hipMemcpyAsync(dst as *mut c_void, src as *const c_void, n * size_of::<T>(), DEVICE_TO_DEVICE, s)) }
}
