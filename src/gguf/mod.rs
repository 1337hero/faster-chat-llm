//! GGUF v2/v3 reader: header, metadata, tensor table, and raw tensor access.
//!
//! The file is memory-mapped for metadata only. Tensor payloads are pulled with
//! `pread` on demand so a 29 GB GGUF never has to sit in page cache at once.

pub mod dequant;
pub mod types;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use types::{GgmlType, MetaValue};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    BadMagic(u32),
    UnsupportedVersion(u32),
    Truncated { at: u64, need: u64 },
    BadValueType(u32),
    BadCount(u64),
    Utf8(std::string::FromUtf8Error),
    MissingTensor(String),
    BadGeometry(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "io: {e}"),
            GgufError::BadMagic(m) => write!(f, "not a GGUF file (magic {m:#010x})"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            GgufError::Truncated { at, need } => write!(f, "truncated at {at}, need {need} bytes"),
            GgufError::BadValueType(t) => write!(f, "unknown metadata value type {t}"),
            GgufError::BadCount(n) => write!(f, "implausible count {n}"),
            GgufError::Utf8(e) => write!(f, "invalid utf-8: {e}"),
            GgufError::MissingTensor(n) => write!(f, "no such tensor: {n}"),
            GgufError::BadGeometry(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for GgufError {}
impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Byte offset relative to the start of the tensor data section.
    pub offset: u64,
    /// Row length in elements = dims[0] (ggml convention: dims are fastest-first).
    pub n_elements: u64,
}

impl TensorInfo {
    pub fn n_bytes(&self) -> u64 {
        self.ty.n_bytes(self.n_elements)
    }

    /// Logical shape, slowest-first (the order people say out loud).
    pub fn shape(&self) -> Vec<u64> {
        let mut s = self.dims.clone();
        s.reverse();
        s
    }

    pub fn dtype_name(&self) -> String {
        self.ty.to_string()
    }
}

pub struct Gguf {
    path: PathBuf,
    file: File,
    pub version: u32,
    pub alignment: u64,
    /// Absolute file offset of the tensor data section.
    pub data_offset: u64,
    pub meta: BTreeMap<String, MetaValue>,
    tensors: Vec<TensorInfo>,
    by_name: BTreeMap<String, usize>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated {
            at: self.pos as u64,
            need: n as u64,
        })?;
        if end > self.buf.len() {
            return Err(GgufError::Truncated {
                at: self.pos as u64,
                need: n as u64,
            });
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > 1 << 26 {
            return Err(GgufError::BadCount(len));
        }
        String::from_utf8(self.take(len as usize)?.to_vec()).map_err(GgufError::Utf8)
    }
}

fn read_value(c: &mut Cursor<'_>, vtype: u32) -> Result<MetaValue, GgufError> {
    Ok(match vtype {
        0 => MetaValue::U8(c.u8()?),
        1 => MetaValue::I8(c.take(1)?[0] as i8),
        2 => MetaValue::U16(u16::from_le_bytes(c.take(2)?.try_into().unwrap())),
        3 => MetaValue::I16(i16::from_le_bytes(c.take(2)?.try_into().unwrap())),
        4 => MetaValue::U32(c.u32()?),
        5 => MetaValue::I32(i32::from_le_bytes(c.take(4)?.try_into().unwrap())),
        6 => MetaValue::F32(f32::from_le_bytes(c.take(4)?.try_into().unwrap())),
        7 => MetaValue::Bool(c.u8()? != 0),
        8 => MetaValue::String(c.string()?),
        9 => {
            let item_type = c.u32()?;
            let count = c.u64()?;
            if count > 1 << 32 {
                return Err(GgufError::BadCount(count));
            }
            let mut items = Vec::with_capacity((count as usize).min(4096));
            for _ in 0..count {
                items.push(read_value(c, item_type)?);
            }
            MetaValue::Array { item_type, items }
        }
        10 => MetaValue::U64(c.u64()?),
        11 => MetaValue::I64(i64::from_le_bytes(c.take(8)?.try_into().unwrap())),
        12 => MetaValue::F64(f64::from_le_bytes(c.take(8)?.try_into().unwrap())),
        t => return Err(GgufError::BadValueType(t)),
    })
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();

        // The header + metadata + tensor table is a few MB at most; read a bounded prefix.
        let head_len = (file_len as usize).min(64 << 20);
        let mut head = vec![0u8; head_len];
        file.read_exact(&mut head)?;

        let mut c = Cursor::new(&head);
        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        if n_tensors > 1 << 20 || n_kv > 1 << 20 {
            return Err(GgufError::BadCount(n_tensors.max(n_kv)));
        }

        let mut meta = BTreeMap::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let vtype = c.u32()?;
            let value = read_value(&mut c, vtype)?;
            meta.insert(key, value);
        }

        let alignment = meta
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);

        if !alignment.is_power_of_two() {
            return Err(GgufError::BadGeometry(format!(
                "invalid GGUF alignment: {alignment}"
            )));
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims > 8 {
                return Err(GgufError::BadGeometry(format!("{name}: {n_dims} dims")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ty = GgmlType::from_u32(c.u32()?);
            let offset = c.u64()?;
            let n_elements = dims.iter().product::<u64>();
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                n_elements,
            });
        }

        let header_end = c.pos as u64;
        let data_offset = header_end
            .checked_add(alignment - 1)
            .map(|end| end & !(alignment - 1))
            .ok_or_else(|| GgufError::BadGeometry("GGUF alignment overflow".into()))?;
        if data_offset > file_len {
            return Err(GgufError::Truncated {
                at: header_end,
                need: alignment,
            });
        }

        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok(Self {
            path,
            file,
            version,
            alignment,
            data_offset,
            meta,
            tensors,
            by_name,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))
    }

    /// Read a tensor's stored bytes (no dequantisation).
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>, GgufError> {
        let info = self.tensor(name)?;
        let mut buf = vec![0u8; info.n_bytes() as usize];
        let start = self.data_offset + info.offset;
        self.file.read_exact_at(&mut buf, start)?;
        Ok(buf)
    }

    /// Read `byte_len` bytes of the file starting at `file_offset`.
    pub fn pread(&self, file_offset: u64, buf: &mut [u8]) -> Result<(), GgufError> {
        self.file.read_exact_at(buf, file_offset)?;
        Ok(())
    }

    pub fn file_len(&self) -> Result<u64, GgufError> {
        Ok(self.file.metadata()?.len())
    }

    // ---- metadata helpers ----

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.meta.get(key)
    }

    pub fn u64_of(&self, key: &str) -> Option<u64> {
        self.meta.get(key).and_then(|v| v.as_u64())
    }

    pub fn str_of(&self, key: &str) -> Option<&str> {
        match self.meta.get(key) {
            Some(MetaValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn f32_of(&self, key: &str) -> Option<f32> {
        self.meta.get(key).and_then(|v| v.as_f32())
    }

    pub fn bool_of(&self, key: &str) -> Option<bool> {
        match self.meta.get(key) {
            Some(MetaValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Sanity check the tensor table against the file length.
    pub fn validate_layout(&self) -> Result<(), GgufError> {
        let file_len = self.file_len()?;
        for t in &self.tensors {
            let end = self.data_offset + t.offset + t.n_bytes();
            if end > file_len {
                return Err(GgufError::BadGeometry(format!(
                    "{}: data ends at {end} > file {file_len}",
                    t.name
                )));
            }
        }
        Ok(())
    }
}
