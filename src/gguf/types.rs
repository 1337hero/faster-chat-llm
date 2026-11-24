//! GGML tensor types and GGUF metadata values.

use std::fmt;

/// ggml_type ids. Only the ones this engine can read today have block geometry;
/// the rest report `None` so the loader can refuse them cleanly instead of guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        use GgmlType::*;
        match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => IQ2XXS,
            17 => IQ2XS,
            18 => IQ3XXS,
            19 => IQ1S,
            20 => IQ4NL,
            21 => IQ3S,
            22 => IQ2S,
            23 => IQ4XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1M,
            30 => BF16,
            other => Unknown(other),
        }
    }

    /// Number of weights in one storage block.
    pub fn block_elems(self) -> Option<u64> {
        use GgmlType::*;
        Some(match self {
            F32 | F16 | BF16 | I8 | I16 | I32 | I64 | F64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K => 256,
            _ => return None,
        })
    }

    /// Bytes one block occupies on disk.
    pub fn block_bytes(self) -> Option<u64> {
        use GgmlType::*;
        Some(match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            F64 | I64 => 8,
            I8 => 1,
            Q4_0 => 2 + 16, // d:f16 + 16 bytes of 4-bit pairs
            Q4_1 => 2 + 2 + 16,
            Q5_0 => 2 + 4 + 16,
            Q5_1 => 2 + 2 + 4 + 16,
            Q8_0 => 2 + 32,     // d:f16 + 32 int8
            Q8_1 => 2 + 2 + 32, // d:f16 + s:f16 + 32 int8
            Q2K => 256 / 4 + 2 + 2 + 2 + 2,
            Q3K => 256 / 4 + 2 + 1 + 12 + 2 + 2,
            Q4K => 256 / 2 + 2 + 2 + 12,
            Q5K => 256 / 2 + 2 + 2 + 12 + 32,
            Q6K => 256 / 2 + 2 + 12 + 16 + 2,
            Q8K => 256 + 2 + 2 + 4,
            _ => return None,
        })
    }

    /// Total stored bytes for `n_elements` weights.
    pub fn n_bytes(self, n_elements: u64) -> u64 {
        match (self.block_elems(), self.block_bytes()) {
            (Some(be), Some(bb)) if be > 0 => {
                let blocks = n_elements.div_ceil(be);
                blocks * bb
            }
            _ => 0,
        }
    }

    pub fn is_supported(self) -> bool {
        matches!(
            self,
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::Q8_0
        )
    }

    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32",
            F16 => "F16",
            Q4_0 => "Q4_0",
            Q4_1 => "Q4_1",
            Q5_0 => "Q5_0",
            Q5_1 => "Q5_1",
            Q8_0 => "Q8_0",
            Q8_1 => "Q8_1",
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            Q8K => "Q8_K",
            IQ2XXS => "IQ2_XXS",
            IQ2XS => "IQ2_XS",
            IQ3XXS => "IQ3_XXS",
            IQ1S => "IQ1_S",
            IQ4NL => "IQ4_NL",
            IQ3S => "IQ3_S",
            IQ2S => "IQ2_S",
            IQ4XS => "IQ4_XS",
            I8 => "I8",
            I16 => "I16",
            I32 => "I32",
            I64 => "I64",
            F64 => "F64",
            IQ1M => "IQ1_M",
            BF16 => "BF16",
            Unknown(_) => "UNKNOWN",
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgmlType::Unknown(v) => write!(f, "UNKNOWN({v})"),
            other => f.write_str(other.name()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        item_type: u32,
        items: Vec<MetaValue>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) => u64::try_from(*v).ok()?,
            MetaValue::I16(v) => u64::try_from(*v).ok()?,
            MetaValue::I32(v) => u64::try_from(*v).ok()?,
            MetaValue::I64(v) => u64::try_from(*v).ok()?,
            MetaValue::Bool(b) => *b as u64,
            _ => return None,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            MetaValue::F32(v) => *v,
            MetaValue::F64(v) => *v as f32,
            other => other.as_u64()? as f32,
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            MetaValue::U8(_) => "u8",
            MetaValue::I8(_) => "i8",
            MetaValue::U16(_) => "u16",
            MetaValue::I16(_) => "i16",
            MetaValue::U32(_) => "u32",
            MetaValue::I32(_) => "i32",
            MetaValue::F32(_) => "f32",
            MetaValue::Bool(_) => "bool",
            MetaValue::String(_) => "string",
            MetaValue::Array { .. } => "array",
            MetaValue::U64(_) => "u64",
            MetaValue::I64(_) => "i64",
            MetaValue::F64(_) => "f64",
        }
    }
}

impl fmt::Display for MetaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetaValue::Array { item_type, items } => {
                write!(f, "array<{item_type}>[{}]", items.len())?;
                if items.len() <= 8 {
                    let shown: Vec<String> = items.iter().map(|v| v.to_string()).collect();
                    write!(f, " [{}]", shown.join(", "))?;
                } else {
                    write!(f, " [{} .. ]", items[0])?;
                }
                Ok(())
            }
            MetaValue::String(s) => {
                let one_line: String = s
                    .chars()
                    .take(60)
                    .map(|c| if c == '\n' { ' ' } else { c })
                    .collect();
                if s.chars().count() > 60 {
                    write!(f, "\"{one_line}…\" ({} chars)", s.chars().count())
                } else {
                    write!(f, "{one_line:?}")
                }
            }
            other => write!(f, "{other:?}"),
        }
    }
}
