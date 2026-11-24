//! Qwen3.8 byte-level BPE, read straight out of the GGUF.
//!
//! Vocab and merges come from `tokenizer.ggml.*`; the pre-tokenisation split follows
//! llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35` (the model stores `qwen35`, which differs from
//! `qwen2` only by treating combining marks as letters):
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//! |\p{N}
//! | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//! |\s*[\r\n]+
//! |\s+(?!\S)
//! |\s+
//! ```
//!
//! Hand-written rather than a regex because rule 6 needs a negative lookahead and the
//! alternation order is load-bearing.
//!
//! BPE operates on symbol ids (byte alphabet 0..=255, then merge results), never on strings,
//! so a merge is a hash lookup and an index, not a string concat.

use std::collections::HashMap;
use std::sync::OnceLock;

use unicode_general_category::{GeneralCategory as GC, get_general_category};

use crate::gguf::Gguf;

/// Token type 2 is "unused"; unknown merges can only happen on a corrupt vocab.
pub const UNK: u32 = 0;

fn byte_to_char_table() -> &'static [char; 256] {
    static T: OnceLock<[char; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut table = ['\0'; 256];
        let mut used = [false; 256];
        let mut mark = |b: u16| {
            used[b as usize] = true;
            table[b as usize] = char::from_u32(b as u32).unwrap();
        };
        for b in 33..=126u16 {
            mark(b);
        }
        for b in 161..=172u16 {
            mark(b);
        }
        for b in 174..=255u16 {
            mark(b);
        }
        let mut n = 0u32;
        for b in 0..256usize {
            if !used[b] {
                table[b] = char::from_u32(256 + n).unwrap();
                n += 1;
            }
        }
        table
    })
}

fn char_to_byte_table() -> &'static HashMap<char, u8> {
    static T: OnceLock<HashMap<char, u8>> = OnceLock::new();
    T.get_or_init(|| {
        let b2c = byte_to_char_table();
        (0..256usize).map(|b| (b2c[b], b as u8)).collect()
    })
}

/// Encode raw bytes the way GPT-2's byte-level BPE represents them.
pub fn byte_level_encode(bytes: &[u8]) -> String {
    let t = byte_to_char_table();
    bytes.iter().map(|b| t[*b as usize]).collect()
}

fn is_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GC::UppercaseLetter
            | GC::LowercaseLetter
            | GC::TitlecaseLetter
            | GC::ModifierLetter
            | GC::OtherLetter
    )
}

fn is_mark(c: char) -> bool {
    matches!(
        get_general_category(c),
        GC::NonspacingMark | GC::SpacingMark | GC::EnclosingMark
    )
}

fn is_number(c: char) -> bool {
    matches!(
        get_general_category(c),
        GC::DecimalNumber | GC::LetterNumber | GC::OtherNumber
    )
}

fn is_letter_or_mark(c: char) -> bool {
    is_letter(c) || is_mark(c)
}

/// Split `text` into pre-tokenisation pieces, in the order llama.cpp's QWEN35 regex would.
pub fn pretokenise(text: &str) -> Vec<String> {
    let cs: Vec<char> = text.chars().collect();
    let n = cs.len();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < n {
        let c = cs[i];
        let mut end: Option<usize> = None;

        // 1. contractions
        if c == '\'' && i + 1 < n {
            let d = cs[i + 1];
            if matches!(d, 's' | 'S' | 't' | 'T' | 'm' | 'M' | 'd' | 'D') {
                end = Some(i + 2);
            } else if i + 2 < n {
                let e = cs[i + 2];
                if matches!(
                    (d, e),
                    ('r' | 'R', 'e' | 'E') | ('v' | 'V', 'e' | 'E') | ('l' | 'L', 'l' | 'L')
                ) {
                    end = Some(i + 3);
                }
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if end.is_none() {
            if is_letter_or_mark(c) {
                let mut j = i;
                while j < n && is_letter_or_mark(cs[j]) {
                    j += 1;
                }
                end = Some(j);
            } else if !matches!(c, '\r' | '\n')
                && !is_letter(c)
                && !is_number(c)
                && i + 1 < n
                && is_letter_or_mark(cs[i + 1])
            {
                let mut j = i + 1;
                while j < n && is_letter_or_mark(cs[j]) {
                    j += 1;
                }
                end = Some(j);
            }
        }

        // 3. \p{N}
        if end.is_none() && is_number(c) {
            end = Some(i + 1);
        }

        // 4.  ?<punctuation run>[\r\n]*
        if end.is_none() {
            let mut j = i;
            if c == ' ' && i + 1 < n {
                let d = cs[i + 1];
                if !d.is_whitespace() && !is_letter_or_mark(d) && !is_number(d) {
                    j = i + 1;
                }
            }
            if j < n {
                let d = cs[j];
                if !d.is_whitespace() && !is_letter_or_mark(d) && !is_number(d) {
                    while j < n
                        && !cs[j].is_whitespace()
                        && !is_letter_or_mark(cs[j])
                        && !is_number(cs[j])
                    {
                        j += 1;
                    }
                    while j < n && matches!(cs[j], '\r' | '\n') {
                        j += 1;
                    }
                    end = Some(j);
                }
            }
        }

        // 5-7. whitespace: \s*[\r\n]+ then \s+(?!\S) then \s+
        if end.is_none() && c.is_whitespace() {
            let mut j = i;
            while j < n && cs[j].is_whitespace() {
                j += 1;
            }
            end = match (i..j).rev().find(|&k| matches!(cs[k], '\r' | '\n')) {
                Some(k) => Some(k + 1),
                None if j < n && j - i > 1 => Some(j - 1),
                None => Some(j),
            };
        }

        let e = end.unwrap_or(i + 1);
        out.push(cs[i..e].iter().collect());
        i = e;
    }
    out
}

pub struct Tokenizer {
    id_to_bytes: Vec<Vec<u8>>,
    /// (left sym, right sym) -> (rank, merged sym)
    merges: HashMap<(u32, u32), (u32, u32)>,
    sym_bytes: Vec<Vec<u8>>,
    // `sym_bytes` may hold pieces the vocab has no token for; `sym_token` maps a symbol to
    // its vocab id when one exists.
    sym_token: Vec<u32>,
    pub bos: u32,
    pub eos: u32,
    pub token_types: Vec<i32>,
    specials: Vec<(u32, String)>,
    add_bos: bool,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self, String> {
        use crate::gguf::types::MetaValue;

        let tokens = match g.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array { items, .. }) => items,
            _ => return Err("tokenizer.ggml.tokens missing".into()),
        };
        let merges = match g.get("tokenizer.ggml.merges") {
            Some(MetaValue::Array { items, .. }) => items,
            _ => return Err("tokenizer.ggml.merges missing".into()),
        };
        let token_types: Vec<i32> = match g.get("tokenizer.ggml.token_type") {
            Some(MetaValue::Array { items, .. }) => items
                .iter()
                .map(|v| match v {
                    MetaValue::I32(x) => *x,
                    other => other.as_u64().unwrap_or(1) as i32,
                })
                .collect(),
            _ => vec![1; tokens.len()],
        };

        let c2b = char_to_byte_table();
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_bytes = Vec::with_capacity(tokens.len());
        let mut specials = Vec::new();

        for (id, t) in tokens.iter().enumerate() {
            let s = match t {
                MetaValue::String(s) => s,
                _ => return Err(format!("token {id} is not a string")),
            };
            let mut bytes = Vec::with_capacity(s.len());
            for ch in s.chars() {
                match c2b.get(&ch) {
                    Some(b) => bytes.push(*b),
                    // Control/added tokens (`<|im_start|>`) are literal text, not byte-level.
                    None => bytes.extend_from_slice(ch.encode_utf8(&mut [0u8; 4]).as_bytes()),
                }
            }
            token_to_id.insert(s.clone(), id as u32);
            id_to_bytes.push(bytes);
            let ty = token_types.get(id).copied().unwrap_or(1);
            if ty == 3 || ty == 4 {
                specials.push((id as u32, s.clone()));
            }
        }
        specials.sort_by_key(|token| std::cmp::Reverse(token.1.len()));

        let mut sym_bytes: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut sym_id: HashMap<Vec<u8>, u32> = sym_bytes
            .iter()
            .enumerate()
            .map(|(i, b)| (b.clone(), i as u32))
            .collect();

        let mut merges_map: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let s = match m {
                MetaValue::String(s) => s,
                _ => continue,
            };
            let mut it = s.split(' ');
            let (a, b) = match (it.next(), it.next()) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };
            let to_bytes = |piece: &str| -> Option<Vec<u8>> {
                piece.chars().map(|ch| c2b.get(&ch).copied()).collect()
            };
            let (ab, bb) = match (to_bytes(a), to_bytes(b)) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            };
            // Merge sides are written as symbol *strings*, so a side can be a multi-byte
            // symbol that this file never produced as a merge result. Intern on demand.
            let intern = |bytes: Vec<u8>,
                          sym_bytes: &mut Vec<Vec<u8>>,
                          sym_id: &mut HashMap<Vec<u8>, u32>|
             -> u32 {
                if let Some(&id) = sym_id.get(&bytes) {
                    return id;
                }
                let id = sym_bytes.len() as u32;
                sym_bytes.push(bytes.clone());
                sym_id.insert(bytes, id);
                id
            };
            let ia = intern(ab.clone(), &mut sym_bytes, &mut sym_id);
            let ib = intern(bb.clone(), &mut sym_bytes, &mut sym_id);
            let mut merged = ab.clone();
            merged.extend_from_slice(&bb);
            let mid = intern(merged, &mut sym_bytes, &mut sym_id);
            merges_map.insert((ia, ib), (rank as u32, mid));
        }

        // Vocab ids for symbols: a symbol is only emittable if the vocab holds its bytes.
        let sym_token: Vec<u32> = sym_bytes
            .iter()
            .map(|b| {
                token_to_id
                    .get(&byte_level_encode(b))
                    .copied()
                    .unwrap_or(UNK)
            })
            .collect();

        Ok(Self {
            id_to_bytes,
            merges: merges_map,
            sym_bytes,
            sym_token,
            bos: g.u64_of("tokenizer.ggml.bos_token_id").unwrap_or(0) as u32,
            eos: g.u64_of("tokenizer.ggml.eos_token_id").unwrap_or(0) as u32,
            token_types,
            specials,
            add_bos: g.bool_of("tokenizer.ggml.add_bos_token").unwrap_or(true),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    pub fn validate_id(&self, value: u64) -> Result<u32, String> {
        let id = u32::try_from(value).map_err(|_| format!("token ID {value} exceeds u32"))?;
        self.validate_ids(&[id])?;
        Ok(id)
    }

    pub fn validate_ids(&self, ids: &[u32]) -> Result<(), String> {
        if let Some(id) = ids
            .iter()
            .find(|&&id| id as usize >= self.vocab_size() || id > i32::MAX as u32)
        {
            return Err(format!("token ID {id} outside vocabulary or kernel range"));
        }
        Ok(())
    }

    pub fn decode_id(&self, id: u32) -> &[u8] {
        &self.id_to_bytes[id as usize]
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(b) = self.id_to_bytes.get(id as usize) {
                out.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = piece.iter().map(|b| *b as u32).collect();
        loop {
            // lowest-rank adjacent pair; ties impossible (ranks are unique)
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1]))
                    && best.is_none_or(|(r, _, _)| rank < r)
                {
                    best = Some((rank, i, merged));
                }
            }
            let (_, idx, merged) = match best {
                Some(b) => b,
                None => break,
            };
            syms[idx] = merged;
            syms.remove(idx + 1);
        }
        out.extend(syms.iter().map(|s| self.sym_token[*s as usize]));
        let _ = &self.sym_bytes;
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            out.push(self.bos);
        }
        let mut rest = text;
        while !rest.is_empty() {
            let mut hit: Option<(usize, u32, usize)> = None;
            for (id, s) in &self.specials {
                if let Some(pos) = rest.find(s.as_str())
                    && hit.is_none_or(|(p, _, _)| pos < p)
                {
                    hit = Some((pos, *id, s.len()));
                }
            }
            match hit {
                Some((pos, id, len)) => {
                    if pos > 0 {
                        self.encode_plain(&rest[..pos], &mut out);
                    }
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    rest = "";
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pretokenise(text) {
            self.bpe(piece.as_bytes(), out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_words_and_space() {
        assert_eq!(pretokenise("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn split_digits_individually() {
        assert_eq!(
            pretokenise("3.14159"),
            vec!["3", ".", "1", "4", "1", "5", "9"]
        );
    }

    #[test]
    fn split_contraction() {
        assert_eq!(pretokenise("it's"), vec!["it", "'s"]);
        assert_eq!(pretokenise("we'll"), vec!["we", "'ll"]);
    }

    #[test]
    fn split_newline_keeps_leading_space() {
        assert_eq!(pretokenise("a\nb"), vec!["a", "\n", "b"]);
        assert_eq!(pretokenise("a \n\nb"), vec!["a", " \n\n", "b"]);
    }

    #[test]
    fn split_punctuation_run() {
        assert_eq!(pretokenise("foo!!!"), vec!["foo", "!!!"]);
        assert_eq!(pretokenise("end.\n"), vec!["end", ".\n"]);
    }

    #[test]
    fn byte_level_roundtrip() {
        assert_eq!(byte_level_encode(b"hello world"), "helloĠworld");
    }
}
