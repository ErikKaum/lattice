//! A purpose-built, IDs-only WordPiece tokenizer.
//!
//! The HF `tokenizers` crate's `encode`/`encode_batch` build full `Encoding`
//! objects (offsets, alignment, masks, type IDs, special tokens) of which
//! `LatticeTokenizer` uses only `get_ids()`. This module implements just the
//! forward text -> IDs mapping for the BertNormalizer + BertPreTokenizer +
//! WordPiece pipeline lattice-retrieval ships — the only pipeline any shipped
//! artifact uses, since every variant is sliced from the same base model —
//! parsing `tokenizer.json` directly so the `tokenizers` crate (and its
//! native `onig` dependency) isn't needed at runtime at all. It's kept as a
//! dev-dependency purely to check parity in tests.
//!
//! Literal added-token text (e.g. `[MASK]`, `[CLS]`) is matched and resolved
//! to its id before normalization, matching the crate's `AddedVocabulary`
//! step — see [`FastWordPiece::split_added_tokens`].

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

/// IDs-only WordPiece tokenizer for the BERT pipeline.
#[derive(Debug, Clone)]
pub(crate) struct FastWordPiece {
    vocab: HashMap<String, u32>,
    /// Vocab IDs of single-ASCII-byte tokens (-1 when absent), so isolated
    /// punctuation words skip the map lookup.
    single_byte: [i32; 128],
    unk_id: Option<u32>,
    max_input_chars_per_word: usize,
    continuing_subword_prefix: String,
    // BertNormalizer settings
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
    /// `normalized: false` entries from `added_tokens`, longest-content-first.
    /// The crate's `AddedVocabulary::extract_and_normalize` matches these as
    /// literal substrings of the raw text *before* the normalizer/pre-tokenizer
    /// run, regardless of `add_special_tokens` (that flag only gates the
    /// post-processor's CLS/SEP insertion, not this literal-match step) — so a
    /// literal `[MASK]` mid-sentence must resolve to its added-token id here
    /// too, not fall through to WordPiece on the lowercased text. Only
    /// `normalized: false` entries are handled; lattice-retrieval's
    /// tokenizer.json has none with `normalized: true` (that would need
    /// matching against normalized text instead — unimplemented, unneeded).
    added_tokens: Vec<(String, u32)>,
    /// First byte of every entry in `added_tokens`, so `split_added_tokens`
    /// can skip the `starts_with` scan at byte positions that can't possibly
    /// start a match — added tokens are rare (typically absent) in bulk
    /// embedding text, so this keeps the common case a single array lookup
    /// per byte instead of up to `added_tokens.len()` string comparisons.
    added_token_first_bytes: [bool; 256],
}

/// A slice of input text as it flows through [`FastWordPiece::split_added_tokens`]:
/// either a literal added-token match (already resolved to its id) or a span
/// of ordinary text still needing normalization/pre-tokenization/WordPiece.
enum Piece<'a> {
    Added(u32),
    Text(&'a str),
}

impl FastWordPiece {
    /// Load and parse `tokenizer.json` at `path`. Errors if it isn't the
    /// BertNormalizer + BertPreTokenizer + WordPiece pipeline — lattice-
    /// retrieval never ships anything else, so there is no fallback pipeline
    /// to degrade to.
    pub(crate) fn load(path: &Path) -> Result<FastWordPiece> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading tokenizer json from {}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parsing tokenizer json from {}", path.display()))?;
        Self::from_json(&json).with_context(|| {
            format!(
                "{} is not a BertNormalizer + BertPreTokenizer + WordPiece pipeline",
                path.display()
            )
        })
    }

    fn from_json(json: &serde_json::Value) -> Result<FastWordPiece> {
        let model = json.get("model").context("missing `model`")?;
        if model.get("type").and_then(|t| t.as_str()) != Some("WordPiece") {
            bail!("model.type is not \"WordPiece\"");
        }
        if json
            .get("pre_tokenizer")
            .and_then(|p| p.get("type"))
            .and_then(|t| t.as_str())
            != Some("BertPreTokenizer")
        {
            bail!("pre_tokenizer.type is not \"BertPreTokenizer\"");
        }
        let norm = json.get("normalizer").context("missing `normalizer`")?;
        if norm.get("type").and_then(|t| t.as_str()) != Some("BertNormalizer") {
            bail!("normalizer.type is not \"BertNormalizer\"");
        }
        let lowercase = norm
            .get("lowercase")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let clean_text = norm
            .get("clean_text")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let handle_chinese_chars = norm
            .get("handle_chinese_chars")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        // `strip_accents: null` follows the lowercase setting, matching HuggingFace.
        let strip_accents = norm
            .get("strip_accents")
            .and_then(|v| v.as_bool())
            .unwrap_or(lowercase);

        // model.vocab is the flat token -> id map; for lattice-retrieval's
        // tokenizer.json every added special token (PAD/UNK/CLS/SEP/MASK) is
        // already present in it, so no separate `added_tokens` merge is needed.
        let vocab: HashMap<String, u32> =
            serde_json::from_value(model.get("vocab").context("missing model.vocab")?.clone())
                .context("model.vocab is not a string -> id map")?;
        let unk_token = model
            .get("unk_token")
            .and_then(|v| v.as_str())
            .unwrap_or("[UNK]");
        let unk_id = vocab.get(unk_token).copied();
        let continuing_subword_prefix = model
            .get("continuing_subword_prefix")
            .and_then(|v| v.as_str())
            .unwrap_or("##")
            .to_string();
        let max_input_chars_per_word = model
            .get("max_input_chars_per_word")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;

        let mut single_byte = [-1_i32; 128];
        for (b, slot) in single_byte.iter_mut().enumerate() {
            if let Some(&id) = vocab.get(&(b as u8 as char).to_string()) {
                *slot = id as i32;
            }
        }

        let mut added_tokens: Vec<(String, u32)> = Vec::new();
        for t in json.get("added_tokens").and_then(|v| v.as_array()).into_iter().flatten() {
            let content = t.get("content").and_then(|c| c.as_str()).context("added_tokens entry missing content")?;
            if content.is_empty() {
                bail!("added_tokens entry has empty content");
            }
            let id = t.get("id").and_then(|v| v.as_u64()).context("added_tokens entry missing id")? as u32;
            let normalized = t.get("normalized").and_then(|v| v.as_bool()).unwrap_or(true);
            if normalized {
                bail!(
                    "added token {content:?} has normalized: true, which FastWordPiece doesn't implement (only normalized: false literal matching is supported)"
                );
            }
            added_tokens.push((content.to_string(), id));
        }
        // Longest-content-first so a shorter added token can't shadow a longer
        // one that starts with the same prefix.
        added_tokens.sort_by_key(|(content, _)| std::cmp::Reverse(content.len()));

        let mut added_token_first_bytes = [false; 256];
        for (content, _) in &added_tokens {
            added_token_first_bytes[content.as_bytes()[0] as usize] = true;
        }

        Ok(FastWordPiece {
            vocab,
            single_byte,
            unk_id,
            max_input_chars_per_word,
            continuing_subword_prefix,
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
            added_tokens,
            added_token_first_bytes,
        })
    }

    /// Encode text into token IDs, matching `encode_batch_fast(...).get_ids()`
    /// (unknown words emit the unk id, which the caller filters out).
    pub(crate) fn encode_ids(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut scratch = String::new();
        for piece in self.split_added_tokens(text) {
            match piece {
                Piece::Added(id) => out.push(id),
                Piece::Text(t) => {
                    let normalized = self.normalize(t);
                    for word in pre_tokenize(&normalized) {
                        self.word_to_ids(word, &mut out, &mut scratch);
                    }
                }
            }
        }
        out
    }

    /// Split `text` on literal, case-sensitive matches of `added_tokens`
    /// (greedy longest-match-first, left to right), leaving everything else
    /// as `Text` pieces for the normal normalize/pre-tokenize/WordPiece path.
    fn split_added_tokens<'a>(&self, text: &'a str) -> Vec<Piece<'a>> {
        if self.added_tokens.is_empty() {
            return vec![Piece::Text(text)];
        }
        let bytes = text.as_bytes();
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < text.len() {
            let hit = self.added_token_first_bytes[bytes[i] as usize]
                .then(|| self.added_tokens.iter().find(|(tok, _)| text[i..].starts_with(tok.as_str())))
                .flatten();
            if let Some((tok, id)) = hit {
                if start < i {
                    pieces.push(Piece::Text(&text[start..i]));
                }
                pieces.push(Piece::Added(*id));
                i += tok.len();
                start = i;
            } else {
                i += text[i..].chars().next().expect("i < text.len()").len_utf8();
            }
        }
        if start < text.len() {
            pieces.push(Piece::Text(&text[start..]));
        }
        pieces
    }

    // --- WordPiece ---------------------------------------------------------

    fn push_unk(&self, out: &mut Vec<u32>) {
        if let Some(u) = self.unk_id {
            out.push(u);
        }
    }

    /// Greedy longest-match-first. A word that cannot be fully tokenized (or
    /// exceeds `max_input_chars_per_word`) maps to a single unk token.
    fn word_to_ids(&self, word: &str, out: &mut Vec<u32>, scratch: &mut String) {
        // Rune count never exceeds byte length, so short words skip the scan.
        if word.len() > self.max_input_chars_per_word
            && word.chars().count() > self.max_input_chars_per_word
        {
            self.push_unk(out);
            return;
        }

        if word.len() == 1 {
            let b = word.as_bytes()[0];
            let id = self.single_byte[b as usize];
            if id >= 0 {
                out.push(id as u32);
            } else {
                self.push_unk(out);
            }
            return;
        }

        let mark = out.len();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while end > start {
                let sub = &word[start..end];
                let id = if start == 0 {
                    self.vocab.get(sub).copied()
                } else {
                    scratch.clear();
                    scratch.push_str(&self.continuing_subword_prefix);
                    scratch.push_str(sub);
                    self.vocab.get(scratch.as_str()).copied()
                };
                if let Some(id) = id {
                    found = Some(id);
                    break;
                }
                // Step back one char (start/end stay on char boundaries).
                end -= sub.chars().next_back().unwrap().len_utf8();
            }
            match found {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    out.truncate(mark);
                    self.push_unk(out);
                    return;
                }
            }
        }
    }

    // --- BertNormalizer ----------------------------------------------------

    fn normalize(&self, input: &str) -> String {
        if input.is_empty() {
            return String::new();
        }
        if input.is_ascii() {
            self.normalize_ascii(input)
        } else {
            self.normalize_unicode(input)
        }
    }

    /// Fused clean + lowercase + trim over ASCII bytes. Accent stripping (NFD
    /// is identity below 0x80) and Chinese handling are no-ops for ASCII.
    fn normalize_ascii(&self, input: &str) -> String {
        let mut buf = Vec::with_capacity(input.len());
        for &orig in input.as_bytes() {
            let mut b = orig;
            if self.clean_text {
                match b {
                    b'\t' | b'\n' | b'\r' => b = b' ',
                    0..=0x1f | 0x7f => continue,
                    _ => {}
                }
            }
            if self.lowercase && b.is_ascii_uppercase() {
                b += b'a' - b'A';
            }
            buf.push(b);
        }
        let mut start = 0;
        let mut end = buf.len();
        while start < end && is_ascii_space(buf[start]) {
            start += 1;
        }
        while end > start && is_ascii_space(buf[end - 1]) {
            end -= 1;
        }
        // Bytes stayed ASCII throughout, so this is valid UTF-8.
        String::from_utf8(buf[start..end].to_vec()).unwrap()
    }

    fn normalize_unicode(&self, input: &str) -> String {
        // Fused clean + Chinese-char pass.
        let mut chars: Vec<char> = Vec::with_capacity(input.len());
        for c in input.chars() {
            if self.clean_text {
                if c == '\0' || c == '\u{fffd}' || is_control(c) {
                    continue;
                }
                if is_whitespace(c) {
                    chars.push(' ');
                    continue;
                }
            }
            if self.handle_chinese_chars && is_chinese_char(c) {
                chars.push(' ');
                chars.push(c);
                chars.push(' ');
                continue;
            }
            chars.push(c);
        }
        let staged: String = chars.into_iter().collect();

        let processed: String = if self.strip_accents {
            // NFD, drop nonspacing marks, then (optionally) lowercase; left
            // decomposed, matching HuggingFace.
            let mut out = String::with_capacity(staged.len());
            for c in staged.nfd() {
                if c.is_mark_nonspacing() {
                    continue;
                }
                if self.lowercase {
                    out.extend(c.to_lowercase());
                } else {
                    out.push(c);
                }
            }
            out
        } else if self.lowercase {
            staged.chars().flat_map(|c| c.to_lowercase()).collect()
        } else {
            staged
        };

        processed
            .trim_matches(|c: char| c.is_whitespace())
            .to_string()
    }
}

// --- BertPreTokenizer -------------------------------------------------------

/// ASCII byte class: 0 = word, 1 = space, 2 = punctuation. Agrees with the
/// rune-level predicates on every value below 0x80.
fn ascii_class(b: u8) -> u8 {
    match b {
        b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r' => 1,
        b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~' => 2,
        _ => 0,
    }
}

/// Split like HuggingFace's BertPreTokenizer: drop whitespace, isolate each
/// punctuation char as its own word. Returned words are zero-copy substrings.
fn pre_tokenize(s: &str) -> Vec<&str> {
    let bytes = s.as_bytes();
    let mut words = Vec::with_capacity(s.len() / 4 + 1);
    let mut start: isize = -1;
    let mut i = 0;
    while i < s.len() {
        let (class, size) = if bytes[i] < 0x80 {
            (ascii_class(bytes[i]), 1)
        } else {
            let c = s[i..].chars().next().unwrap();
            let class = if c.is_whitespace() {
                1
            } else if c.is_ascii_punctuation() || c.is_punctuation() {
                2
            } else {
                0
            };
            (class, c.len_utf8())
        };
        match class {
            1 => {
                if start >= 0 {
                    words.push(&s[start as usize..i]);
                    start = -1;
                }
            }
            2 => {
                if start >= 0 {
                    words.push(&s[start as usize..i]);
                    start = -1;
                }
                words.push(&s[i..i + size]);
            }
            _ => {
                if start < 0 {
                    start = i as isize;
                }
            }
        }
        i += size;
    }
    if start >= 0 {
        words.push(&s[start as usize..]);
    }
    words
}

// --- Unicode predicates (match HuggingFace's bert normalizer) ----------------

fn is_ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => c.is_other(),
    }
}

fn is_chinese_char(c: char) -> bool {
    let r = c as u32;
    (0x4E00..=0x9FFF).contains(&r)
        || (0x3400..=0x4DBF).contains(&r)
        || (0x20000..=0x2A6DF).contains(&r)
        || (0x2A700..=0x2B73F).contains(&r)
        || (0x2B740..=0x2B81F).contains(&r)
        || (0x2B920..=0x2CEAF).contains(&r)
        || (0xF900..=0xFAFF).contains(&r)
        || (0x2F800..=0x2FA1F).contains(&r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::Tokenizer;

    fn corpus() -> Vec<&'static str> {
        vec![
            // --- existing ---
            "Hello, world!",
            "The QUICK brown fox jumps.",
            "café résumé naïve fiancé",
            "Ünïçödé ACCENTS Straße",
            "你好世界 mixed 测试 text",
            "a  b\tc\n  d",
            "supercalifragilisticexpialidocious antidisestablishmentarianism",
            "email@example.com https://x.co/y?z=1 $5.00 <html> C++ #hashtag",
            "123 456.78 -90 1e10",
            "",
            "   ",
            "MiXeD CaSe WoRdS 42",

            // --- clean_text: control/format chars removed, whitespace→space ---
            "a\u{0007}b\u{0000}c",          // bell + NUL removed
            "a\u{000b}b\u{000c}c",          // vertical tab / form feed (Cc → removed, NOT spaced)
            "a\u{007f}b",                   // DEL removed
            "a\u{200b}b\u{feff}c",          // zero-width space + BOM (Cf → removed)
            "a\u{00a0}b\u{2009}c",          // NBSP + thin space (Zs → become space → split)
            "a\u{fffd}b",                   // replacement char removed
            "\u{0001}\u{0002}\u{0003}",     // all-control → empty after normalize
            "  leading and trailing  ",     // edge whitespace trim
            "\t\n\r",                       // whitespace/control only

            // --- casing that changes length or is context-sensitive ---
            "İstanbul",                     // U+0130: NFD→I+dot, strip dot, lower→i
            "STRASSE straße GROSS",         // eszett upper/lower asymmetry
            "ﬁle ﬀ ﬄ",                      // ligatures (fi, ff, ffl)
            "ΟΔΟΣ Σίσυφος",                 // Greek final-sigma: verify char-level (no context)
            "ＦＵＬＬＷＩＤＴＨ ａｂｃ",          // fullwidth latin casing

            // --- accents / combining marks, both compositions ---
            "e\u{0301} a\u{0300} n\u{0303}",   // pre-decomposed base+combining
            "é à ñ ü ç",                       // precomposed (NFD path)
            "Å Ω µ",                           // chars with compatibility/canonical quirks
            "क्षत्रि",                        // Devanagari combining (non-Latin marks)

            // --- CJK / handle_chinese_chars boundaries and non-CJK scripts ---
            "一鿿㐀䶿",                       // BMP CJK range endpoints
            "𠀀 𪛖 豈",                       // ext-B (U+20000) + compatibility (U+F900)
            "你好world世界",                  // CJK adjacent to latin, no spaces
            "，。、！？；：",                  // fullwidth/CJK punctuation (category Po → isolate)
            "こんにちは カタカナ 안녕하세요",   // kana + hangul: must NOT be space-wrapped

            // --- punctuation isolation: ASCII vs Unicode P vs Unicode symbol ---
            "a≤b x→y n≠m",                  // Sm math symbols: NOT punctuation → stay in word
            "«guillemets» \u{201c}smart\u{201d} \u{2018}quotes\u{2019}", // Pi/Pf → isolate
            "em—dash en–dash …ellipsis",     // Pd + Po
            "e.g., i.e. U.S.A. 3.14 1,000",  // dotted abbreviations / decimals
            "!!!??? ((())) [a]{b}",          // punctuation runs / adjacency
            "$100 €50 £20 ¥30 ₹40 ₿1",       // currency symbols (Sc)

            // --- emoji / symbols / combining sequences ---
            "hello 👋 world 🌍",
            "family 👨‍👩‍👧 ZWJ",             // ZWJ sequence (ZWJ is Cf → removed)
            "🇺🇸 🇯🇵 flags",                  // regional indicators
            "thumbs 👍🏽 skintone",           // emoji + skin-tone modifier

            // --- WordPiece internals: subwords, unk, max_input_chars ---
            "internationalization preprocessing tokenization playing",  // multi-## splits
            "ᚠᚢᚦ 𐐀𐐁 zzqxjk",               // runic/deseret/rare → unk path
            "🎉party mixed🔥emoji",           // unk char glued to word
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",     // 100 chars: at boundary
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",    // 101 chars: exceeds → unk
            "字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字字",  // long multibyte: byte-len ≫ char-len (exercises the byte shortcut)

            // --- literal special/added tokens ---
            "hello [MASK] world",
            "hello [mask] world",
            "[CLS] hi [SEP]",
            "[PAD][UNK][MASK]",
            "no special tokens here",
            "[MASK] [CLS] [SEP] [PAD] [UNK]",
            "foo[SEP]bar text[MASK]here",
            "[unk] lowercase specials",     // normalized-form collision check
        ]
    }

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../data/int4-dim-512-artifact/tokenizer.json"
    );

    #[test]
    #[ignore = "requires a local lattice-retrieval tokenizer under data/ (see README: slicer slice)"]
    fn fast_matches_crate() {
        let tok = Tokenizer::from_file(FIXTURE).expect("load fixture tokenizer");
        let ft = FastWordPiece::load(Path::new(FIXTURE)).expect("fixture is a WordPiece pipeline");

        for text in corpus() {
            let want: Vec<u32> = tok
                .encode_fast(text, false)
                .expect("crate encode")
                .get_ids()
                .to_vec();
            let got = ft.encode_ids(text);
            assert_eq!(got, want, "token-id mismatch for {text:?}");
        }
    }
}
