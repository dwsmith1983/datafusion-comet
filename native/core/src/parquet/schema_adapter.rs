// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::parquet::cast_column::CometCastColumnExpr;
use crate::parquet::parquet_support::{spark_parquet_convert, SparkParquetOptions};
use arrow::array::new_empty_array;
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ColumnarValue;
use datafusion::scalar::ScalarValue;
use datafusion_comet_common::SparkError;
use datafusion_comet_spark_expr::{Cast, SparkCastOptions};
use datafusion_physical_expr_adapter::{
    replace_columns_with_literals, DefaultPhysicalExprAdapterFactory, PhysicalExprAdapter,
    PhysicalExprAdapterFactory,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Factory for creating Spark-compatible physical expression adapters.
///
/// This factory creates adapters that rewrite expressions at planning time
/// to inject Spark-compatible casts where needed.
#[derive(Clone, Debug)]
pub struct SparkPhysicalExprAdapterFactory {
    /// Spark-specific parquet options for type conversions
    parquet_options: SparkParquetOptions,
    /// Default values for columns that may be missing from the physical schema.
    /// The key is the Column (containing name and index).
    default_values: Option<HashMap<Column, ScalarValue>>,
}

impl SparkPhysicalExprAdapterFactory {
    /// Create a new factory with the given options.
    pub fn new(
        parquet_options: SparkParquetOptions,
        default_values: Option<HashMap<Column, ScalarValue>>,
    ) -> Self {
        Self {
            parquet_options,
            default_values,
        }
    }
}

/// Read the Parquet field id stored under arrow-rs's `PARQUET_FIELD_ID_META_KEY`.
fn parse_field_id(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|v| v.parse::<i32>().ok())
}

fn schema_has_field_ids(schema: &SchemaRef) -> bool {
    schema.fields().iter().any(|f| parse_field_id(f).is_some())
}

// ---------------------------------------------------------------------------------------------
// JVM-shipped case tables: reproduce the PLANNING JVM's `String.toLowerCase(Locale.ROOT)`,
// which Spark's Parquet footer field matching is built on.
//
// The data arrives on `NativeScanCommon` (populated by `JvmCaseTables.scala` when
// case_sensitive = false), generated from the very JVM that plans the query, so the native
// matcher is correct BY CONSTRUCTION for whatever JDK runs Spark. The one contextual mapping
// (`Locale.ROOT` has exactly one: Greek capital sigma U+03A3) cannot be a per-codepoint table
// entry, so its condition is ported as an algorithm over a shipped per-codepoint
// classification -- see `JvmCaseTables::lowercase`.
// ---------------------------------------------------------------------------------------------

// Sigma-scan classes: the wire contract shared with `JvmCaseTables.scala`. The classes are
// the UAX#29-style word-break classes (ALetter, Numeric, MidLetter, MidNum, MidNumLet,
// Extend, Format) as the PLANNING JVM's legacy break iterator actually realizes them --
// probed per codepoint from `BreakIterator.isBoundary` on the JVM side -- plus classes for
// its pre-UAX#29 extensions (the danda and the supplementary-plane behaviors of its UTF-16
// DFA), with cased variants split out via the JDK's `isCased`. Any codepoint outside every
// shipped range -- and any class value this build does not know -- is a word boundary: the
// sigma context scan stops there, which is also the safe reading for class values added by a
// NEWER JVM-side generator.
const CLASS_ALETTER_CASED: u8 = 1;
const CLASS_ALETTER: u8 = 2;
const CLASS_NUMERIC: u8 = 3;
const CLASS_MID_LETTER: u8 = 4;
const CLASS_MID_NUM: u8 = 5;
const CLASS_MID_NUM_LET: u8 = 6;
/// Cased supplementary char: attaches to the preceding word and closes it (and forms a word
/// of its own at raw text start).
const CLASS_SUPP_CASED: u8 = 7;
/// U+0964/U+0965: word-terminal, chains only into digits.
const CLASS_DANDA: u8 = 8;
/// U+0345, the one cased combining mark: cased only when its run is attached to a word.
const CLASS_EXTEND_CASED: u8 = 9;
/// Cased digit-base (Nl Roman numerals): joins like CLASS_ALETTER_CASED when reached
/// directly, but bridges only mid-num punctuation, never mid-letter.
const CLASS_NUMERIC_CASED: u8 = 10;
/// Non-cased Mn/Me marks: riders that attach only to genuine letter/digit bases.
const CLASS_EXTEND: u8 = 11;
/// Word-forming non-cased supplementary letter: a genuine letter-base that closes the word
/// immediately after itself.
const CLASS_SUPP_LETTER: u8 = 12;
/// Cf format characters: fully transparent (WB4-style) -- deleted from the sequence before
/// the scans run, so a pure-format rider chain bridges mid punctuation ("AΣ-<ZWJ>b" is one
/// word exactly like "AΣ-b").
const CLASS_FORMAT: u8 = 13;
/// Supplementary chars that attach to the preceding word but never form one themselves
/// (supplementary combining marks, tag characters): a cased mark riding on one belongs to
/// the sigma's word only when the run hangs off a real base (`supp_mn_anchor`).
const CLASS_SUPP_MN: u8 = 14;
/// Word-forming supplementary digit: like CLASS_SUPP_LETTER except a riding cased mark
/// carries only across mid-num (digit-context) punctuation, never mid-letter.
const CLASS_SUPP_NUM: u8 = 15;
/// Not on the wire: the absence of a class.
const CLASS_BOUNDARY: u8 = 0;

const CAPITAL_SIGMA: char = '\u{03A3}';
const SMALL_SIGMA: char = '\u{03C3}';
const SMALL_FINAL_SIGMA: char = '\u{03C2}';

/// What a supplementary-mark run ultimately hangs off (see `supp_mn_anchor`).
#[derive(PartialEq, Eq, Clone, Copy)]
enum SuppMnAnchor {
    None,
    Letter,
    Digit,
}

/// The planning JVM's case data, parsed once per scan from `NativeScanCommon` and attached to
/// [`SparkParquetOptions`]. Two tables:
///
///   - `lower`: every codepoint the JVM lowercases non-identically, with its full (possibly
///     multi-char, e.g. U+0130 -> "i" + U+0307) replacement; codepoints absent here lowercase
///     to themselves;
///   - `class_ranges`: sorted, disjoint `(start, end, class)` codepoint ranges holding the
///     word-break classification the JVM probed from its own `BreakIterator`.
///
/// `lowercase` applies Java's algorithm over that data: per codepoint, U+03A3 takes its
/// contextual final/non-final form via the ported `isFinalCased` condition -- word-boundary
/// based (the JDK's legacy break-iterator word rules), NOT the Unicode-standard Final_Sigma
/// case-ignorable skip, so e.g. "A1Σ" lowers to "a1ς" -- and every other codepoint takes its
/// table replacement. `JvmCaseTables.mirrorLowercase` on the Scala side is the line-for-line
/// mirror of this function over the same generated data; the JVM-side parity suite proves the
/// pair equal to the running JDK's `String.toLowerCase(Locale.ROOT)` across the full codepoint
/// space (calibrated to zero mismatches on JDK 17, 21, and 25).
#[derive(Debug)]
pub struct JvmCaseTables {
    /// Non-identity lowercase mappings: codepoint -> full replacement string.
    lower: HashMap<char, String>,
    /// Sorted, disjoint (start, end, class) inclusive codepoint ranges for the sigma scan.
    class_ranges: Vec<(u32, u32, u8)>,
    /// Precomputed content hash so `SparkParquetOptions`'s derived `Hash` stays cheap.
    fingerprint: u64,
}

impl PartialEq for JvmCaseTables {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
            && self.class_ranges == other.class_ranges
            && self.lower == other.lower
    }
}

impl Eq for JvmCaseTables {}

impl Hash for JvmCaseTables {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Equal contents always produce the equal (deterministically computed) fingerprint,
        // so hashing only the fingerprint is consistent with `PartialEq`.
        state.write_u64(self.fingerprint);
    }
}

fn is_letter_base(cls: u8) -> bool {
    cls == CLASS_ALETTER_CASED
        || cls == CLASS_ALETTER
        || cls == CLASS_SUPP_CASED
        || cls == CLASS_SUPP_LETTER
}

fn is_digit_base(cls: u8) -> bool {
    cls == CLASS_NUMERIC || cls == CLASS_NUMERIC_CASED
}

impl JvmCaseTables {
    /// Parse the proto representation: `lower_cp`/`lower_repl` are index-aligned, and
    /// `class_ranges` holds (start, end, class) triples. Malformed input (length mismatch,
    /// trailing partial triple, out-of-range codepoints) is dropped entry-by-entry rather
    /// than rejected: every dropped entry degrades one codepoint to identity/boundary
    /// behavior instead of failing the scan.
    pub fn from_proto(lower_cp: &[u32], lower_repl: &[String], class_ranges: &[u32]) -> Self {
        let mut lower = HashMap::with_capacity(lower_cp.len());
        for (cp, repl) in lower_cp.iter().zip(lower_repl.iter()) {
            if let Some(c) = char::from_u32(*cp) {
                lower.insert(c, repl.clone());
            }
        }
        let ranges: Vec<(u32, u32, u8)> = class_ranges
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|t| t[0] <= t[1] && t[1] <= 0x10FFFF && u8::try_from(t[2]).is_ok())
            .map(|t| (t[0], t[1], t[2] as u8))
            .collect();

        let mut hasher = DefaultHasher::new();
        for (start, end, class) in &ranges {
            hasher.write_u32(*start);
            hasher.write_u32(*end);
            hasher.write_u8(*class);
        }
        let mut lower_sorted: Vec<(&char, &String)> = lower.iter().collect();
        lower_sorted.sort_by_key(|(c, _)| **c);
        for (c, repl) in lower_sorted {
            hasher.write_u32(*c as u32);
            hasher.write(repl.as_bytes());
        }

        Self {
            lower,
            class_ranges: ranges,
            fingerprint: hasher.finish(),
        }
    }

    /// Sigma-scan class of `c`; `CLASS_BOUNDARY` when no shipped range covers it.
    fn class_of(&self, c: char) -> u8 {
        let cp = c as u32;
        let mut lo = 0usize;
        let mut hi = self.class_ranges.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (start, end, class) = self.class_ranges[mid];
            if cp < start {
                hi = mid;
            } else if cp > end {
                lo = mid + 1;
            } else {
                return class;
            }
        }
        CLASS_BOUNDARY
    }

    /// First position at or beyond `start` (stepping by `step`, i.e. -1 backward / +1
    /// forward) whose class is not CLASS_EXTEND. Returns `None` if the scan runs off the
    /// array without finding one.
    fn skip_extends(&self, cps: &[char], start: isize, step: isize) -> Option<usize> {
        let mut k = start;
        while k >= 0 && (k as usize) < cps.len() {
            if self.class_of(cps[k as usize]) != CLASS_EXTEND {
                return Some(k as usize);
            }
            k += step;
        }
        None
    }

    /// As [`Self::skip_extends`], but also skips CLASS_EXTEND_CASED, reporting whether one
    /// was walked: a cased mark (U+0345) crossed while looking for a base is itself cased
    /// whenever the landing validates the run.
    fn skip_extends_tracking_cased(
        &self,
        cps: &[char],
        start: isize,
        step: isize,
    ) -> (Option<usize>, bool) {
        let mut k = start;
        let mut saw_cased = false;
        while k >= 0 && (k as usize) < cps.len() {
            let cls = self.class_of(cps[k as usize]);
            if cls != CLASS_EXTEND && cls != CLASS_EXTEND_CASED {
                return (Some(k as usize), saw_cased);
            }
            if cls == CLASS_EXTEND_CASED {
                saw_cased = true;
            }
            k += step;
        }
        (None, saw_cased)
    }

    /// What the supplementary-mark run at `k` (CLASS_SUPP_MN) ultimately hangs off, walking
    /// down through further marks and supplementary chars: a letter-flavored base, a
    /// digit-flavored base, or nothing word-forming. A cased mark riding the run belongs to
    /// the sigma's word only per this anchor.
    fn supp_mn_anchor(&self, cps: &[char], k: usize) -> SuppMnAnchor {
        let mut m = k as isize - 1;
        while m >= 0 {
            let cls = self.class_of(cps[m as usize]);
            if cls != CLASS_SUPP_MN && cls != CLASS_EXTEND && cls != CLASS_EXTEND_CASED {
                break;
            }
            m -= 1;
        }
        if m < 0 {
            return SuppMnAnchor::None;
        }
        match self.class_of(cps[m as usize]) {
            CLASS_ALETTER_CASED | CLASS_ALETTER | CLASS_SUPP_CASED | CLASS_SUPP_LETTER => {
                SuppMnAnchor::Letter
            }
            CLASS_NUMERIC | CLASS_NUMERIC_CASED | CLASS_SUPP_NUM => SuppMnAnchor::Digit,
            _ => SuppMnAnchor::None,
        }
    }

    /// Backward half of the ported `isFinalCased`: is there a cased letter before position
    /// `i` within the sigma's word? Runs over the FORMAT-FILTERED sequence; `leading_format`
    /// says whether format chars were filtered off the raw text start.
    fn scan_back_finds_cased(&self, cps: &[char], i: usize, leading_format: bool) -> bool {
        let mut last_letter = true; // the sigma itself is a letter
        let mut j = i as isize - 1;
        while j >= 0 {
            match self.class_of(cps[j as usize]) {
                CLASS_ALETTER_CASED | CLASS_NUMERIC_CASED => return true,
                CLASS_ALETTER => {
                    last_letter = true;
                    j -= 1;
                }
                CLASS_NUMERIC => {
                    last_letter = false;
                    j -= 1;
                }
                CLASS_EXTEND => {
                    // Non-cased marks attach only to a real base below them; anything else
                    // (mid punctuation, danda, boundary, text start) leaves the run
                    // unattached.
                    let Some(k) = self.skip_extends(cps, j, -1) else {
                        return false;
                    };
                    let b = self.class_of(cps[k]);
                    let is_continuer = b == CLASS_ALETTER_CASED
                        || b == CLASS_NUMERIC_CASED
                        || b == CLASS_NUMERIC
                        || b == CLASS_EXTEND_CASED
                        || b == CLASS_ALETTER
                        || b == CLASS_SUPP_CASED
                        || b == CLASS_SUPP_LETTER
                        || b == CLASS_SUPP_NUM;
                    if !is_continuer {
                        return false;
                    }
                    j = k as isize;
                }
                CLASS_SUPP_CASED => {
                    // Closes the preceding word, so the scan stops -- except at RAW text
                    // start (no filtered-out leading format chars), where the DFA keeps it
                    // joined to what follows.
                    return j == 0 && !leading_format;
                }
                CLASS_SUPP_LETTER | CLASS_SUPP_MN | CLASS_SUPP_NUM => {
                    // Attach/close and never themselves cased; nothing beyond is reachable.
                    return false;
                }
                CLASS_EXTEND_CASED => {
                    // Cased combining mark (U+0345): cased when its run hangs off a base --
                    // a BMP letter/digit, a word-forming supplementary char (which closes a
                    // word right below the mark, merging the mark into the sigma's
                    // segment), or an ANCHORED supplementary mark.
                    let (Some(k), _) = self.skip_extends_tracking_cased(cps, j - 1, -1) else {
                        return false;
                    };
                    let b = self.class_of(cps[k]);
                    if b == CLASS_ALETTER_CASED
                        || b == CLASS_NUMERIC
                        || b == CLASS_NUMERIC_CASED
                        || b == CLASS_ALETTER
                        || b == CLASS_SUPP_CASED
                        || b == CLASS_SUPP_LETTER
                        || b == CLASS_SUPP_NUM
                    {
                        return true;
                    }
                    if b == CLASS_SUPP_MN {
                        return self.supp_mn_anchor(cps, k) != SuppMnAnchor::None;
                    }
                    return false;
                }
                CLASS_DANDA => {
                    // Backward across a danda: the word part before it must end in letters
                    // (grammar: letters, optional danda, then number+word chains) -- or
                    // carry a riding cased mark on a word-forming base, or be a cased
                    // supplementary char at text start -- and the danda itself chains only
                    // into digits after it.
                    if last_letter {
                        return false;
                    }
                    let (Some(k), saw_cased_mark) =
                        self.skip_extends_tracking_cased(cps, j - 1, -1)
                    else {
                        return false;
                    };
                    let b = self.class_of(cps[k]);
                    if b == CLASS_ALETTER_CASED {
                        return true;
                    }
                    if b == CLASS_SUPP_CASED {
                        return saw_cased_mark || (k == 0 && !leading_format);
                    }
                    if b == CLASS_SUPP_LETTER {
                        return saw_cased_mark;
                    }
                    if b == CLASS_SUPP_MN {
                        return saw_cased_mark
                            && self.supp_mn_anchor(cps, k) == SuppMnAnchor::Letter;
                    }
                    if b != CLASS_ALETTER {
                        return false;
                    }
                    if saw_cased_mark {
                        return true;
                    }
                    last_letter = true;
                    j = k as isize;
                }
                cls @ (CLASS_MID_LETTER | CLASS_MID_NUM | CLASS_MID_NUM_LET) => {
                    // `<mid-letter><let>` / `<mid-num><digit>` require a genuine
                    // letter/digit base before the punctuation; scanning backward
                    // legitimately walks marks-then-base (marks trail their base). A cased
                    // mark walked over rides whatever the punctuation hangs off, including
                    // a context-matching anchored supplementary mark or supplementary
                    // digit.
                    let mw_ok = cls == CLASS_MID_LETTER || cls == CLASS_MID_NUM_LET;
                    let mn_ok = cls == CLASS_MID_NUM || cls == CLASS_MID_NUM_LET;
                    let (Some(real_pos), saw_cased_mark) =
                        self.skip_extends_tracking_cased(cps, j - 1, -1)
                    else {
                        return false;
                    };
                    let b = self.class_of(cps[real_pos]);
                    if last_letter
                        && mw_ok
                        && saw_cased_mark
                        && b == CLASS_SUPP_MN
                        && self.supp_mn_anchor(cps, real_pos) == SuppMnAnchor::Letter
                    {
                        return true;
                    }
                    if !last_letter
                        && mn_ok
                        && saw_cased_mark
                        && (b == CLASS_SUPP_NUM
                            || (b == CLASS_SUPP_MN
                                && self.supp_mn_anchor(cps, real_pos) == SuppMnAnchor::Digit))
                    {
                        return true;
                    }
                    let bridge_valid = (last_letter && mw_ok && is_letter_base(b))
                        || (!last_letter && mn_ok && is_digit_base(b));
                    if !bridge_valid {
                        return false;
                    }
                    if saw_cased_mark {
                        return true;
                    }
                    j = real_pos as isize;
                }
                _ => return false,
            }
        }
        false
    }

    /// Forward half of the ported `isFinalCased`: is there a cased letter after position `i`
    /// within the sigma's word? Runs over the FORMAT-FILTERED sequence.
    fn scan_fwd_finds_cased(&self, cps: &[char], i: usize) -> bool {
        let mut last_letter = true;
        let mut j = i + 1;
        while j < cps.len() {
            match self.class_of(cps[j]) {
                CLASS_ALETTER_CASED | CLASS_NUMERIC_CASED => return true,
                CLASS_ALETTER => {
                    last_letter = true;
                    j += 1;
                }
                CLASS_NUMERIC => {
                    last_letter = false;
                    j += 1;
                }
                CLASS_EXTEND => {
                    // A mark run trailing the anchor is properly attached in text order, so
                    // the run stays open past it, including onto mid punctuation on its far
                    // side.
                    let Some(k) = self.skip_extends(cps, j as isize, 1) else {
                        return false;
                    };
                    let b = self.class_of(cps[k]);
                    let is_continuer = b == CLASS_ALETTER_CASED
                        || b == CLASS_NUMERIC_CASED
                        || b == CLASS_NUMERIC
                        || b == CLASS_EXTEND_CASED
                        || b == CLASS_ALETTER
                        || b == CLASS_SUPP_CASED
                        || b == CLASS_SUPP_LETTER
                        || b == CLASS_SUPP_MN
                        || b == CLASS_SUPP_NUM
                        || b == CLASS_DANDA
                        || b == CLASS_MID_LETTER
                        || b == CLASS_MID_NUM
                        || b == CLASS_MID_NUM_LET;
                    if !is_continuer {
                        return false;
                    }
                    j = k;
                }
                CLASS_SUPP_CASED | CLASS_EXTEND_CASED => {
                    // Attaches to the current word, so the scan sees it (cased).
                    return true;
                }
                CLASS_SUPP_LETTER | CLASS_SUPP_MN | CLASS_SUPP_NUM => {
                    // Attach to the current word and close it; never themselves cased, and
                    // nothing beyond is reachable.
                    return false;
                }
                // The danda attaches only to a word part that ends in letters (reached
                // after digits the word is already closed) and continues only into a digit
                // -- unless that digit is itself cased (a Roman numeral), which resolves
                // the scan immediately.
                CLASS_DANDA if !last_letter => return false,
                CLASS_DANDA
                    if j + 1 < cps.len() && self.class_of(cps[j + 1]) == CLASS_NUMERIC_CASED =>
                {
                    return true;
                }
                CLASS_DANDA if j + 1 < cps.len() && self.class_of(cps[j + 1]) == CLASS_NUMERIC => {
                    last_letter = false;
                    j += 2;
                }
                cls @ (CLASS_MID_LETTER | CLASS_MID_NUM | CLASS_MID_NUM_LET) => {
                    // `<mid-letter><let>` / `<mid-num><digit>` require a genuine
                    // letter/digit base IMMEDIATELY after the punctuation -- unlike the
                    // backward scan, marks here are never skipped past: a mark directly
                    // after the punctuation is attached to the punctuation, not a base, so
                    // it blocks the bridge. (Format chars are already filtered out, which
                    // is what lets "AΣ-<ZWJ>b" bridge exactly like "AΣ-b".)
                    let mw_ok = cls == CLASS_MID_LETTER || cls == CLASS_MID_NUM_LET;
                    let mn_ok = cls == CLASS_MID_NUM || cls == CLASS_MID_NUM_LET;
                    if j + 1 >= cps.len() {
                        return false;
                    }
                    let b = self.class_of(cps[j + 1]);
                    if (last_letter && mw_ok && is_letter_base(b))
                        || (!last_letter && mn_ok && is_digit_base(b))
                    {
                        j += 1;
                    } else {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        false
    }

    /// Lowercase `s` the way the planning JVM's `String.toLowerCase(Locale.ROOT)` does.
    pub fn lowercase(&self, s: &str) -> String {
        let raw: Vec<char> = s.chars().collect();
        // Built lazily on the first sigma: the format-filtered sequence the scans run over
        // (WB4-style: the legacy break iterator's `<ignore>` class loops on every DFA
        // state), the raw->filtered index map, and whether format chars led the raw text.
        let mut filtered: Option<(Vec<char>, Vec<usize>, bool)> = None;
        let mut out = String::with_capacity(s.len());
        for (i, &c) in raw.iter().enumerate() {
            if c == CAPITAL_SIGMA {
                // The condition consults the ORIGINAL neighbors, exactly as the JDK scans
                // `src`, not the partially-lowered output. The final/non-final target chars
                // are Unicode-stable (pinned in `ConditionalSpecialCasing`'s entry table).
                let (f, idx, leading_format) = filtered.get_or_insert_with(|| {
                    let mut f = Vec::with_capacity(raw.len());
                    let mut idx = vec![0usize; raw.len()];
                    for (k, &rc) in raw.iter().enumerate() {
                        idx[k] = f.len();
                        if self.class_of(rc) != CLASS_FORMAT {
                            f.push(rc);
                        }
                    }
                    let leading_format = self.class_of(raw[0]) == CLASS_FORMAT;
                    (f, idx, leading_format)
                });
                let fi = idx[i];
                let is_final = self.scan_back_finds_cased(f, fi, *leading_format)
                    && !self.scan_fwd_finds_cased(f, fi);
                out.push(if is_final {
                    SMALL_FINAL_SIGMA
                } else {
                    SMALL_SIGMA
                });
            } else if let Some(repl) = self.lower.get(&c) {
                out.push_str(repl);
            } else {
                out.push(c);
            }
        }
        out
    }
}

/// Lowercase `s` for case-insensitive field matching. With tables (populated whenever
/// `case_sensitive = false`, shared by the core Parquet scan and the Delta contrib scan) this
/// reproduces the planning JVM's `String.toLowerCase(Locale.ROOT)` exactly.
///
/// Without tables, fall back to Rust's `str::to_lowercase`, which agrees with Java on all
/// simple mappings and differs only where the two Unicode snapshots or the sigma context
/// diverge. This is a real, live path, not just a defensive default: the Iceberg native scan
/// (`SparkPhysicalExprAdapterFactory::new(_, None)`) defaults `case_sensitive = false` and
/// always reaches this fallback for its schema name remap, alongside
/// `parquet_convert_struct_to_struct`'s general struct-cast matching and Rust-only unit tests
/// that construct `SparkParquetOptions` directly.
pub(crate) fn java_lowercase(s: &str, tables: Option<&JvmCaseTables>) -> String {
    match tables {
        Some(t) => t.lowercase(s),
        None => {
            log::debug!(
                "case-insensitive name matching without JVM case tables; \
                 falling back to str::to_lowercase for {s:?}"
            );
            s.to_lowercase()
        }
    }
}

/// Case-insensitive name equality mirroring Spark's actual Parquet footer field matching,
/// which groups/looks up physical column names by full-string `toLowerCase(Locale.ROOT)`
/// (see `ParquetReadSupport.clipParquetGroupFields`'s `caseInsensitiveParquetFieldMap`, and
/// `ParquetSchemaConverter.normalizeFieldName` for the vectorized `ParquetColumn` path -- both
/// key on `name.toLowerCase(Locale.ROOT)`, not a per-character `String.equalsIgnoreCase`
/// comparison). Two names are equal here iff their [`java_lowercase`] forms are equal.
pub(crate) fn names_equal_ignore_case_java(
    a: &str,
    b: &str,
    tables: Option<&JvmCaseTables>,
) -> bool {
    if a.is_ascii() && b.is_ascii() {
        return a.eq_ignore_ascii_case(b);
    }
    java_lowercase(a, tables) == java_lowercase(b, tables)
}

/// Remap physical schema field names to match logical schema field names. Mirrors Spark's
/// `clipParquetGroupFields`: prefer ID match for any logical field that carries a
/// `PARQUET:field_id`, fall back to case-insensitive name match otherwise.
///
/// The remap only changes top-level field NAMES so that `DefaultPhysicalExprAdapter`'s
/// exact-name lookup hits. Indices, types, nullability, and metadata stay as in the file.
/// Returns the rewritten schema and a `logical_name -> original_physical_name` map used
/// downstream to restore the original physical names before stream consumption.
fn remap_physical_schema(
    logical_schema: &SchemaRef,
    physical_schema: &SchemaRef,
    case_sensitive: bool,
    case_tables: Option<&JvmCaseTables>,
    use_field_id: bool,
    ignore_missing_field_id: bool,
) -> DataFusionResult<(SchemaRef, HashMap<String, String>)> {
    let should_match_by_id = use_field_id && schema_has_field_ids(logical_schema);

    if should_match_by_id && !ignore_missing_field_id && !schema_has_field_ids(physical_schema) {
        // Mirrors `ParquetReadSupport.inferSchema`'s eager check (Spark throws a runtime
        // error rather than silently returning null columns).
        return Err(DataFusionError::External(Box::new(
            SparkError::ParquetMissingFieldIds,
        )));
    }

    // Build id -> all matching physical field names. We need the full list so we can mirror
    // Spark's `_LEGACY_ERROR_TEMP_2094` "Found duplicate field(s)" error when an ID-bearing
    // logical field would resolve to more than one physical field.
    let mut id_to_phys_names: HashMap<i32, Vec<String>> = HashMap::new();
    if should_match_by_id {
        for pf in physical_schema.fields() {
            if let Some(id) = parse_field_id(pf) {
                id_to_phys_names
                    .entry(id)
                    .or_default()
                    .push(pf.name().clone());
            }
        }
        for lf in logical_schema.fields() {
            if let Some(id) = parse_field_id(lf) {
                if let Some(matches) = id_to_phys_names.get(&id) {
                    if matches.len() > 1 {
                        return Err(DataFusionError::External(Box::new(
                            SparkError::DuplicateFieldByFieldId {
                                required_id: id,
                                matched_fields: matches.join(", "),
                            },
                        )));
                    }
                }
            }
        }
    }

    // Pre-build id -> first matching logical field for the per-physical rename pass below.
    let id_to_logical: HashMap<i32, &FieldRef> = if should_match_by_id {
        let mut map = HashMap::new();
        for lf in logical_schema.fields() {
            if let Some(id) = parse_field_id(lf) {
                map.entry(id).or_insert(lf);
            }
        }
        map
    } else {
        HashMap::new()
    };

    // Names of ID-bearing logical fields. Spark's `matchIdField` resolves these strictly by
    // ID and never falls back to a name match, so a physical field that carries such a name
    // WITHOUT being the ID match (its ID is absent, different, or the logical ID matched a
    // different physical field) must be renamed to something the `DefaultPhysicalExprAdapter`
    // cannot name-match; otherwise the read would silently resolve the wrong column instead
    // of null-filling. Spark's `matchIdField` solves the same problem with
    // `generateFakeColumnName` (see `ParquetReadSupport.scala`).
    let id_logical_names: std::collections::HashSet<&str> = if should_match_by_id {
        logical_schema
            .fields()
            .iter()
            .filter(|lf| parse_field_id(lf).is_some())
            .map(|lf| lf.name().as_str())
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    // Fake names must never collide with a real column from either schema: a physical column
    // legitimately named like the fake pattern would otherwise become indistinguishable from
    // the shield's output and could steal an exact-name match. Spark gets the same guarantee
    // from the random UUID in `generateFakeColumnName`; here the counter is bumped past any
    // reserved name instead so the result stays deterministic.
    let reserved_names: std::collections::HashSet<&str> = logical_schema
        .fields()
        .iter()
        .chain(physical_schema.fields().iter())
        .map(|f| f.name().as_str())
        .collect();
    let mut fake_counter: usize = 0;
    let mut next_fake_name = move || loop {
        fake_counter += 1;
        let candidate = format!("__comet_unmatched_field_id_{}", fake_counter);
        if !reserved_names.contains(candidate.as_str()) {
            return candidate;
        }
    };

    let mut name_map: HashMap<String, String> = HashMap::new();
    let remapped_fields: Vec<FieldRef> = physical_schema
        .fields()
        .iter()
        .map(|field| {
            // ID match first when the logical schema is ID-bearing.
            if should_match_by_id {
                if let Some(phys_id) = parse_field_id(field) {
                    if let Some(logical_field) = id_to_logical.get(&phys_id) {
                        if logical_field.name() != field.name() {
                            name_map.insert(logical_field.name().clone(), field.name().clone());
                            return Arc::new(
                                Field::new(
                                    logical_field.name(),
                                    field.data_type().clone(),
                                    field.is_nullable(),
                                )
                                .with_metadata(field.metadata().clone()),
                            );
                        }
                        return Arc::clone(field);
                    }
                }
            }

            // Name match. Spark resolves every non-ID-bearing logical field by name --
            // `matchCaseSensitiveField` / `matchCaseInsensitiveField` in
            // `clipParquetGroupFields` -- even when field-ID matching is on, and that
            // resolution takes the physical field regardless of any ID-bearing logical field
            // with a similar name (`matchIdField` only ever fakes the REQUESTED field's name,
            // never the physical column's). Only ID-bearing logical fields skip the name
            // fallback when the schema is ID-bearing. Case-sensitive mode needs no rename
            // here (the downstream adapter's exact-name lookup already hits); the
            // case-insensitive lookup rewrites the physical name, and a successful match
            // claims the field before the shield below can hide it.
            if !case_sensitive {
                let logical_field = logical_schema.fields().iter().find(|lf| {
                    let lf_has_id = should_match_by_id && parse_field_id(lf).is_some();
                    !lf_has_id && names_equal_ignore_case_java(lf.name(), field.name(), case_tables)
                });
                if let Some(logical_field) = logical_field {
                    if logical_field.name() != field.name() {
                        name_map.insert(logical_field.name().clone(), field.name().clone());
                        return Arc::new(
                            Field::new(
                                logical_field.name(),
                                field.data_type().clone(),
                                field.is_nullable(),
                            )
                            .with_metadata(field.metadata().clone()),
                        );
                    }
                    return Arc::clone(field);
                }
            }

            // Shield: any remaining physical field whose name would hit an ID-bearing
            // logical field downstream gets a fake name (Spark's `generateFakeColumnName`
            // equivalent). ID-bearing logical fields resolve strictly by ID, so a name hit
            // on one would read the wrong column instead of null-filling it or leaving it to
            // its real ID match. The collision test mirrors the matcher that would otherwise
            // hit: exact names in case-sensitive mode (Spark's `matchCaseSensitiveField` /
            // the row converter's exact `catalystFieldIdxByName`), the planning JVM's
            // lowercase fold otherwise.
            if should_match_by_id {
                let collides = if case_sensitive {
                    id_logical_names.contains(field.name().as_str())
                } else {
                    id_logical_names
                        .iter()
                        .any(|name| names_equal_ignore_case_java(name, field.name(), case_tables))
                };
                if collides {
                    return Arc::new(
                        Field::new(
                            next_fake_name(),
                            field.data_type().clone(),
                            field.is_nullable(),
                        )
                        .with_metadata(field.metadata().clone()),
                    );
                }
            }

            Arc::clone(field)
        })
        .collect();

    Ok((Arc::new(Schema::new(remapped_fields)), name_map))
}

/// Format an Arrow `DataType` as Spark's catalog string (e.g. `Int64` -> `bigint`),
/// so SchemaColumnConvertNotSupportedException messages match Spark's vectorized reader.
fn spark_catalog_name(dt: &DataType) -> String {
    match dt {
        DataType::Boolean => "boolean".to_string(),
        DataType::Int8 => "tinyint".to_string(),
        DataType::Int16 => "smallint".to_string(),
        DataType::Int32 => "int".to_string(),
        DataType::Int64 => "bigint".to_string(),
        DataType::Float32 => "float".to_string(),
        DataType::Float64 => "double".to_string(),
        DataType::Utf8 | DataType::LargeUtf8 => "string".to_string(),
        DataType::Binary | DataType::LargeBinary => "binary".to_string(),
        DataType::Date32 => "date".to_string(),
        DataType::Timestamp(_, Some(_)) => "timestamp".to_string(),
        DataType::Timestamp(_, None) => "timestamp_ntz".to_string(),
        DataType::Decimal128(p, s) | DataType::Decimal256(p, s) => {
            format!("decimal({p},{s})")
        }
        _ => "unknown".to_string(),
    }
}

/// Format an Arrow `DataType` as the Parquet primitive type name
/// (e.g. `Int64` -> `INT64`, matching `PrimitiveTypeName.toString()` in parquet-mr).
fn parquet_primitive_name(dt: &DataType) -> &'static str {
    match dt {
        DataType::Boolean => "BOOLEAN",
        DataType::Int8 | DataType::Int16 | DataType::Int32 => "INT32",
        DataType::Int64 => "INT64",
        DataType::Float32 => "FLOAT",
        DataType::Float64 => "DOUBLE",
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => "BINARY",
        // Spark stores DATE as INT32 with a DATE logical-type annotation.
        DataType::Date32 => "INT32",
        // Spark stores TIMESTAMP as INT64 with a timestamp annotation, or as
        // INT96 (legacy nanos). arrow-rs surfaces both as `Timestamp`; without
        // the original physical name we report INT64, which matches the
        // common case.
        DataType::Timestamp(_, _) => "INT64",
        // Mirror Spark's `SparkToParquetSchemaConverter` decimal mapping:
        // precision 1-9 -> INT32, 10-18 -> INT64, 19+ -> FIXED_LEN_BYTE_ARRAY.
        DataType::Decimal128(p, _) | DataType::Decimal256(p, _) => {
            if *p <= 9 {
                "INT32"
            } else if *p <= 18 {
                "INT64"
            } else {
                "FIXED_LEN_BYTE_ARRAY"
            }
        }
        _ => "UNKNOWN",
    }
}

fn is_string_or_binary(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
    )
}

/// Build a Spark-shaped `SchemaColumnConvertNotSupportedException` carrier for a
/// rejected Parquet -> Spark conversion. The bracketed column wrapping mirrors
/// `Arrays.toString(descriptor.getPath())` in Spark's vectorized reader.
fn parquet_schema_convert_err(
    field_name: &str,
    physical_type: &DataType,
    target_type: &DataType,
) -> DataFusionError {
    DataFusionError::External(Box::new(SparkError::ParquetSchemaConvert {
        file_path: String::new(),
        column: format!("[{}]", field_name),
        physical_type: parquet_primitive_name(physical_type).to_string(),
        spark_type: spark_catalog_name(target_type),
    }))
}

/// Build a `RejectOnNonEmpty` expr wrapping `child`. The rejection fires only
/// when the input batch is non-empty (mirrors Spark's per-row-group check).
fn reject_on_non_empty_expr(
    child: Arc<dyn PhysicalExpr>,
    target_field: &FieldRef,
    field_name: &str,
    physical_type: &DataType,
    target_type: &DataType,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(RejectOnNonEmpty {
        child,
        target_field: Arc::clone(target_field),
        column: format!("[{}]", field_name),
        physical_type: parquet_primitive_name(physical_type).to_string(),
        spark_type: spark_catalog_name(target_type),
    })
}

/// Check if a specific column name has duplicate matches in the physical schema
/// (case-insensitive). Returns the error info if so.
fn check_column_duplicate(
    col_name: &str,
    physical_schema: &SchemaRef,
    case_tables: Option<&JvmCaseTables>,
) -> Option<(String, String)> {
    let matches: Vec<&str> = physical_schema
        .fields()
        .iter()
        .filter(|pf| names_equal_ignore_case_java(pf.name(), col_name, case_tables))
        .map(|pf| pf.name().as_str())
        .collect();
    if matches.len() > 1 {
        // Include brackets to match the format expected by ShimSparkErrorConverter
        Some((col_name.to_string(), format!("[{}]", matches.join(", "))))
    } else {
        None
    }
}

impl PhysicalExprAdapterFactory for SparkPhysicalExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> DataFusionResult<Arc<dyn PhysicalExprAdapter>> {
        // Remap physical schema field names to match logical names by Parquet field id
        // (when the logical schema carries IDs and `use_field_id` is set) and/or by
        // case-insensitive name match. The DefaultPhysicalExprAdapter uses exact name
        // matching, so without this remapping, columns whose file names differ from the
        // logical names won't match and will be filled with NULLs.
        //
        // We also keep a reverse map (logical name -> original physical name) so that
        // after the default adapter produces expressions, we can remap column names back
        // to the original physical names. This is necessary because downstream code
        // (reassign_expr_columns) looks up columns by name in the actual stream schema,
        // which uses the original physical file column names.
        let needs_remap = !self.parquet_options.case_sensitive
            || (self.parquet_options.use_field_id && schema_has_field_ids(&logical_file_schema));
        let (adapted_physical_schema, logical_to_physical_names, original_physical_schema) =
            if needs_remap {
                let (remapped, logical_to_physical) = remap_physical_schema(
                    &logical_file_schema,
                    &physical_file_schema,
                    self.parquet_options.case_sensitive,
                    self.parquet_options.jvm_case_tables.as_deref(),
                    self.parquet_options.use_field_id,
                    self.parquet_options.ignore_missing_field_id,
                )?;
                (
                    remapped,
                    if logical_to_physical.is_empty() {
                        None
                    } else {
                        Some(logical_to_physical)
                    },
                    // Keep original physical schema for per-column duplicate detection.
                    // Only meaningful in case-insensitive mode (matches existing behavior).
                    if !self.parquet_options.case_sensitive {
                        Some(Arc::clone(&physical_file_schema))
                    } else {
                        None
                    },
                )
            } else {
                (Arc::clone(&physical_file_schema), None, None)
            };

        let default_factory = DefaultPhysicalExprAdapterFactory;
        let default_adapter = default_factory.create(
            Arc::clone(&logical_file_schema),
            Arc::clone(&adapted_physical_schema),
        )?;

        Ok(Arc::new(SparkPhysicalExprAdapter {
            logical_file_schema,
            physical_file_schema: adapted_physical_schema,
            parquet_options: self.parquet_options.clone(),
            default_values: self.default_values.clone(),
            default_adapter,
            logical_to_physical_names,
            original_physical_schema,
        }))
    }
}

/// Spark-compatible physical expression adapter.
///
/// This adapter rewrites expressions at planning time to:
/// 1. Replace references to missing columns with default values or nulls
/// 2. Replace standard DataFusion cast expressions with Spark-compatible casts
/// 3. Handle case-insensitive column matching
#[derive(Debug)]
struct SparkPhysicalExprAdapter {
    /// The logical schema expected by the query
    logical_file_schema: SchemaRef,
    /// The physical schema of the actual file being read
    physical_file_schema: SchemaRef,
    /// Spark-specific options for type conversions
    parquet_options: SparkParquetOptions,
    /// Default values for missing columns (keyed by Column)
    default_values: Option<HashMap<Column, ScalarValue>>,
    /// The default DataFusion adapter to delegate standard handling to
    default_adapter: Arc<dyn PhysicalExprAdapter>,
    /// Mapping from logical column names to original physical column names,
    /// used for case-insensitive mode where names differ in casing.
    /// After the default adapter rewrites expressions using the remapped
    /// physical schema (with logical names), we need to restore the original
    /// physical names so that downstream reassign_expr_columns can find
    /// columns in the actual stream schema.
    logical_to_physical_names: Option<HashMap<String, String>>,
    /// The original (un-remapped) physical schema, kept for per-column duplicate
    /// detection in case-insensitive mode. Only set when `!case_sensitive`.
    original_physical_schema: Option<SchemaRef>,
}

impl PhysicalExprAdapter for SparkPhysicalExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        // In case-insensitive mode, check if any Column in this expression references
        // a field with multiple case-insensitive matches in the physical schema.
        // Only the columns actually referenced trigger the error (not the whole schema).
        if let Some(orig_physical) = &self.original_physical_schema {
            // ID-bearing logical fields resolve strictly by field ID (`matchIdField`) and
            // never reach Spark's case-insensitive name lookup, so its duplicate-name error
            // (`foundDuplicateFieldInCaseInsensitiveModeError`) never fires for them; exempt
            // them here the same way.
            let match_by_id = self.parquet_options.use_field_id
                && schema_has_field_ids(&self.logical_file_schema);
            // Walk the expression tree to find Column references
            let mut duplicate_err: Option<DataFusionError> = None;
            let _ = Arc::<dyn PhysicalExpr>::clone(&expr).transform(|e| {
                if let Some(col) = e.downcast_ref::<Column>() {
                    let id_routed = match_by_id
                        && self
                            .logical_file_schema
                            .field_with_name(col.name())
                            .ok()
                            .and_then(parse_field_id)
                            .is_some();
                    if id_routed {
                        return Ok(Transformed::no(e));
                    }
                    if let Some((req, matched)) = check_column_duplicate(
                        col.name(),
                        orig_physical,
                        self.parquet_options.jvm_case_tables.as_deref(),
                    ) {
                        duplicate_err = Some(DataFusionError::External(Box::new(
                            SparkError::DuplicateFieldCaseInsensitive {
                                required_field_name: req,
                                matched_fields: matched,
                            },
                        )));
                    }
                }
                Ok(Transformed::no(e))
            });
            if let Some(err) = duplicate_err {
                return Err(err);
            }
        }

        // First let the default adapter handle column remapping, missing columns,
        // and simple scalar type casts. Then replace DataFusion's CastColumnExpr
        // with Spark-compatible equivalents.
        //
        // The default adapter may fail for complex nested type casts (List, Map).
        // In that case, fall back to wrapping everything ourselves.
        let expr = self.replace_missing_with_defaults(expr)?;
        let expr = match self.default_adapter.rewrite(Arc::clone(&expr)) {
            Ok(rewritten) => {
                // Replace references to missing columns with default values
                // Replace DataFusion's CastColumnExpr with either:
                // - CometCastColumnExpr (for Struct/List/Map, uses spark_parquet_convert)
                // - Spark Cast (for simple scalar types)
                rewritten
                    .transform(|e| self.replace_with_spark_cast(e))
                    .data()?
            }
            Err(e) => {
                // Default adapter failed (likely complex nested type cast).
                // Handle all type mismatches ourselves using spark_parquet_convert.
                log::debug!("Default schema adapter error: {}", e);
                self.wrap_all_type_mismatches(expr)?
            }
        };

        // For case-insensitive mode: remap column names from logical back to
        // original physical names. The default adapter was given a remapped
        // physical schema (with logical names) so it could find columns. But
        // downstream code (reassign_expr_columns) looks up columns by name in
        // the actual parquet stream schema, which uses the original physical names.
        let expr = if let Some(name_map) = &self.logical_to_physical_names {
            expr.transform(|e| {
                if let Some(col) = e.downcast_ref::<Column>() {
                    if let Some(physical_name) = name_map.get(col.name()) {
                        return Ok(Transformed::yes(Arc::new(Column::new(
                            physical_name,
                            col.index(),
                        ))));
                    }
                }
                Ok(Transformed::no(e))
            })
            .data()?
        } else {
            expr
        };

        Ok(expr)
    }
}

impl SparkPhysicalExprAdapter {
    /// Wrap ALL Column expressions that have type mismatches with CometCastColumnExpr.
    /// This is the fallback path when the default adapter fails (e.g., for complex
    /// nested type casts like List<Struct> or Map). Uses `spark_parquet_convert`
    /// under the hood for the actual type conversion.
    fn wrap_all_type_mismatches(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        expr.transform(|e| {
            if let Some(column) = e.downcast_ref::<Column>() {
                let col_name = column.name();

                // Resolve fields by name because this is the fallback path
                // that runs on the original expression when the default
                // adapter fails. The original expression was built against
                // the required (pruned) schema, so column indices refer to
                // that schema — not the logical or physical file schemas.
                // DataFusion's DefaultPhysicalExprAdapter::resolve_physical_column
                // also resolves by name for the same reason.
                let logical_field = if self.parquet_options.case_sensitive {
                    self.logical_file_schema
                        .fields()
                        .iter()
                        .find(|f| f.name() == col_name)
                } else {
                    self.logical_file_schema.fields().iter().find(|f| {
                        names_equal_ignore_case_java(
                            f.name(),
                            col_name,
                            self.parquet_options.jvm_case_tables.as_deref(),
                        )
                    })
                };
                let physical_field = if self.parquet_options.case_sensitive {
                    self.physical_file_schema
                        .fields()
                        .iter()
                        .find(|f| f.name() == col_name)
                } else {
                    self.physical_file_schema.fields().iter().find(|f| {
                        names_equal_ignore_case_java(
                            f.name(),
                            col_name,
                            self.parquet_options.jvm_case_tables.as_deref(),
                        )
                    })
                };

                // Remap the column index to the physical file schema so
                // downstream evaluation reads the correct column from the
                // parquet batch.
                let physical_index = if self.parquet_options.case_sensitive {
                    self.physical_file_schema.index_of(col_name).ok()
                } else {
                    self.physical_file_schema.fields().iter().position(|f| {
                        names_equal_ignore_case_java(
                            f.name(),
                            col_name,
                            self.parquet_options.jvm_case_tables.as_deref(),
                        )
                    })
                };

                if let (Some(logical_field), Some(physical_field), Some(phys_idx)) =
                    (logical_field, physical_field, physical_index)
                {
                    let remapped: Arc<dyn PhysicalExpr> = if column.index() != phys_idx {
                        Arc::new(Column::new(col_name, phys_idx))
                    } else {
                        Arc::clone(&e)
                    };

                    if logical_field.data_type() != physical_field.data_type() {
                        // Mirror the same string/binary -> non-string/binary rejection in
                        // `replace_with_spark_cast`; this branch is reached when the default
                        // adapter rejected the cast and we'd otherwise build a CometCastColumnExpr
                        // that can't actually convert (e.g. BINARY -> DECIMAL with no
                        // `DecimalLogicalTypeAnnotation`). See #4088 and #4351.
                        let physical_type = physical_field.data_type();
                        let target_type = logical_field.data_type();
                        if is_string_or_binary(physical_type) && !is_string_or_binary(target_type) {
                            return Err(parquet_schema_convert_err(
                                physical_field.name(),
                                physical_type,
                                target_type,
                            ));
                        }

                        let cast_expr: Arc<dyn PhysicalExpr> = Arc::new(
                            CometCastColumnExpr::new(
                                remapped,
                                Arc::clone(physical_field),
                                Arc::clone(logical_field),
                                None,
                            )
                            .with_parquet_options(self.parquet_options.clone()),
                        );
                        return Ok(Transformed::yes(cast_expr));
                    } else if column.index() != phys_idx {
                        return Ok(Transformed::yes(remapped));
                    }
                }
            }
            Ok(Transformed::no(e))
        })
        .data()
    }

    /// Replace CastExpr (DataFusion's cast) with Spark's Cast expression.
    fn replace_with_spark_cast(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> DataFusionResult<Transformed<Arc<dyn PhysicalExpr>>> {
        // Check for CastExpr and replace with spark_expr::Cast
        if let Some(cast) = expr.downcast_ref::<datafusion::physical_expr::expressions::CastExpr>()
        {
            let child = Arc::clone(cast.expr());
            let target_type = cast.target_field().data_type();

            // Derive input field from the child Column expression and the physical schema.
            // DF main removed CastColumnExpr in favor of CastExpr, so we recover the input
            // field from the child Column rather than calling cast.input_field().
            let input_field = if let Some(col) = child.downcast_ref::<Column>() {
                Arc::new(self.physical_file_schema.field(col.index()).clone())
            } else {
                // Fallback: synthesize a field from the target field name and child data type
                let child_type = cast.expr().data_type(&self.physical_file_schema)?;
                Arc::new(Field::new(cast.target_field().name(), child_type, true))
            };
            let physical_type = input_field.data_type();

            // Identity cast: DataFusion's default adapter inserts a CastExpr
            // whenever the logical and physical Arrow Fields differ in any
            // attribute (data type, nullability, or metadata), so with identical
            // data types but mismatched nullability or metadata, we receive a
            // no-op cast. Unwrapping is safe because Spark `Cast` with equal
            // source and target types is value-level identity (it does not
            // null-strip or enforce non-null), and field nullability/metadata is
            // informational rather than computational. Leaving the wrapper in
            // place blocks DataFusion's pruning-predicate analyzer from
            // recognizing the column reference, defeating row-group / page-index
            // stats pruning.
            if physical_type == target_type {
                return Ok(Transformed::yes(child));
            }

            // Reject reading a string/binary Parquet column as anything else. Spark's
            // `ParquetVectorUpdaterFactory.getUpdater` BINARY case allows StringType /
            // BinaryType, or DecimalType only when the column carries a
            // `DecimalLogicalTypeAnnotation` (which arrow-rs surfaces as `Decimal128`,
            // not `Binary`). Without this guard, runtime cast paths silently return
            // nulls, parse strings, or surface as a generic Arrow type-mismatch error.
            // See #4088 and #4351.
            if is_string_or_binary(physical_type) && !is_string_or_binary(target_type) {
                return Err(parquet_schema_convert_err(
                    input_field.name(),
                    physical_type,
                    target_type,
                ));
            }

            // Reject reading a primitive numeric Parquet column as StringType /
            // BinaryType. Spark has no `int -> string` etc. updater. Defer to
            // runtime via `RejectOnNonEmpty` so empty Parquet files (SPARK-26709)
            // pass and the JVM shim translates to
            // `SchemaColumnConvertNotSupportedException`.
            let physical_is_primitive_numeric = matches!(
                physical_type,
                DataType::Boolean
                    | DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Float32
                    | DataType::Float64
            );
            if physical_is_primitive_numeric && is_string_or_binary(target_type) {
                let rejection = reject_on_non_empty_expr(
                    child,
                    cast.target_field(),
                    input_field.name(),
                    physical_type,
                    target_type,
                );
                return Ok(Transformed::yes(rejection));
            }

            // Decimal-to-decimal narrowing. Spark's `isDecimalTypeMatched` (the
            // `DecimalLogicalTypeAnnotation` branch) allows the read only when
            //   `dst_scale >= src_scale` AND
            //   `dst_precision - dst_scale >= src_precision - src_scale`.
            // Either failure means silently dropping fractional digits or losing
            // integer-side magnitude. See #4089 and #4343.
            if let (DataType::Decimal128(src_p, src_s), DataType::Decimal128(dst_p, dst_s)) =
                (physical_type, target_type)
            {
                let src_int_precision = i32::from(*src_p) - i32::from(*src_s);
                let dst_int_precision = i32::from(*dst_p) - i32::from(*dst_s);
                if dst_s < src_s || dst_int_precision < src_int_precision {
                    return Err(parquet_schema_convert_err(
                        input_field.name(),
                        physical_type,
                        target_type,
                    ));
                }
            }

            // Integer-to-decimal narrowing. Spark's `canReadAsDecimal` requires
            // `precision - scale >= 10` for an INT32 source and `>= 20` for INT64.
            // Unconditional in all Spark versions, so reject at plan time. See #4344.
            let int_decimal_min_int_precision = match physical_type {
                DataType::Int8 | DataType::Int16 | DataType::Int32 => Some(10i32),
                DataType::Int64 => Some(20i32),
                _ => None,
            };
            if let Some(min_int_precision) = int_decimal_min_int_precision {
                let dst_precision_scale = match target_type {
                    DataType::Decimal128(p, s) | DataType::Decimal256(p, s) => Some((*p, *s)),
                    _ => None,
                };
                if let Some((dst_p, dst_s)) = dst_precision_scale {
                    let dst_int_precision = i32::from(dst_p) - i32::from(dst_s);
                    if dst_int_precision < min_int_precision {
                        return Err(parquet_schema_convert_err(
                            input_field.name(),
                            physical_type,
                            target_type,
                        ));
                    }
                }
            }

            // Type promotion (widening). When `allow_type_promotion` is false,
            // reject the three widenings (INT32→INT64, FLOAT→DOUBLE, INT32→DOUBLE)
            // that Spark 3.x's vectorized reader rejects. The flag tracks Comet's
            // per-Spark-version constant in ShimCometConf. Deferred to runtime so
            // empty files (SPARK-26709) pass.
            if !self.parquet_options.allow_type_promotion {
                let is_disallowed_promotion = matches!(
                    (physical_type, target_type),
                    (DataType::Int32, DataType::Int64)
                        | (DataType::Float32, DataType::Float64)
                        | (DataType::Int32, DataType::Float64)
                );
                if is_disallowed_promotion {
                    let rejection = reject_on_non_empty_expr(
                        Arc::clone(&child),
                        cast.target_field(),
                        input_field.name(),
                        physical_type,
                        target_type,
                    );
                    return Ok(Transformed::yes(rejection));
                }
            }

            // Reject primitive Parquet conversions Spark's vectorized reader rejects
            // on every supported version (no matching branch in
            // `ParquetVectorUpdaterFactory.getUpdater`):
            //
            //   - `INT64 -> Int*` truncates lower bits.
            //   - `INT64 -> Float*` and `INT32 -> Float32` lose precision.
            //   - `Float* -> Int*` and `Float64 -> Float32` truncate / overflow.
            //   - `INT32 -> Timestamp` / `INT64 -> Date32` / `INT64 -> Timestamp`:
            //     date/timestamp-annotated columns surface as Date32 / Timestamp,
            //     so reaching this branch means the column was un-annotated.
            //   - `Date32 -> Timestamp(LTZ)`: Spark only allows Date -> TimestampNTZ.
            //   - `Timestamp -> Date32`: no Timestamp updater branches into Date.
            //
            // Deferred to runtime (SPARK-26709). See #4297.
            let is_spark_rejected_conversion = matches!(
                (physical_type, target_type),
                // Long -> narrower int.
                (
                    DataType::Int64,
                    DataType::Int8 | DataType::Int16 | DataType::Int32,
                )
                // Long -> floating point.
                | (DataType::Int64, DataType::Float32 | DataType::Float64)
                // Long -> date / timestamp (raw INT64; annotated columns surface as Date32/Timestamp).
                | (DataType::Int64, DataType::Date32)
                | (DataType::Int64, DataType::Timestamp(_, _))
                // Int -> float (DoubleType is allowed via IntegerToDoubleUpdater; FloatType is not).
                | (
                    DataType::Int8 | DataType::Int16 | DataType::Int32,
                    DataType::Float32,
                )
                // Int -> timestamp (raw INT32; DATE-annotated columns surface as Date32).
                | (
                    DataType::Int8 | DataType::Int16 | DataType::Int32,
                    DataType::Timestamp(_, _),
                )
                // Float -> int / Double -> int (no integer branches under FLOAT/DOUBLE).
                | (
                    DataType::Float32 | DataType::Float64,
                    DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64,
                )
                // Double -> float (narrowing).
                | (DataType::Float64, DataType::Float32)
                // Date -> Timestamp(LTZ). Spark allows Date -> TimestampNTZ only.
                | (DataType::Date32, DataType::Timestamp(_, Some(_)))
                // Timestamp -> Date.
                | (DataType::Timestamp(_, _), DataType::Date32)
            );
            if is_spark_rejected_conversion {
                let rejection = reject_on_non_empty_expr(
                    child,
                    cast.target_field(),
                    input_field.name(),
                    physical_type,
                    target_type,
                );
                return Ok(Transformed::yes(rejection));
            }

            // Spark 3.x refuses to read a Parquet TimestampLTZ column as
            // TimestampNTZ (SPARK-36182); Spark 4.0 (SPARK-47447) lifted that.
            // The flag tracks Comet's per-Spark-version constant in
            // ShimCometConf. Deferred to runtime so empty files (SPARK-26709)
            // still pass. See #4219.
            //
            // This catches all LTZ physical encodings: TIMESTAMP_MICROS /
            // TIMESTAMP_MILLIS arrive as `Timestamp(_, Some(_))` directly, and
            // INT96 arrives as `Timestamp(_, Some("UTC"))` because `coerce_int96_tz`
            // attaches the UTC timezone (see `get_options`) instead of letting
            // `coerce_int96` strip it to a timezone-free `Timestamp(_, None)`.
            if !self.parquet_options.allow_timestamp_ltz_to_ntz
                && matches!(
                    (physical_type, target_type),
                    (
                        DataType::Timestamp(_, Some(_)),
                        DataType::Timestamp(_, None)
                    )
                )
            {
                let rejection = reject_on_non_empty_expr(
                    Arc::clone(&child),
                    cast.target_field(),
                    input_field.name(),
                    physical_type,
                    target_type,
                );
                return Ok(Transformed::yes(rejection));
            }

            // Scalar/complex mismatch (e.g. TIMESTAMP read as ARRAY<TIMESTAMP>):
            // Spark's vectorized reader rejects with
            // SchemaColumnConvertNotSupportedException (SPARK-45604). Same-shape
            // complex pairs and timestamp→timestamp / timestamp→int64 fall through
            // to CometCastColumnExpr below.
            let is_complex = |t: &DataType| {
                matches!(
                    t,
                    DataType::Struct(_) | DataType::List(_) | DataType::Map(_, _)
                )
            };
            if is_complex(physical_type) != is_complex(target_type) {
                return Err(parquet_schema_convert_err(
                    input_field.name(),
                    physical_type,
                    target_type,
                ));
            }

            // Same-shape complex casts, timestamp tz relabel (e.g. Timestamp(us, None)
            // -> Timestamp(us, Some("UTC")) for INT96 reads), and Timestamp -> Int64
            // (Spark's `nanosAsLong`) need spark_parquet_convert: it handles nested
            // field selection, metadata-only tz changes, and raw-value reinterpretation
            // that Spark's Cast would otherwise convert incorrectly.
            if matches!(
                (physical_type, target_type),
                (DataType::Struct(_), DataType::Struct(_))
                    | (DataType::List(_), DataType::List(_))
                    | (DataType::Map(_, _), DataType::Map(_, _))
                    | (DataType::Timestamp(_, _), DataType::Timestamp(_, _))
                    | (DataType::Timestamp(_, _), DataType::Int64)
            ) {
                let comet_cast: Arc<dyn PhysicalExpr> = Arc::new(
                    CometCastColumnExpr::new(
                        child,
                        input_field,
                        Arc::clone(cast.target_field()),
                        None,
                    )
                    .with_parquet_options(self.parquet_options.clone()),
                );
                return Ok(Transformed::yes(comet_cast));
            }

            // For simple scalar type casts, use Spark-compatible Cast expression
            let mut cast_options = SparkCastOptions::new(
                self.parquet_options.eval_mode,
                &self.parquet_options.timezone,
                self.parquet_options.allow_incompat,
            );
            cast_options.allow_cast_unsigned_ints = self.parquet_options.allow_cast_unsigned_ints;
            cast_options.is_adapting_schema = true;

            let spark_cast = Arc::new(Cast::new(
                child,
                target_type.clone(),
                cast_options,
                None,
                None,
            ));

            return Ok(Transformed::yes(spark_cast as Arc<dyn PhysicalExpr>));
        }

        Ok(Transformed::no(expr))
    }

    /// Replace references to missing columns with default values.
    fn replace_missing_with_defaults(
        &self,
        expr: Arc<dyn PhysicalExpr>,
    ) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        let Some(defaults) = &self.default_values else {
            return Ok(expr);
        };

        if defaults.is_empty() {
            return Ok(expr);
        }

        // Build owned (column_name, default_value) pairs for columns missing from the physical file.
        // For each default: filter to only columns absent from physical schema, then type-cast
        // the value to match the logical schema's field type if they differ (using Spark cast semantics).
        let missing_column_defaults: Vec<(String, ScalarValue)> = defaults
            .iter()
            .filter_map(|(col, val)| {
                let col_name = col.name();

                // Only include defaults for columns missing from the physical file schema
                let is_missing = if self.parquet_options.case_sensitive {
                    self.physical_file_schema.field_with_name(col_name).is_err()
                } else {
                    !self.physical_file_schema.fields().iter().any(|f| {
                        names_equal_ignore_case_java(
                            f.name(),
                            col_name,
                            self.parquet_options.jvm_case_tables.as_deref(),
                        )
                    })
                };

                if !is_missing {
                    return None;
                }

                // Cast value to logical schema type if needed (only if types differ)
                let value = self
                    .logical_file_schema
                    .field_with_name(col_name)
                    .ok()
                    .filter(|field| val.data_type() != *field.data_type())
                    .and_then(|field| {
                        spark_parquet_convert(
                            ColumnarValue::Scalar(val.clone()),
                            field.data_type(),
                            &self.parquet_options,
                        )
                        .ok()
                        .and_then(|cv| match cv {
                            ColumnarValue::Scalar(s) => Some(s),
                            _ => None,
                        })
                    })
                    .unwrap_or_else(|| val.clone());

                Some((col_name.to_string(), value))
            })
            .collect();

        let name_based: HashMap<&str, &ScalarValue> = missing_column_defaults
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();

        if name_based.is_empty() {
            return Ok(expr);
        }

        replace_columns_with_literals(expr, &name_based)
    }
}

/// Defers a Parquet type-promotion rejection to runtime: returns an empty array
/// when the input batch has no rows, and raises `ParquetSchemaConvert` otherwise.
///
/// Mirrors Spark's vectorized reader, which only invokes
/// `ParquetVectorUpdaterFactory.getUpdater` while decoding a row group. A
/// Parquet file with no row groups (e.g. one written from an empty DataFrame)
/// never triggers the per-row-group check, so a partition mixing such a file
/// with another whose schema would otherwise fail the type-promotion check
/// (SPARK-26709) is still readable.
#[derive(Debug, Eq)]
struct RejectOnNonEmpty {
    child: Arc<dyn PhysicalExpr>,
    target_field: FieldRef,
    column: String,
    physical_type: String,
    spark_type: String,
}

impl PartialEq for RejectOnNonEmpty {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
            && self.target_field.eq(&other.target_field)
            && self.column == other.column
            && self.physical_type == other.physical_type
            && self.spark_type == other.spark_type
    }
}

impl Hash for RejectOnNonEmpty {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        self.target_field.hash(state);
        self.column.hash(state);
        self.physical_type.hash(state);
        self.spark_type.hash(state);
    }
}

impl Display for RejectOnNonEmpty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "REJECT_PARQUET_TYPE_PROMOTION({} AS {})",
            self.column, self.spark_type
        )
    }
}

impl PhysicalExpr for RejectOnNonEmpty {
    fn data_type(&self, _input_schema: &Schema) -> DataFusionResult<DataType> {
        Ok(self.target_field.data_type().clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> DataFusionResult<bool> {
        Ok(self.target_field.is_nullable())
    }

    fn evaluate(&self, batch: &RecordBatch) -> DataFusionResult<ColumnarValue> {
        if batch.num_rows() == 0 {
            return Ok(ColumnarValue::Array(new_empty_array(
                self.target_field.data_type(),
            )));
        }
        Err(DataFusionError::External(Box::new(
            SparkError::ParquetSchemaConvert {
                file_path: String::new(),
                column: self.column.clone(),
                physical_type: self.physical_type.clone(),
                spark_type: self.spark_type.clone(),
            },
        )))
    }

    fn return_field(&self, _input_schema: &Schema) -> DataFusionResult<FieldRef> {
        Ok(Arc::clone(&self.target_field))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DataFusionResult<Arc<dyn PhysicalExpr>> {
        assert_eq!(children.len(), 1);
        Ok(Arc::new(RejectOnNonEmpty {
            child: children.pop().expect("child"),
            target_field: Arc::clone(&self.target_field),
            column: self.column.clone(),
            physical_type: self.physical_type.clone(),
            spark_type: self.spark_type.clone(),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

#[cfg(test)]
mod test {
    use crate::parquet::parquet_support::SparkParquetOptions;
    use crate::parquet::schema_adapter::{
        java_lowercase, names_equal_ignore_case_java, remap_physical_schema, JvmCaseTables,
        SparkPhysicalExprAdapterFactory, CLASS_ALETTER_CASED, CLASS_DANDA, CLASS_EXTEND,
        CLASS_EXTEND_CASED, CLASS_FORMAT, CLASS_MID_LETTER, CLASS_MID_NUM, CLASS_MID_NUM_LET,
        CLASS_NUMERIC, CLASS_NUMERIC_CASED, CLASS_SUPP_CASED, CLASS_SUPP_LETTER, CLASS_SUPP_MN,
        CLASS_SUPP_NUM,
    };
    use arrow::array::UInt32Array;
    use arrow::array::{
        Array, BinaryArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int32Array,
        Int64Array, StringArray, TimestampMicrosecondArray,
    };
    use arrow::datatypes::SchemaRef;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::DataFusionError;
    use datafusion::datasource::listing::PartitionedFile;
    use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource};
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::execution::object_store::ObjectStoreUrl;
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion_comet_spark_expr::test_common::file_util::get_temp_filename;
    use datafusion_comet_spark_expr::EvalMode;
    use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
    use futures::StreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    /// Reading a non-BINARY Parquet column as `StringType` must raise the same
    /// `_LEGACY_ERROR_TEMP_2063`-shaped error as Spark's vectorized reader
    /// (`ParquetVectorUpdaterFactory.getUpdater` has no INT32 -> string updater).
    #[tokio::test]
    async fn parquet_int_read_as_string_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int32, false),
            values,
            DataType::Utf8,
        )
        .await?;
        assert!(
            msg.contains("Column: [[a]]")
                && msg.contains("Expected: string")
                && msg.contains("Found: INT32"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// Companion: BINARY (string physical) read as IntegerType must raise the
    /// same Spark-compatible error.
    #[tokio::test]
    async fn parquet_string_read_as_int_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(StringArray::from(vec!["bcd", "efg"])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Utf8, false),
            values,
            DataType::Int32,
        )
        .await?;
        assert!(
            msg.contains("Column: [[a]]")
                && msg.contains("Expected: int")
                && msg.contains("Found: BINARY"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// Reading a plain BINARY Parquet column (no `DecimalLogicalTypeAnnotation`)
    /// as `DecimalType` must raise a Spark-compatible `ParquetSchemaConvert`
    /// error. Spark's `canReadAsDecimal` / `canReadAsBinaryDecimal` both require
    /// the column to carry a `DecimalLogicalTypeAnnotation`. See #4351.
    #[tokio::test]
    async fn parquet_binary_read_as_decimal_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(BinaryArray::from_vec(vec![b"1.2", b"3.4"])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Binary, false),
            values,
            DataType::Decimal128(37, 1),
        )
        .await?;
        assert!(
            msg.contains("Column: [[a]]")
                && msg.contains("Expected: decimal(37,1)")
                && msg.contains("Found: BINARY"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// INT32 -> Decimal where `precision - scale < 10` (the minimum that can
    /// represent the full INT32 range). See #4344.
    #[tokio::test]
    async fn parquet_int32_read_as_narrow_decimal_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int32, false),
            values,
            DataType::Decimal128(9, 0),
        )
        .await?;
        assert!(
            msg.contains("Column: [[a]]")
                && msg.contains("Expected: decimal")
                && msg.contains("Found: INT32"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// INT32 -> Decimal where `precision - scale >= 10` is allowed.
    #[tokio::test]
    async fn parquet_int32_read_as_wide_decimal_succeeds() -> Result<(), DataFusionError> {
        let file_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let values = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(Arc::clone(&file_schema), vec![values])?;
        let required_schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Decimal128(10, 0),
            false,
        )]));
        let _ = roundtrip(&batch, required_schema).await?;
        Ok(())
    }

    /// INT64 -> Decimal where `precision - scale < 20`. See #4344.
    #[tokio::test]
    async fn parquet_int64_read_as_narrow_decimal_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(Int64Array::from(vec![1i64, 2, 3])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int64, false),
            values,
            DataType::Decimal128(19, 0),
        )
        .await?;
        assert!(
            msg.contains("Column: [[a]]")
                && msg.contains("Expected: decimal")
                && msg.contains("Found: INT64"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// Non-zero scale that pushes `precision - scale` below the integer minimum
    /// (INT32 -> Decimal(10, 1) leaves int-precision 9).
    #[tokio::test]
    async fn parquet_int32_read_as_decimal_with_scale_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int32, false),
            values,
            DataType::Decimal128(10, 1),
        )
        .await?;
        assert!(
            msg.contains("Column: [[a]]")
                && msg.contains("Expected: decimal")
                && msg.contains("Found: INT32"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// Helper to build a tiny decimal Parquet batch for the decimal-to-decimal tests.
    fn decimal_batch(precision: u8, scale: i8) -> Result<RecordBatch, DataFusionError> {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Decimal128(precision, scale),
            false,
        )]));
        let values = Arc::new(
            Decimal128Array::from(vec![123i128, 456])
                .with_precision_and_scale(precision, scale)
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
        ) as Arc<dyn arrow::array::Array>;
        Ok(RecordBatch::try_new(file_schema, vec![values])?)
    }

    /// Reading Decimal(P, S) as Decimal(P', S) where P' < P (precision-only
    /// narrowing, equal scale) must raise the Spark-compatible error. Spark's
    /// `isDecimalTypeMatched` rejects this because `precisionIncrease < 0`
    /// while `scaleIncrease == 0`. See #4343.
    #[tokio::test]
    async fn parquet_decimal_precision_narrowing_errors() -> Result<(), DataFusionError> {
        let batch = decimal_batch(10, 2)?;
        let required_schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Decimal128(5, 2),
            false,
        )]));

        let err = roundtrip(&batch, required_schema)
            .await
            .expect_err("expected ParquetSchemaConvert for Decimal(10, 2) -> Decimal(5, 2)");
        let msg = err.to_string();
        assert!(
            msg.contains("Column: [[a]]") && msg.contains("Expected: decimal(5,2)"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// Reading Decimal(P, S) as Decimal(P', S') where the integer-precision
    /// `P - S` shrinks must raise the Spark-compatible error. Example:
    /// Decimal(10, 4) (int-precision 6) -> Decimal(5, 2) (int-precision 3).
    /// See #4343.
    #[tokio::test]
    async fn parquet_decimal_int_precision_narrowing_errors() -> Result<(), DataFusionError> {
        let batch = decimal_batch(10, 4)?;
        let required_schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Decimal128(5, 2),
            false,
        )]));

        let err = roundtrip(&batch, required_schema)
            .await
            .expect_err("expected ParquetSchemaConvert for Decimal(10, 4) -> Decimal(5, 2)");
        let msg = err.to_string();
        assert!(msg.contains("Column: [[a]]"), "unexpected error: {msg}");
        Ok(())
    }

    /// Reading Decimal(P, S) as Decimal(P, S') where S' > S but `P - S` did
    /// not grow means the cast would shift integer digits into the fractional
    /// part and lose the most-significant digit. Example: Decimal(5, 2) ->
    /// Decimal(5, 3): scaleIncrease=1, precisionIncrease=0. See #4343.
    #[tokio::test]
    async fn parquet_decimal_scale_widening_without_precision_errors() -> Result<(), DataFusionError>
    {
        let batch = decimal_batch(5, 2)?;
        let required_schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Decimal128(5, 3),
            false,
        )]));

        let err = roundtrip(&batch, required_schema)
            .await
            .expect_err("expected ParquetSchemaConvert for Decimal(5, 2) -> Decimal(5, 3)");
        let msg = err.to_string();
        assert!(msg.contains("Column: [[a]]"), "unexpected error: {msg}");
        Ok(())
    }

    /// Sanity check: widening both precision and scale by the same amount is
    /// allowed (the cast is lossless). Decimal(5, 2) -> Decimal(7, 4) gives
    /// scaleIncrease=2, precisionIncrease=2, so `precisionIncrease >= scaleIncrease`.
    #[tokio::test]
    async fn parquet_decimal_widening_succeeds() -> Result<(), DataFusionError> {
        let batch = decimal_batch(5, 2)?;
        let required_schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Decimal128(7, 4),
            false,
        )]));

        let _ = roundtrip(&batch, required_schema).await?;
        Ok(())
    }

    /// Helper for the #4297 rejection tests: write a 1-row batch and assert
    /// that reading it under `read_type` raises `ParquetSchemaConvert`.
    async fn assert_rejected_conversion(
        file_field: Field,
        values: Arc<dyn arrow::array::Array>,
        read_type: DataType,
    ) -> Result<String, DataFusionError> {
        let file_schema = Arc::new(Schema::new(vec![file_field]));
        let batch = RecordBatch::try_new(Arc::clone(&file_schema), vec![values])?;
        let read_field_name = file_schema.field(0).name();
        let required_schema = Arc::new(Schema::new(vec![Field::new(
            read_field_name,
            read_type,
            false,
        )]));
        let err = roundtrip(&batch, required_schema)
            .await
            .expect_err("expected ParquetSchemaConvert");
        Ok(err.to_string())
    }

    /// `INT64 -> INT32` truncates to the lower 32 bits in DataFusion's cast.
    /// Spark's vectorized reader rejects this. See #4297.
    #[tokio::test]
    async fn parquet_long_read_as_int_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(Int64Array::from(vec![1i64, 1 << 33])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int64, false),
            values,
            DataType::Int32,
        )
        .await?;
        assert!(
            msg.contains("Found: INT64") && msg.contains("Expected: int"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `INT64 -> Float64` loses precision for large values; Spark rejects.
    #[tokio::test]
    async fn parquet_long_read_as_double_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(Int64Array::from(vec![1i64, (1i64 << 54) + 1]))
            as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int64, false),
            values,
            DataType::Float64,
        )
        .await?;
        assert!(
            msg.contains("Found: INT64") && msg.contains("Expected: double"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Float64 -> Float32` overflows / loses precision; Spark rejects.
    #[tokio::test]
    async fn parquet_double_read_as_float_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(Float64Array::from(vec![1.5_f64, 1e40])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Float64, false),
            values,
            DataType::Float32,
        )
        .await?;
        assert!(
            msg.contains("Found: DOUBLE") && msg.contains("Expected: float"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Float32 -> Int64` truncates the fractional part; Spark rejects.
    #[tokio::test]
    async fn parquet_float_read_as_long_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(Float32Array::from(vec![1.5_f32, 2.5])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Float32, false),
            values,
            DataType::Int64,
        )
        .await?;
        assert!(
            msg.contains("Found: FLOAT") && msg.contains("Expected: bigint"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Float64 -> Int64` similarly.
    #[tokio::test]
    async fn parquet_double_read_as_long_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(Float64Array::from(vec![1.5_f64, 2.5])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Float64, false),
            values,
            DataType::Int64,
        )
        .await?;
        assert!(
            msg.contains("Found: DOUBLE") && msg.contains("Expected: bigint"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Int32 -> Float32` loses precision for values past `2^24`. Spark
    /// allows `Int32 -> Float64` but rejects `Int32 -> Float32`.
    #[tokio::test]
    async fn parquet_int_read_as_float_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(Int32Array::from(vec![1, (1 << 25) + 1])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int32, false),
            values,
            DataType::Float32,
        )
        .await?;
        assert!(
            msg.contains("Found: INT32") && msg.contains("Expected: float"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Int32 -> Timestamp(_, None)`: raw INT32 reinterpreted as epoch seconds
    /// produces dates near the Unix epoch. Only DATE-annotated INT32 columns
    /// (which surface as `Date32`) are allowed to read as `TimestampNTZ`.
    #[tokio::test]
    async fn parquet_int_read_as_timestamp_ntz_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int32, false),
            values,
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
        )
        .await?;
        assert!(
            msg.contains("Found: INT32") && msg.contains("Expected: timestamp"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Int64 -> Date32` similarly: raw INT64 (no DATE annotation, otherwise
    /// the file would surface as `Date32`).
    #[tokio::test]
    async fn parquet_long_read_as_date_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(Int64Array::from(vec![1i64, 2])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Int64, false),
            values,
            DataType::Date32,
        )
        .await?;
        assert!(
            msg.contains("Found: INT64") && msg.contains("Expected: date"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Date32 -> Timestamp(_, Some(_))` (LTZ). Spark's vectorized reader
    /// allows `Date -> TimestampNTZ` but not `Date -> Timestamp(LTZ)`.
    #[tokio::test]
    async fn parquet_date_read_as_ltz_timestamp_errors() -> Result<(), DataFusionError> {
        let values =
            Arc::new(Date32Array::from(vec![18262, 18263])) as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new("a", DataType::Date32, false),
            values,
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into())),
        )
        .await?;
        assert!(
            msg.contains("Found: INT32") && msg.contains("Expected: timestamp"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// `Timestamp(_, _) -> Date32`: no Timestamp updater branches into
    /// `DateType`, so Spark rejects.
    #[tokio::test]
    async fn parquet_timestamp_read_as_date_errors() -> Result<(), DataFusionError> {
        let values = Arc::new(TimestampMicrosecondArray::from(vec![0i64, 1_000_000]))
            as Arc<dyn arrow::array::Array>;
        let msg = assert_rejected_conversion(
            Field::new(
                "a",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
                false,
            ),
            values,
            DataType::Date32,
        )
        .await?;
        assert!(msg.contains("Expected: date"), "unexpected error: {msg}");
        Ok(())
    }

    /// SPARK-26709: an empty Parquet file with a column that would otherwise fail
    /// the type-promotion check (INT32 read as INT64 when allow_type_promotion is
    /// false) must still be readable. Spark's vectorized reader only enforces the
    /// check per row group, so a file with no row groups passes silently. The
    /// adapter's plan-time rejection must not fire for the empty-file case.
    #[tokio::test]
    async fn parquet_empty_file_disallowed_widening() -> Result<(), DataFusionError> {
        let file_schema = Arc::new(Schema::new(vec![Field::new("col", DataType::Int32, false)]));
        let filename = get_temp_filename();
        let filename = filename.as_path().as_os_str().to_str().unwrap().to_string();
        let file = File::create(&filename)?;
        let writer = ArrowWriter::try_new(file, Arc::clone(&file_schema), None)?;
        writer.close()?;

        let required_schema =
            Arc::new(Schema::new(vec![Field::new("col", DataType::Int64, false)]));

        let mut spark_parquet_options = SparkParquetOptions::new(EvalMode::Legacy, "UTC", false);
        spark_parquet_options.allow_type_promotion = false;

        let expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory> = Arc::new(
            SparkPhysicalExprAdapterFactory::new(spark_parquet_options, None),
        );

        let object_store_url = ObjectStoreUrl::local_filesystem();
        let parquet_source = ParquetSource::new(required_schema);
        let files = FileGroup::new(vec![PartitionedFile::from_path(filename)?]);
        let file_scan_config =
            FileScanConfigBuilder::new(object_store_url, Arc::new(parquet_source))
                .with_file_groups(vec![files])
                .with_expr_adapter(Some(expr_adapter_factory))
                .build();

        let parquet_exec = DataSourceExec::new(Arc::new(file_scan_config));
        let mut stream = parquet_exec.execute(0, Arc::new(TaskContext::default()))?;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            assert_eq!(batch.num_rows(), 0);
        }
        Ok(())
    }

    /// Companion to `parquet_empty_file_disallowed_widening`: a file with rows
    /// must still raise `ParquetSchemaConvert` when the same widening is
    /// rejected. Verifies the runtime check fires on non-empty input,
    /// matching Spark's per-row-group behavior.
    #[tokio::test]
    async fn parquet_non_empty_file_disallowed_widening_errors() -> Result<(), DataFusionError> {
        let file_schema = Arc::new(Schema::new(vec![Field::new("col", DataType::Int32, false)]));
        let values = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(Arc::clone(&file_schema), vec![values])?;

        let filename = get_temp_filename();
        let filename = filename.as_path().as_os_str().to_str().unwrap().to_string();
        let file = File::create(&filename)?;
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&file_schema), None)?;
        writer.write(&batch)?;
        writer.close()?;

        let required_schema =
            Arc::new(Schema::new(vec![Field::new("col", DataType::Int64, false)]));

        let mut spark_parquet_options = SparkParquetOptions::new(EvalMode::Legacy, "UTC", false);
        spark_parquet_options.allow_type_promotion = false;

        let expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory> = Arc::new(
            SparkPhysicalExprAdapterFactory::new(spark_parquet_options, None),
        );

        let object_store_url = ObjectStoreUrl::local_filesystem();
        let parquet_source = ParquetSource::new(required_schema);
        let files = FileGroup::new(vec![PartitionedFile::from_path(filename)?]);
        let file_scan_config =
            FileScanConfigBuilder::new(object_store_url, Arc::new(parquet_source))
                .with_file_groups(vec![files])
                .with_expr_adapter(Some(expr_adapter_factory))
                .build();

        let parquet_exec = DataSourceExec::new(Arc::new(file_scan_config));
        let mut stream = parquet_exec.execute(0, Arc::new(TaskContext::default()))?;
        let first = stream.next().await.unwrap();
        let err = first.expect_err("expected ParquetSchemaConvert error on non-empty file");
        let msg = err.to_string();
        // The JVM shim sees the inner "[col]" via the JSON `column` field, matching
        // Spark's `Arrays.toString(descriptor.getPath())` format. The Rust display
        // wraps with another `[...]` from the error template.
        assert!(
            msg.contains("Column: [[col]]")
                && msg.contains("Expected: bigint")
                && msg.contains("Found: INT32"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn parquet_roundtrip_unsigned_int() -> Result<(), DataFusionError> {
        let file_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt32, false)]));

        let ids = Arc::new(UInt32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(Arc::clone(&file_schema), vec![ids])?;

        let required_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let _ = roundtrip(&batch, required_schema).await?;

        Ok(())
    }

    /// Create a Parquet file containing a single batch and then read the batch back using
    /// the specified required_schema. This will cause the PhysicalExprAdapter code to be used.
    async fn roundtrip(
        batch: &RecordBatch,
        required_schema: SchemaRef,
    ) -> Result<RecordBatch, DataFusionError> {
        let filename = get_temp_filename();
        let filename = filename.as_path().as_os_str().to_str().unwrap().to_string();
        let file = File::create(&filename)?;
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&batch.schema()), None)?;
        writer.write(batch)?;
        writer.close()?;

        let object_store_url = ObjectStoreUrl::local_filesystem();

        let mut spark_parquet_options = SparkParquetOptions::new(EvalMode::Legacy, "UTC", false);
        spark_parquet_options.allow_cast_unsigned_ints = true;

        // Create expression adapter factory for Spark-compatible schema adaptation
        let expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory> = Arc::new(
            SparkPhysicalExprAdapterFactory::new(spark_parquet_options, None),
        );

        let parquet_source = ParquetSource::new(required_schema);

        let files = FileGroup::new(vec![PartitionedFile::from_path(filename.to_string())?]);
        let file_scan_config =
            FileScanConfigBuilder::new(object_store_url, Arc::new(parquet_source))
                .with_file_groups(vec![files])
                .with_expr_adapter(Some(expr_adapter_factory))
                .build();

        let parquet_exec = DataSourceExec::new(Arc::new(file_scan_config));

        let mut stream = parquet_exec.execute(0, Arc::new(TaskContext::default()))?;
        stream.next().await.unwrap()
    }

    #[tokio::test]
    async fn parquet_duplicate_fields_case_insensitive() {
        // Parquet file has columns "A", "B", "b" - reading "b" in case-insensitive mode
        // should fail with duplicate field error matching Spark's _LEGACY_ERROR_TEMP_2093
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("A", DataType::Int32, false),
            Field::new("B", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));

        let col_a = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow::array::Array>;
        let col_b1 = Arc::new(Int32Array::from(vec![4, 5, 6])) as Arc<dyn arrow::array::Array>;
        let col_b2 = Arc::new(Int32Array::from(vec![7, 8, 9])) as Arc<dyn arrow::array::Array>;
        let batch =
            RecordBatch::try_new(Arc::clone(&file_schema), vec![col_a, col_b1, col_b2]).unwrap();

        let filename = get_temp_filename();
        let filename = filename.as_path().as_os_str().to_str().unwrap().to_string();
        let file = File::create(&filename).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&batch.schema()), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Read with case-insensitive mode, requesting column "b" which matches both "B" and "b"
        let required_schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Int32, false)]));

        let mut spark_parquet_options = SparkParquetOptions::new(EvalMode::Legacy, "UTC", false);
        spark_parquet_options.case_sensitive = false;

        let expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory> = Arc::new(
            SparkPhysicalExprAdapterFactory::new(spark_parquet_options, None),
        );

        let object_store_url = ObjectStoreUrl::local_filesystem();
        let parquet_source = ParquetSource::new(required_schema);
        let files = FileGroup::new(vec![
            PartitionedFile::from_path(filename.to_string()).unwrap()
        ]);
        let file_scan_config =
            FileScanConfigBuilder::new(object_store_url, Arc::new(parquet_source))
                .with_file_groups(vec![files])
                .with_expr_adapter(Some(expr_adapter_factory))
                .build();

        let parquet_exec = DataSourceExec::new(Arc::new(file_scan_config));
        let mut stream = parquet_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let result = stream.next().await.unwrap();

        // Should fail with duplicate field error
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Found duplicate field"),
            "Expected duplicate field error, got: {err_msg}"
        );
    }

    /// Build a nullable Int64 field carrying a Parquet field ID.
    fn field_with_id(name: &str, id: i32) -> Field {
        Field::new(name, DataType::Int64, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    /// Write a Parquet file from `file_schema`/`columns`, then scan it with
    /// `required_schema` through the Spark expression adapter and return the first batch.
    async fn scan_with_adapter(
        file_schema: SchemaRef,
        columns: Vec<Arc<dyn arrow::array::Array>>,
        required_schema: SchemaRef,
        spark_parquet_options: SparkParquetOptions,
    ) -> Result<RecordBatch, DataFusionError> {
        let batch = RecordBatch::try_new(Arc::clone(&file_schema), columns).unwrap();

        let filename = get_temp_filename();
        let filename = filename.as_path().as_os_str().to_str().unwrap().to_string();
        let file = File::create(&filename).unwrap();
        let mut writer = ArrowWriter::try_new(file, file_schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory> = Arc::new(
            SparkPhysicalExprAdapterFactory::new(spark_parquet_options, None),
        );

        let object_store_url = ObjectStoreUrl::local_filesystem();
        let parquet_source = ParquetSource::new(required_schema);
        let files = FileGroup::new(vec![PartitionedFile::from_path(filename).unwrap()]);
        let file_scan_config =
            FileScanConfigBuilder::new(object_store_url, Arc::new(parquet_source))
                .with_file_groups(vec![files])
                .with_expr_adapter(Some(expr_adapter_factory))
                .build();

        let parquet_exec = DataSourceExec::new(Arc::new(file_scan_config));
        let mut stream = parquet_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        stream.next().await.unwrap()
    }

    /// File: one column `κ` (U+03BA) with field ID 2 holding 7. Required: `Κ` (U+039A,
    /// field ID 1) and ID-less `κ`; case-sensitive, field-ID reading on. Spark routes `Κ`
    /// through `matchIdField` (no ID 1 in the file -> null-filled behind a faked REQUESTED
    /// name) and resolves `κ` by exact name through `matchCaseSensitiveField`, reading the
    /// real column: the result is (NULL, 7), never (NULL, NULL).
    #[tokio::test]
    async fn parquet_field_id_miss_null_fills_but_exact_name_sibling_still_reads() {
        let file_schema = Arc::new(Schema::new(vec![field_with_id("\u{3BA}", 2)]));
        let col = Arc::new(Int64Array::from(vec![7])) as Arc<dyn arrow::array::Array>;
        let required_schema = Arc::new(Schema::new(vec![
            field_with_id("\u{39A}", 1),
            Field::new("\u{3BA}", DataType::Int64, true),
        ]));

        let mut opts = SparkParquetOptions::new(EvalMode::Legacy, "UTC", false);
        opts.case_sensitive = true;
        opts.use_field_id = true;

        let batch = scan_with_adapter(file_schema, vec![col], required_schema, opts)
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let capital = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(capital.is_null(0));
        let small = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(!small.is_null(0));
        assert_eq!(small.value(0), 7);
    }

    /// Case-insensitive variant of the Kappa scenario. Spark's `matchCaseInsensitiveField`
    /// resolves the ID-less requested `κ` through the `toLowerCase(Locale.ROOT)`-keyed map
    /// of the file's fields, which holds the physical `κ`; the unmatched-ID requested `Κ`
    /// is null-filled and never blocks that lookup. Same (NULL, 7) result as the
    /// case-sensitive read.
    #[tokio::test]
    async fn parquet_field_id_miss_case_insensitive_sibling_still_reads() {
        let file_schema = Arc::new(Schema::new(vec![field_with_id("\u{3BA}", 2)]));
        let col = Arc::new(Int64Array::from(vec![7])) as Arc<dyn arrow::array::Array>;
        let required_schema = Arc::new(Schema::new(vec![
            field_with_id("\u{39A}", 1),
            Field::new("\u{3BA}", DataType::Int64, true),
        ]));

        let mut opts = SparkParquetOptions::new(EvalMode::Legacy, "UTC", false);
        opts.case_sensitive = false;
        opts.use_field_id = true;

        let batch = scan_with_adapter(file_schema, vec![col], required_schema, opts)
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let capital = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(capital.is_null(0));
        let small = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(!small.is_null(0));
        assert_eq!(small.value(0), 7);
    }

    /// File: a stray ID-less column literally named `A` = [10, 20] FIRST, then `a` with
    /// field ID 1 = [1, 2]. Required: `A` with field ID 1, case-insensitive, field-ID
    /// reading on. Spark's `matchIdField` resolves requested `A` to physical `a` by ID; the
    /// stray `A` is never requested, and no case-insensitive duplicate error fires because
    /// ID-routed requested fields never enter the name lookup. Expect [1, 2] -- neither the
    /// stray column's data nor a spurious duplicate-field error.
    #[tokio::test]
    async fn parquet_field_id_match_beats_stray_column_with_requested_name() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("A", DataType::Int64, true),
            field_with_id("a", 1),
        ]));
        let stray = Arc::new(Int64Array::from(vec![10, 20])) as Arc<dyn arrow::array::Array>;
        let matched = Arc::new(Int64Array::from(vec![1, 2])) as Arc<dyn arrow::array::Array>;
        let required_schema = Arc::new(Schema::new(vec![field_with_id("A", 1)]));

        let mut opts = SparkParquetOptions::new(EvalMode::Legacy, "UTC", false);
        opts.case_sensitive = false;
        opts.use_field_id = true;

        let batch = scan_with_adapter(file_schema, vec![stray, matched], required_schema, opts)
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 2);
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.value(0), 1);
        assert_eq!(values.value(1), 2);
    }

    // -- JvmCaseTables mechanics: the native algorithm over a miniature, JDK-17-sourced
    // table. These pin the NATIVE half; the runtime consumes tables generated by the live
    // planning JVM (JvmCaseTables.scala), proven equal to `String.toLowerCase(Locale.ROOT)`
    // by the JVM-side parity suite (JvmLowercaseParitySuite). --

    /// Builds a miniature `JvmCaseTables` from literal data recorded from a real JDK 17
    /// (zulu 17.0.18) `String.toLowerCase(Locale.ROOT)` run, covering exactly the codepoints
    /// these tests touch. Expected strings in the tests below are JDK-17-sourced the same way.
    fn jdk17_test_tables() -> JvmCaseTables {
        let mut lower_cp: Vec<u32> = Vec::new();
        let mut lower_repl: Vec<String> = Vec::new();
        for cp in 0x41u32..=0x5A {
            lower_cp.push(cp);
            lower_repl.push(char::from_u32(cp + 0x20).unwrap().to_string());
        }
        let mut pair = |cp: u32, repl: &str| {
            lower_cp.push(cp);
            lower_repl.push(repl.to_string());
        };
        pair(0xC9, "\u{E9}"); // É -> é
        pair(0x130, "i\u{307}"); // İ -> "i" + COMBINING DOT ABOVE (multi-char expansion)
        pair(0x391, "\u{3B1}"); // Α -> α
        pair(0x392, "\u{3B2}"); // Β -> β
        pair(0x394, "\u{3B4}"); // Δ -> δ
        pair(0x395, "\u{3B5}"); // Ε -> ε
        pair(0x39F, "\u{3BF}"); // Ο -> ο
        pair(0x3A3, "\u{3C3}"); // Σ -> σ (the isolated form; contexts handled by the scan)
        pair(0x3A5, "\u{3C5}"); // Υ -> υ
        pair(0x212A, "k"); // KELVIN SIGN -> ASCII k
        pair(0x10400, "\u{10428}"); // DESERET CAPITAL LONG I -> small
        pair(0x2160, "\u{2170}"); // ROMAN NUMERAL ONE -> small roman numeral one

        #[rustfmt::skip]
        let class_ranges: Vec<u32> = vec![
            0x22, 0x22, CLASS_MID_NUM_LET as u32,   // '"'
            0x27, 0x27, CLASS_MID_NUM_LET as u32,   // '\''
            0x2C, 0x2C, CLASS_MID_NUM as u32,    // ','
            0x2D, 0x2D, CLASS_MID_LETTER as u32,   // '-'
            0x2E, 0x2E, CLASS_MID_NUM_LET as u32,   // '.'
            0x30, 0x39, CLASS_NUMERIC as u32,      // 0-9
            0x41, 0x5A, CLASS_ALETTER_CASED as u32,      // A-Z
            0x5F, 0x5F, CLASS_MID_LETTER as u32,   // '_'
            0x61, 0x7A, CLASS_ALETTER_CASED as u32,      // a-z
            0xC9, 0xC9, CLASS_ALETTER_CASED as u32,      // É
            0xDF, 0xDF, CLASS_ALETTER_CASED as u32,      // ß
            0xE9, 0xE9, CLASS_ALETTER_CASED as u32,      // é
            0x130, 0x131, CLASS_ALETTER_CASED as u32,    // İ, ı
            0x301, 0x301, CLASS_EXTEND as u32, // COMBINING ACUTE (Mn, non-cased)
            0x307, 0x307, CLASS_EXTEND as u32, // COMBINING DOT ABOVE (Mn, non-cased)
            0x345, 0x345, CLASS_EXTEND_CASED as u32, // COMBINING GREEK YPOGEGRAMMENI
            0x391, 0x3A9, CLASS_ALETTER_CASED as u32,    // Greek capitals
            0x3B1, 0x3C9, CLASS_ALETTER_CASED as u32,    // Greek smalls (incl. σ, ς)
            0x64E, 0x64E, CLASS_EXTEND as u32, // ARABIC FATHA (Mn, non-cased)
            0x964, 0x964, CLASS_DANDA as u32,    // DEVANAGARI DANDA
            0x200D, 0x200D, CLASS_FORMAT as u32, // ZERO WIDTH JOINER (Cf)
            0x212A, 0x212A, CLASS_ALETTER_CASED as u32,  // KELVIN SIGN
            0x2160, 0x2170, CLASS_NUMERIC_CASED as u32, // Roman numeral one, upper and lower
            0x10400, 0x10400, CLASS_SUPP_CASED as u32, // DESERET CAPITAL LONG I
            0x11374, 0x11374, CLASS_SUPP_MN as u32,  // COMBINING GRANTHA LETTER A (Mn, supp)
            0x1D7D3, 0x1D7D3, CLASS_SUPP_NUM as u32, // MATHEMATICAL BOLD DIGIT FIVE (Nd, supp)
            0x20000, 0x20000, CLASS_SUPP_LETTER as u32, // CJK UNIFIED IDEOGRAPH-20000 (Lo)
        ];
        JvmCaseTables::from_proto(&lower_cp, &lower_repl, &class_ranges)
    }

    #[test]
    fn kelvin_sign_matches_ascii_k_and_capital_k() {
        // U+212A KELVIN SIGN lowercases (Locale.ROOT) to ASCII 'k'. Rust's
        // `eq_ignore_ascii_case` never looks past the ASCII range, so it would (wrongly) say
        // these differ.
        let t = jdk17_test_tables();
        assert!(names_equal_ignore_case_java("\u{212A}", "k", Some(&t)));
        assert!(names_equal_ignore_case_java("\u{212A}", "K", Some(&t)));
        assert!(names_equal_ignore_case_java("k", "\u{212A}", Some(&t)));
    }

    #[test]
    fn e_acute_case_pair_matches() {
        // 'É' (U+00C9) / 'é' (U+00E9): a Latin-1 case pair outside the ASCII range.
        let t = jdk17_test_tables();
        assert!(names_equal_ignore_case_java("\u{c9}", "\u{e9}", Some(&t)));
        assert!(names_equal_ignore_case_java(
            "R\u{c9}SUM\u{c9}",
            "r\u{e9}sum\u{e9}",
            Some(&t)
        ));
    }

    #[test]
    fn greek_capital_sigma_matches_regular_lowercase_sigma() {
        // A standalone capital sigma has no preceding cased letter, so the contextual "final
        // sigma" rule does not apply and the unconditional mapping to σ (U+03C3) is used.
        let t = jdk17_test_tables();
        assert!(names_equal_ignore_case_java("\u{3a3}", "\u{3c3}", Some(&t)));
    }

    #[test]
    fn greek_final_sigma_does_not_match_regular_lowercase_sigma() {
        // 'ς' (U+03C2, final lowercase sigma) is already lowercase and maps to itself; it does
        // NOT unify with 'σ' (U+03C3, regular lowercase sigma).
        let t = jdk17_test_tables();
        assert!(!names_equal_ignore_case_java(
            "\u{3c2}",
            "\u{3c3}",
            Some(&t)
        ));
    }

    #[test]
    fn sharp_s_does_not_match_ss() {
        // 'ß' (U+00DF) is already lowercase and maps to itself under `toLowerCase`, so it
        // never unifies with "ss" -- unlike `str::to_uppercase`'s full folding to "SS".
        let t = jdk17_test_tables();
        assert!(!names_equal_ignore_case_java("\u{df}", "ss", Some(&t)));
        assert!(!names_equal_ignore_case_java("\u{df}", "SS", Some(&t)));
    }

    #[test]
    fn capital_i_with_dot_above_does_not_match_ascii_i() {
        // 'İ' (U+0130) lowercases (Locale.ROOT) to the TWO-char string "i" + COMBINING DOT
        // ABOVE, so it does NOT match ASCII 'I'/'i' under toLowerCase-keyed matching.
        let t = jdk17_test_tables();
        assert!(!names_equal_ignore_case_java("\u{130}", "I", Some(&t)));
        assert!(!names_equal_ignore_case_java("\u{130}", "i", Some(&t)));
    }

    #[test]
    fn dotless_i_does_not_match_ascii_i_or_capital_i() {
        // 'ı' (U+0131) is already lowercase and maps to itself, so it does NOT unify with
        // ASCII 'I' (which lowercases to 'i') or 'i' -- unlike Java's `Character`-level
        // `equalsIgnoreCase`, which is not what Spark's footer matching uses.
        let t = jdk17_test_tables();
        assert!(!names_equal_ignore_case_java("\u{131}", "I", Some(&t)));
        assert!(!names_equal_ignore_case_java("\u{131}", "i", Some(&t)));
    }

    #[test]
    fn ascii_pairs_still_match() {
        let t = jdk17_test_tables();
        assert!(names_equal_ignore_case_java("Foo", "foo", Some(&t)));
        assert!(names_equal_ignore_case_java("BAR", "bar", Some(&t)));
        assert!(!names_equal_ignore_case_java("foo", "bar", Some(&t)));
    }

    #[test]
    fn ascii_matching_does_not_depend_on_jvm_tables() {
        let empty = JvmCaseTables::from_proto(&[], &[], &[]);
        assert!(names_equal_ignore_case_java("Foo", "foo", Some(&empty)));
    }

    #[test]
    fn differing_lengths_never_match() {
        let t = jdk17_test_tables();
        assert!(!names_equal_ignore_case_java("ab", "a", Some(&t)));
        assert!(!names_equal_ignore_case_java("", "a", Some(&t)));
    }

    #[test]
    fn digit_keeps_final_sigma_context_open() {
        // JDK-17-sourced: "A1Σ".toLowerCase(Locale.ROOT) == "a1ς" (FINAL sigma). The JDK's
        // Final_Cased condition runs on word boundaries, and a digit keeps "A1Σ" one word, so
        // the trailing sigma takes the final form -- unlike the Unicode-standard Final_Sigma
        // (and unlike `str::to_lowercase`), where the digit is not case-ignorable and blocks
        // the context, giving "a1σ".
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A1\u{3A3}"), "a1\u{3C2}");
        assert_ne!("A1\u{3A3}".to_lowercase(), "a1\u{3C2}");
        assert!(names_equal_ignore_case_java(
            "A1\u{3A3}",
            "a1\u{3C2}",
            Some(&t)
        ));
        assert!(!names_equal_ignore_case_java(
            "A1\u{3A3}",
            "a1\u{3C3}",
            Some(&t)
        ));
    }

    #[test]
    fn word_boundaries_and_following_cased_letters_block_final_sigma() {
        let t = jdk17_test_tables();
        // JDK-17-sourced expected values:
        assert_eq!(t.lowercase("A \u{3A3}"), "a \u{3C3}"); // space breaks the word
        assert_eq!(t.lowercase("A\u{3A3}B"), "a\u{3C3}b"); // cased letter follows
        assert_eq!(t.lowercase("\u{3A3}"), "\u{3C3}"); // isolated
        assert_eq!(t.lowercase("\u{3A3}A"), "\u{3C3}a"); // nothing cased before
    }

    #[test]
    fn greek_word_with_trailing_capital_sigma_lowercases_with_final_sigma() {
        // JDK-17-sourced: a transliteration of "Odysseus" in all-capital Greek, chosen for two
        // medial sigmas plus a word-final one: both medial Σ fold to plain σ and only the
        // word-final Σ folds to ς.
        let t = jdk17_test_tables();
        assert_eq!(
            t.lowercase("\u{39F}\u{394}\u{3A5}\u{3A3}\u{3A3}\u{395}\u{3A5}\u{3A3}"),
            "\u{3BF}\u{3B4}\u{3C5}\u{3C3}\u{3C3}\u{3B5}\u{3C5}\u{3C2}"
        );
        assert!(names_equal_ignore_case_java(
            "\u{39F}\u{394}\u{3A5}\u{3A3}\u{3A3}\u{395}\u{3A5}\u{3A3}",
            "\u{3BF}\u{3B4}\u{3C5}\u{3C3}\u{3C3}\u{3B5}\u{3C5}\u{3C2}",
            Some(&t)
        ));
    }

    #[test]
    fn greek_word_with_medial_sigma_lowercases_with_regular_sigma() {
        // JDK-17-sourced: cased letter, sigma, cased letter -- the sigma is medial, so it folds
        // to plain σ, and a name spelled with ς instead must NOT match.
        let t = jdk17_test_tables();
        assert_eq!(
            t.lowercase("\u{391}\u{3A3}\u{392}"),
            "\u{3B1}\u{3C3}\u{3B2}"
        );
        assert!(!names_equal_ignore_case_java(
            "\u{391}\u{3A3}\u{392}",
            "\u{3B1}\u{3C2}\u{3B2}",
            Some(&t)
        ));
    }

    #[test]
    fn mid_punctuation_joins_words_per_the_jdk_rules() {
        // JDK-17-sourced expected values. Mid-word punctuation (underscore, dash, period,
        // apostrophe) joins letter..letter and keeps the sigma context open; it does NOT join
        // against digits, two in a row break, and mid-num-only punctuation (comma) never joins
        // letters.
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A_\u{3A3}"), "a_\u{3C2}");
        assert_eq!(t.lowercase("A-\u{3A3}"), "a-\u{3C2}");
        assert_eq!(t.lowercase("A.\u{3A3}"), "a.\u{3C2}");
        assert_eq!(t.lowercase("A,\u{3A3}"), "a,\u{3C3}");
        assert_eq!(t.lowercase("A..\u{3A3}"), "a..\u{3C3}");
        assert_eq!(t.lowercase("A1.\u{3A3}"), "a1.\u{3C3}");
        assert_eq!(t.lowercase("A-1\u{3A3}"), "a-1\u{3C3}");
        assert_eq!(t.lowercase("A\u{3A3}_b"), "a\u{3C3}_b"); // joins to a cased letter after
    }

    #[test]
    fn cased_digit_is_cased_but_joins_words_like_a_digit_not_a_letter() {
        // U+2160 ROMAN NUMERAL ONE: cased (the JDK's hardcoded Other_Uppercase list) but
        // digit-typed (Nl), so it satisfies the scan's "found a cased letter" check when
        // reached directly, yet -- unlike an ordinary cased letter -- does NOT let mid-word
        // punctuation (only letter..letter) bridge past it; only mid-num punctuation
        // (digit..digit) does. This is the exact class-sequence gap the multi-special pair
        // sweep in `JvmLowercaseParitySuite` found: folding cased digits into the plain
        // cased-letter class let mid-word marks wrongly bridge past them.
        let t = jdk17_test_tables();
        // Directly adjacent to the sigma: cased, so the scan finds it either direction.
        assert_eq!(t.lowercase("\u{2160}\u{3A3}"), "\u{2170}\u{3C2}");
        assert_eq!(t.lowercase("A\u{3A3}\u{2160}"), "a\u{3C3}\u{2170}");
        // A mid-word mark ('-') does NOT bridge into a cased digit: non-final. (The cased
        // digit itself is not directly adjacent to sigma in either case, so the scan must
        // cross the dash to reach it -- and fails to, since mid-word only bridges
        // letter..letter.)
        assert_eq!(t.lowercase("\u{2160}-\u{3A3}"), "\u{2170}-\u{3C3}");
        assert_eq!(t.lowercase("A-\u{2160}-\u{3A3}"), "a-\u{2170}-\u{3C3}");
        // A mid-num mark (',') DOES bridge digit..digit into a cased digit: the '1' adjacent
        // to sigma establishes the digit-run state, then ',' bridges back to the cased digit.
        assert_eq!(t.lowercase("\u{2160},1\u{3A3}"), "\u{2170},1\u{3C2}");
    }

    #[test]
    fn combining_marks_are_transparent_to_the_sigma_scan() {
        // JDK-17-sourced: Mn marks ride along with their base ("Á" decomposed, an Arabic
        // fatha), and the İ expansion's own combining mark does not break adjacency.
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A\u{301}\u{3A3}"), "a\u{301}\u{3C2}");
        assert_eq!(t.lowercase("A\u{64E}\u{3A3}"), "a\u{64E}\u{3C2}");
        assert_eq!(t.lowercase("\u{130}\u{3A3}"), "i\u{307}\u{3C2}");
    }

    #[test]
    fn cased_combining_mark_counts_only_when_attached_to_a_word() {
        // JDK-17-sourced: U+0345 COMBINING GREEK YPOGEGRAMMENI is the one CASED combining
        // mark. Attached to a letter or digit it satisfies the "preceded by cased" condition;
        // base-less at string start it does not.
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A\u{345}\u{3A3}"), "a\u{345}\u{3C2}");
        assert_eq!(t.lowercase("1\u{345}\u{3A3}"), "1\u{345}\u{3C2}");
        assert_eq!(t.lowercase("\u{345}\u{3A3}"), "\u{345}\u{3C3}");
    }

    #[test]
    fn supplementary_cased_letter_closes_the_preceding_word() {
        // JDK-17-sourced: the legacy break iterator attaches a supplementary character to the
        // preceding word and closes it, so U+10400 blocks the backward scan (mid-string) but
        // satisfies it at string start, and always satisfies the forward scan.
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A\u{10400}\u{3A3}"), "a\u{10428}\u{3C3}");
        assert_eq!(t.lowercase("\u{10400}\u{3A3}"), "\u{10428}\u{3C2}");
        assert_eq!(t.lowercase("A\u{3A3}\u{10400}"), "a\u{3C3}\u{10428}");
    }

    #[test]
    fn danda_chains_only_into_numbers() {
        // JDK-17-sourced: the danda terminates a word; the segment continues past it only
        // into digits.
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A\u{964}1\u{3A3}"), "a\u{964}1\u{3C2}");
        assert_eq!(t.lowercase("A\u{964}\u{3A3}"), "a\u{964}\u{3C3}");
    }

    #[test]
    fn cf_format_chars_are_fully_transparent_but_mn_marks_are_not() {
        // Real-JDK-verified: `<ignore>=[:Cf:]` loops on every state of the legacy DFA, so
        // format characters are deleted from the sequence before segmentation -- a ZWJ
        // anywhere in a mid-punctuation bridge leaves the bridge intact ("A-<ZWJ>Σ" and
        // "AΣ-<ZWJ>b" behave exactly like "A-Σ" / "AΣ-b") -- while an Mn mark in the same
        // position blocks it (the mark is orphaned: its text-order predecessor is the
        // punctuation, not a letter-base).
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A-\u{200D}\u{3A3}"), "a-\u{200D}\u{3C2}");
        assert_eq!(t.lowercase("A-\u{301}\u{3A3}"), "a-\u{301}\u{3C3}");
        // The forward mid-letter bridge crosses a ZWJ to the cased letter beyond, so the
        // sigma is NOT final -- the exact residual shape the format filter fixes.
        assert_eq!(t.lowercase("A\u{3A3}-\u{200D}b"), "a\u{3C3}-\u{200D}b");
        assert_eq!(
            t.lowercase("A\u{3A3}-\u{200D}\u{200D}b"),
            "a\u{3C3}-\u{200D}\u{200D}b"
        );
        // An Mn mark anywhere in the same rider chain blocks that bridge.
        assert_eq!(
            t.lowercase("A\u{3A3}-\u{200D}\u{301}b"),
            "a\u{3C2}-\u{200D}\u{301}b"
        );
        // Riders trailing the sigma itself bridge onward regardless of Cf vs Mn.
        assert_eq!(t.lowercase("A\u{3A3}\u{200D}-B"), "a\u{3C3}\u{200D}-b");
        assert_eq!(t.lowercase("A\u{3A3}\u{301}-B"), "a\u{3C3}\u{301}-b");
    }

    #[test]
    fn leading_format_chars_defeat_the_supplementary_text_start_join() {
        // Real-JDK-verified: a cased supplementary char forms a word at RAW text start
        // ("𐐀Σ" is final), but a leading format char occupies the DFA's initial state, so
        // the same shape behind a ZWJ is non-final.
        let t = jdk17_test_tables();
        assert_eq!(
            t.lowercase("\u{200D}\u{10400}\u{3A3}"),
            "\u{200D}\u{10428}\u{3C3}"
        );
    }

    #[test]
    fn supplementary_mark_anchors_a_riding_cased_mark_only_off_a_real_base() {
        // Real-JDK-verified: a supplementary combining mark (U+11374) attaches to the
        // preceding word but never forms one. A U+0345 riding on it counts as cased exactly
        // when the run hangs off a real base -- and for a mid-letter bridge, only a
        // letter-flavored one.
        let t = jdk17_test_tables();
        assert_eq!(
            t.lowercase("A\u{11374}\u{345}\u{3A3}"),
            "a\u{11374}\u{345}\u{3C2}"
        );
        assert_eq!(
            t.lowercase("\u{11374}\u{345}\u{3A3}"),
            "\u{11374}\u{345}\u{3C3}"
        );
        assert_eq!(
            t.lowercase("A\u{11374}\u{345}-\u{3A3}"),
            "a\u{11374}\u{345}-\u{3C2}"
        );
        assert_eq!(
            t.lowercase("\u{2160}\u{11374}\u{345}-\u{3A3}"),
            "\u{2170}\u{11374}\u{345}-\u{3C3}"
        );
    }

    #[test]
    fn supplementary_digit_carries_a_riding_cased_mark_only_in_digit_context() {
        // Real-JDK-verified: a word-forming supplementary digit (U+1D7D3) backs a riding
        // U+0345 against a bare sigma and across mid-num punctuation in digit context, but
        // never across mid-letter punctuation.
        let t = jdk17_test_tables();
        assert_eq!(
            t.lowercase("A\u{1D7D3}\u{345}\u{3A3}"),
            "a\u{1D7D3}\u{345}\u{3C2}"
        );
        assert_eq!(
            t.lowercase("A\u{1D7D3}\u{345},1\u{3A3}"),
            "a\u{1D7D3}\u{345},1\u{3C2}"
        );
        assert_eq!(
            t.lowercase("A\u{1D7D3}\u{345}-\u{3A3}"),
            "a\u{1D7D3}\u{345}-\u{3C3}"
        );
    }

    #[test]
    fn assigned_noncased_supplementary_letter_backs_a_following_cased_mark() {
        // Real-JDK-verified: an assigned non-cased supplementary letter (Lo, e.g. CJK
        // Extension B) is still a genuine letter-base -- unlike a plain word boundary, it can
        // back a following CLASS_EXTEND_CASED (U+0345), which is cased once its run is
        // attached, even though the base itself never counts as cased directly.
        let t = jdk17_test_tables();
        assert_eq!(
            t.lowercase("\u{20000}\u{345}\u{3A3}"),
            "\u{20000}\u{345}\u{3C2}"
        );
        assert_eq!(t.lowercase("\u{20000}\u{3A3}"), "\u{20000}\u{3C3}");
    }

    #[test]
    fn backward_mid_word_bridge_credits_a_cased_mark_riding_a_noncased_base() {
        // Real-JDK-verified: scanning backward through a mid-word connector to find its base
        // legitimately walks riders-then-base (marks trail their base in text order). A
        // CLASS_EXTEND_CASED (U+0345) found along that walk is cased once the bridge
        // validates, regardless of whether the ultimate base underneath it is itself cased.
        let t = jdk17_test_tables();
        assert_eq!(
            t.lowercase("\u{20000}\u{345}_\u{3A3}"),
            "\u{20000}\u{345}_\u{3C2}"
        );
    }

    #[test]
    fn forward_mid_word_bridge_rejects_an_orphaned_mark_after_the_punctuation() {
        // Real-JDK-verified: unlike the backward-scan bridge, a mark found IMMEDIATELY after
        // mid-word punctuation (before any real base) is orphaned -- its text-order
        // predecessor is the punctuation, not a letter -- so the forward bridge must reject
        // it rather than skipping past it to a real letter beyond.
        let t = jdk17_test_tables();
        assert_eq!(t.lowercase("A\u{3A3}-\u{301}B"), "a\u{3C2}-\u{301}b");
    }

    #[test]
    fn absent_tables_fall_back_to_rust_lowercase() {
        // Without shipped tables -- non-scan `SparkParquetOptions` consumers (e.g. general
        // struct-to-struct type conversion), Rust-only unit tests of the matching logic that
        // skip the JVM proto round trip, or a defensively malformed plan -- matching falls
        // back to `str::to_lowercase`: correct for all simple mappings (Kelvin sign, ASCII)
        // and knowingly divergent from the JVM only where the Unicode snapshots or the sigma
        // word-context differ.
        assert_eq!(java_lowercase("A1\u{3A3}", None), "a1\u{3C3}");
        assert!(names_equal_ignore_case_java("\u{212A}", "k", None));
        assert!(names_equal_ignore_case_java("Foo", "foo", None));
        assert!(!names_equal_ignore_case_java(
            "A1\u{3A3}",
            "a1\u{3C2}",
            None
        ));
    }

    #[test]
    fn unknown_class_values_read_as_word_boundaries() {
        // A newer JVM-side generator may ship class values this build does not know; they must
        // degrade to the safe reading (word boundary), never crash or misclassify.
        let t = JvmCaseTables::from_proto(
            &[0x41],
            &["a".to_string()],
            &[0x41, 0x5A, 99, 0x3B1, 0x3C9, CLASS_ALETTER_CASED as u32],
        );
        // 'A' (class 99 -> boundary) does not open the sigma context...
        assert_eq!(t.lowercase("A\u{3A3}"), "a\u{3C3}");
        // ...while a known cased class still does.
        assert_eq!(t.lowercase("\u{3B1}\u{3A3}"), "\u{3B1}\u{3C2}");
    }

    #[test]
    fn malformed_proto_degrades_entry_by_entry() {
        // Misaligned lowercase arrays: extra codepoints without replacements are dropped.
        let t = JvmCaseTables::from_proto(&[0x41, 0x42], &["a".to_string()], &[]);
        assert_eq!(t.lowercase("AB"), "aB");
        // A trailing partial triple and an inverted range are dropped; the valid triple works.
        let t = JvmCaseTables::from_proto(
            &[],
            &[],
            &[
                0x61,
                0x7A,
                CLASS_ALETTER_CASED as u32,
                0x5A,
                0x41, // inverted -- dropped
                CLASS_ALETTER_CASED as u32,
                0x30, // trailing partial triple -- dropped
            ],
        );
        assert_eq!(t.lowercase("a\u{3A3}"), "a\u{3C2}");
        assert_eq!(t.lowercase("A\u{3A3}"), "A\u{3C3}");
    }

    #[test]
    fn equal_tables_hash_and_compare_equal() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = jdk17_test_tables();
        let b = jdk17_test_tables();
        assert_eq!(a, b);
        let hash = |t: &JvmCaseTables| {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&a), hash(&b));
        let c = JvmCaseTables::from_proto(&[0x41], &["a".to_string()], &[]);
        assert_ne!(a, c);
    }

    // -- remap_physical_schema: end-to-end wiring of the Java-parity matcher into the schema
    // remap that Spark's scan relies on. --

    #[test]
    fn remap_case_sensitive_does_not_fold_kelvin_sign_to_ascii_k() {
        let logical = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let physical = Arc::new(Schema::new(vec![Field::new(
            "\u{212A}",
            DataType::Int64,
            true,
        )]));

        let (remapped, name_map) = remap_physical_schema(
            &logical,
            &physical,
            /* case_sensitive */ true,
            Some(&jdk17_test_tables()),
            false,
            false,
        )
        .unwrap();

        // No case-insensitive fallback in case-sensitive mode: the physical field name is left
        // untouched and no remap entry is recorded.
        assert_eq!(remapped.field(0).name(), "\u{212A}");
        assert!(name_map.is_empty());
    }

    #[test]
    fn remap_case_insensitive_folds_kelvin_sign_to_ascii_k() {
        let logical = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let physical = Arc::new(Schema::new(vec![Field::new(
            "\u{212A}",
            DataType::Int64,
            true,
        )]));

        let (remapped, name_map) = remap_physical_schema(
            &logical,
            &physical,
            /* case_sensitive */ false,
            Some(&jdk17_test_tables()),
            false,
            false,
        )
        .unwrap();

        // The physical field is renamed to the logical name so the default expr adapter's
        // exact-name lookup hits, and the reverse map records the original physical name.
        assert_eq!(remapped.field(0).name(), "k");
        assert_eq!(name_map.get("k").map(String::as_str), Some("\u{212A}"));
    }

    #[test]
    fn remap_field_id_shield_is_exact_in_case_sensitive_mode() {
        // Logical `k` carries field ID 5, so Spark's `matchIdField` resolves it strictly by
        // ID; a physical field named exactly `k` with no matching ID must be shielded from
        // the downstream exact-name lookup. Physical `K`, however, is a DIFFERENT name under
        // case-sensitive matching (Spark's `matchCaseSensitiveField` keys on the exact
        // string), so it must stay untouched: nothing can name-match it, and hiding it would
        // wrongly null a legitimate exact-name lookup elsewhere. JVM case tables are only
        // shipped when `case_sensitive = false`, so `tables: None` is the live configuration.
        let logical = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "5".to_string(),
            )]))]));
        let physical = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("K", DataType::Int64, true),
        ]));

        let (remapped, name_map) = remap_physical_schema(
            &logical, &physical, /* case_sensitive */ true, /* case_tables */ None,
            /* use_field_id */ true, /* ignore_missing_field_id */ true,
        )
        .unwrap();

        assert_ne!(remapped.field(0).name(), "k");
        assert_ne!(remapped.field(0).name(), "K");
        assert_eq!(remapped.field(1).name(), "K");
        assert!(name_map.is_empty());
    }

    #[test]
    fn remap_case_sensitive_keeps_exact_name_for_id_less_logical_field() {
        // Greek capital Kappa (U+039A) carries field ID 1; lowercase kappa (U+03BA) carries
        // no ID. The file holds only `κ` with field ID 2. Spark null-fills `Κ` (ID 1 absent,
        // `matchIdField` -> fake REQUESTED name) but resolves the ID-less `κ` by exact name
        // through `matchCaseSensitiveField`, reading the real column. The physical `κ` must
        // therefore survive the remap untouched -- `Κ` and `κ` only collide under a case
        // fold, which case-sensitive matching must not apply.
        let logical = Arc::new(Schema::new(vec![
            Field::new("\u{39A}", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("\u{3BA}", DataType::Int64, true),
        ]));
        let physical = Arc::new(Schema::new(vec![Field::new(
            "\u{3BA}",
            DataType::Int64,
            true,
        )
        .with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )]))]));

        let (remapped, name_map) = remap_physical_schema(
            &logical, &physical, /* case_sensitive */ true, /* case_tables */ None,
            /* use_field_id */ true, /* ignore_missing_field_id */ false,
        )
        .unwrap();

        assert_eq!(remapped.field(0).name(), "\u{3BA}");
        assert!(name_map.is_empty());
    }

    #[test]
    fn remap_case_insensitive_claim_beats_unmatched_id_shield() {
        // Case-insensitive variant of the Kappa scenario. Spark's `matchCaseInsensitiveField`
        // resolves the ID-less requested `κ` through the `toLowerCase(Locale.ROOT)`-keyed
        // physical field map, which contains the file's `κ` -- the unmatched-ID requested `Κ`
        // only gets its own REQUESTED name faked and never blocks that lookup. So the
        // physical `κ` must be claimed by the name match (and kept), not hidden by the
        // shield, even though `Κ` and `κ` are equal under the case fold.
        let logical = Arc::new(Schema::new(vec![
            Field::new("\u{39A}", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("\u{3BA}", DataType::Int64, true),
        ]));
        let physical = Arc::new(Schema::new(vec![Field::new(
            "\u{3BA}",
            DataType::Int64,
            true,
        )
        .with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )]))]));

        let (remapped, name_map) = remap_physical_schema(
            &logical, &physical, /* case_sensitive */ false, /* case_tables */ None,
            /* use_field_id */ true, /* ignore_missing_field_id */ false,
        )
        .unwrap();

        assert_eq!(remapped.field(0).name(), "\u{3BA}");
        assert!(name_map.is_empty());
    }

    #[test]
    fn remap_shields_stray_physical_field_named_like_id_matched_logical_field() {
        // The file holds a stray ID-less `A` FIRST and the real ID match `a` (ID 1) second.
        // Spark reads requested `A` (ID 1) from physical `a` via `matchIdField`; the stray
        // `A` is never requested. After the remap renames `a` -> `A`, the stray physical `A`
        // must not remain as a second exact-name candidate ahead of it, or the downstream
        // adapter's name lookup would resolve `A` to the wrong column's data.
        let logical = Arc::new(Schema::new(vec![Field::new("A", DataType::Int64, true)
            .with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )]))]));
        let physical = Arc::new(Schema::new(vec![
            Field::new("A", DataType::Int64, true),
            Field::new("a", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        let (remapped, name_map) = remap_physical_schema(
            &logical, &physical, /* case_sensitive */ true, /* case_tables */ None,
            /* use_field_id */ true, /* ignore_missing_field_id */ false,
        )
        .unwrap();

        assert_ne!(remapped.field(0).name(), "A");
        assert_ne!(remapped.field(0).name(), "a");
        assert_eq!(remapped.field(1).name(), "A");
        assert_eq!(name_map.get("A").map(String::as_str), Some("a"));
    }

    #[test]
    fn remap_fake_names_never_collide_with_real_columns() {
        // A real column may legitimately be named like the fake-name pattern. Spark's
        // `generateFakeColumnName` embeds a random UUID, so its fakes can never shadow a
        // real column; the deterministic counter here must skip past reserved names to give
        // the same guarantee, otherwise the shielded field would duplicate the real
        // `__comet_unmatched_field_id_1` and could steal its exact-name match.
        let logical = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("__comet_unmatched_field_id_1", DataType::Int64, true),
        ]));
        let physical = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("__comet_unmatched_field_id_1", DataType::Int64, true),
        ]));

        let (remapped, name_map) = remap_physical_schema(
            &logical, &physical, /* case_sensitive */ true, /* case_tables */ None,
            /* use_field_id */ true, /* ignore_missing_field_id */ true,
        )
        .unwrap();

        // Physical `a` collides with the ID-bearing logical `a` (its ID is absent from the
        // file) and gets shielded -- but not with the taken fake name.
        assert_eq!(remapped.field(0).name(), "__comet_unmatched_field_id_2");
        // The real column matching the fake pattern is untouched and still name-matchable.
        assert_eq!(remapped.field(1).name(), "__comet_unmatched_field_id_1");
        assert!(name_map.is_empty());
    }
}
