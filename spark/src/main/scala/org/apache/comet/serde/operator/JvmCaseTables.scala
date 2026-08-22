/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.comet.serde.operator

import java.text.BreakIterator
import java.util.Locale

import org.apache.comet.serde.OperatorOuterClass

/**
 * Generates, from the RUNNING JVM, the case tables the native Parquet scan needs to reproduce
 * this JVM's `String.toLowerCase(Locale.ROOT)`, which Spark's Parquet footer field matching is
 * built on. Shipping the running JVM's own data (rather than pinning a Unicode snapshot) keeps
 * the native matcher correct by construction for whatever JDK executes the query.
 *
 * Two data sets are generated lazily, once per JVM (cached in [[generated]], ~15KB):
 *
 *   - a lowercase table: every codepoint whose single-codepoint string lowercases
 *     non-identically, with its full (possibly multi-char, e.g. U+0130 -> "i" + U+0307)
 *     replacement;
 *   - a word-break classification of every codepoint for the Greek capital sigma's contextual
 *     (final/non-final) lowering, `java.lang.ConditionalSpecialCasing`'s only locale-independent
 *     conditional mapping.
 *
 * Java's final-sigma rule is WORD-BOUNDARY based (`BreakIterator.getWordInstance`), NOT the
 * Unicode-standard Final_Sigma "case-ignorable" skip: `isFinalCased` asks whether a cased
 * character precedes the sigma within its word with none following inside the word. The
 * word-break classes are the UAX#29-style classes (ALetter, Numeric, MidLetter, MidNum,
 * MidNumLet, Extend, Format) as the RUNNING JDK's legacy break iterator actually realizes them,
 * plus classes for its pre-UAX#29 extensions (the danda, and the supplementary-plane behaviors of
 * its UTF-16 DFA). Rather than pinning any character-property model, [[classify]] PROBES every
 * codepoint through `BreakIterator.isBoundary` in discriminating templates, so the shipped
 * classes track the executing JDK's break data exactly; the JDK's `isCased` predicate then splits
 * cased variants. Classes ship as (start, end, class) range triples.
 *
 * [[mirrorLowercase]] is a line-for-line mirror of the native algorithm
 * (`JvmCaseTables::lowercase` in schema_adapter.rs) over these same generated tables -- any
 * change to one side must be mirrored in the other. `JvmLowercaseParitySuite` proves
 * `mirrorLowercase == String.toLowerCase(Locale.ROOT)` for the running JDK across the full
 * codepoint space, which then transfers to native. Calibrated to zero mismatches on JDK 17, 21,
 * and 25 against full-codepoint context sweeps, multi-special triple sweeps, and million-string
 * fuzzing.
 */
private[comet] object JvmCaseTables {

  // Wire contract shared with `JvmCaseTables` in native/core/src/parquet/schema_adapter.rs:
  // lowerCps/lowerRepls are index-aligned; classRanges is a flat (start, end, class) triple
  // list, sorted and disjoint, class ids 1-15 (0 = unshipped word boundary).
  private[comet] val ClassALetterCased = 1 // word-joining BMP letter-base, cased
  private[comet] val ClassALetter = 2 // word-joining BMP letter-base, not cased
  private[comet] val ClassNumeric = 3 // BMP digit-base: joins words, bridges only mid-num
  private[comet] val ClassMidLetter = 4 // joins only letter..letter ('-', '_', U+00AD, ...)
  private[comet] val ClassMidNum = 5 // joins only digit..digit (',', U+066B)
  private[comet] val ClassMidNumLet = 6 // both of the above ('"', '\'', '.')
  private[comet] val ClassSuppCased = 7 // cased supplementary: attaches and closes the word
  private[comet] val ClassDanda = 8 // U+0964/U+0965: word-terminal, chains only into digits
  private[comet] val ClassExtendCased = 9 // U+0345: cased only when attached to a word
  private[comet] val ClassNumericCased = 10 // cased digit-base (Nl Roman numerals)
  private[comet] val ClassExtend = 11 // Mn/Me marks: attach only to letter/digit bases
  private[comet] val ClassSuppLetter = 12 // word-forming non-cased supplementary letter
  private[comet] val ClassFormat = 13 // Cf format chars: fully transparent (WB4-style)
  // Supplementary chars that attach to the preceding word but never form one themselves
  // (supplementary combining marks, tag characters): a cased mark riding on one belongs to
  // the sigma's word only when the run hangs off a real base ([[suppMnAnchor]]).
  private[comet] val ClassSuppMn = 14
  // Word-forming supplementary digit: like ClassSuppLetter except a riding cased mark
  // carries only across mid-num (digit-context) punctuation, never mid-letter.
  private[comet] val ClassSuppNum = 15

  /** The generated tables: aligned lowercase arrays plus (start, end, class) range triples. */
  private[comet] final case class Tables(
      lowerCps: Array[Int],
      lowerRepls: Array[String],
      classRanges: Array[Int])

  /**
   * Pinned from `ConditionalSpecialCasing.isCased`: Lu/Ll/Lt by `getType` plus the JDK's
   * hardcoded Other_Uppercase/Other_Lowercase ranges (byte-identical in JDK 17 and 21).
   */
  private def isCasedJdk(cp: Int): Boolean = {
    val t = Character.getType(cp)
    if (t == Character.LOWERCASE_LETTER || t == Character.UPPERCASE_LETTER ||
      t == Character.TITLECASE_LETTER) {
      true
    } else {
      (cp >= 0x02b0 && cp <= 0x02b8) || (cp >= 0x02c0 && cp <= 0x02c1) ||
      (cp >= 0x02e0 && cp <= 0x02e4) || cp == 0x0345 || cp == 0x037a ||
      (cp >= 0x1d2c && cp <= 0x1d61) || (cp >= 0x2160 && cp <= 0x217f) ||
      (cp >= 0x24b6 && cp <= 0x24e9)
    }
  }

  private def isBoundary(bi: BreakIterator, s: String, pos: Int): Boolean = {
    bi.setText(s)
    bi.isBoundary(pos)
  }

  /**
   * One word-break class per codepoint, derived EMPIRICALLY from the running JDK's own
   * `BreakIterator.getWordInstance(Locale.ROOT)` by probing `isBoundary` in discriminating
   * templates ('a'/'b' anchor letters, '1'/'2' digits, '-' mid-letter, ',' mid-num, U+0345 the
   * cased mark, U+03A3 the sigma). Everything without a joining fingerprint (spaces, symbols,
   * kana/kanji -- which form their own segments a sigma can never share -- unassigned codepoints,
   * surrogates) is a word boundary (class 0, never shipped).
   */
  private[comet] def classify(bi: BreakIterator, cp: Int): Int = {
    if (cp >= 0xd800 && cp <= 0xdfff) {
      return 0 // surrogate halves: unreachable from well-formed UTF-8 column names
    }
    val x = new String(Character.toChars(cp))
    if (cp > 0xffff) {
      // Supplementary: the legacy UTF-16 DFA attaches every non-isolated supplementary char
      // to the preceding word and closes the word after it. Discriminate: attach-close
      // ("aXb"), word-forming at text start ("Xa"), whether a riding mark starts a fresh
      // segment ("X" + U+0345 + sigma), and letter- vs digit-flavored mid bridging.
      if (!isBoundary(bi, "a" + x + "b", 1)) {
        if (!isBoundary(bi, x + "a", 2)) {
          if (isCasedJdk(cp)) return ClassSuppCased
          val casedMark = 0x0345.toChar.toString
          val sigma = 0x03a3.toChar.toString
          val grab = x + casedMark + sigma
          if (isBoundary(bi, grab, grab.length - 1)) return ClassSuppMn
          val mid = "A" + x + casedMark + "-" + sigma
          if (isBoundary(bi, mid, mid.length - 1)) ClassSuppNum else ClassSuppLetter
        } else {
          ClassSuppMn
        }
      } else {
        0
      }
    } else {
      val axb = "a" + x + "b"
      val j1 = !isBoundary(bi, axb, 1)
      val j2 = !isBoundary(bi, axb, 2)
      if (j1 && j2) {
        val axxb = "a" + x + x + "b"
        val dbl = !isBoundary(bi, axxb, 1) && !isBoundary(bi, axxb, 2) &&
          !isBoundary(bi, axxb, 3)
        val n12 = "1" + x + "2"
        val num = !isBoundary(bi, n12, 1) && !isBoundary(bi, n12, 2)
        if (dbl) {
          // Full joiner: letter/digit-base, attached mark, or transparent format char,
          // separated by which mid punctuation it bridges.
          val bridgeL = !isBoundary(bi, "a-" + x + "c", 1)
          val bridgeN = !isBoundary(bi, "1," + x + "2", 1)
          if (bridgeL && bridgeN) ClassFormat
          else if (bridgeL) { if (isCasedJdk(cp)) ClassALetterCased else ClassALetter }
          else if (bridgeN) { if (isCasedJdk(cp)) ClassNumericCased else ClassNumeric }
          else if (isCasedJdk(cp)) ClassExtendCased
          else ClassExtend
        } else {
          // Joins a single letter..letter gap only: mid punctuation.
          if (num) ClassMidNumLet else ClassMidLetter
        }
      } else if (j1 && !j2) {
        // Attaches to the preceding word and closes it; the danda also chains into digits.
        val ax1 = "a" + x + "1"
        if (!isBoundary(bi, ax1, 1) && !isBoundary(bi, ax1, 2)) ClassDanda else 0
      } else {
        val n12 = "1" + x + "2"
        if (!isBoundary(bi, n12, 1) && !isBoundary(bi, n12, 2)) ClassMidNum else 0
      }
    }
  }

  /** Generated once per JVM; both the proto population and the parity suite read this. */
  private[comet] lazy val generated: Tables = {
    val bi = BreakIterator.getWordInstance(Locale.ROOT)
    val cps = new java.util.ArrayList[Integer]()
    val repls = new java.util.ArrayList[String]()
    val ranges = new java.util.ArrayList[Integer]()
    var rangeStart = -1
    var rangeClass = 0
    var cp = 0
    while (cp <= 0x10ffff) {
      if (cp < 0xd800 || cp > 0xdfff) {
        val s = new String(Character.toChars(cp))
        val low = s.toLowerCase(Locale.ROOT)
        if (low != s) {
          cps.add(cp)
          repls.add(low)
        }
      }
      val cls = classify(bi, cp)
      if (cls != rangeClass) {
        if (rangeClass != 0) {
          ranges.add(rangeStart)
          ranges.add(cp - 1)
          ranges.add(rangeClass)
        }
        rangeStart = cp
        rangeClass = cls
      }
      cp += 1
    }
    if (rangeClass != 0) {
      ranges.add(rangeStart)
      ranges.add(0x10ffff)
      ranges.add(rangeClass)
    }
    Tables(
      lowerCps = cps.toArray(new Array[Integer](0)).map(_.intValue()),
      lowerRepls = repls.toArray(new Array[String](0)),
      classRanges = ranges.toArray(new Array[Integer](0)).map(_.intValue()))
  }

  // Boxed views cached once so per-scan proto population is a bulk addAll, not a re-box.
  private lazy val lowerCpsBoxed: java.util.List[Integer] = {
    val list = new java.util.ArrayList[Integer](generated.lowerCps.length)
    generated.lowerCps.foreach(list.add(_))
    java.util.Collections.unmodifiableList(list)
  }
  private lazy val lowerReplsBoxed: java.util.List[String] =
    java.util.Collections.unmodifiableList(java.util.Arrays.asList(generated.lowerRepls: _*))
  private lazy val classRangesBoxed: java.util.List[Integer] = {
    val list = new java.util.ArrayList[Integer](generated.classRanges.length)
    generated.classRanges.foreach(list.add(_))
    java.util.Collections.unmodifiableList(list)
  }

  /** Attach the running JVM's case tables to a scan's common proto (case-insensitive only). */
  private[comet] def populate(builder: OperatorOuterClass.NativeScanCommon.Builder): Unit = {
    builder.addAllJvmLowerCp(lowerCpsBoxed)
    builder.addAllJvmLowerRepl(lowerReplsBoxed)
    builder.addAllJvmSigmaClassRanges(classRangesBoxed)
  }

  // ---------------------------------------------------------------------------------------
  // Mirror of the native algorithm (schema_adapter.rs `JvmCaseTables::lowercase`). Test-facing:
  // the parity suite proves this equals `String.toLowerCase(Locale.ROOT)` on the running JDK.
  //
  // The scans run over a FORMAT-FILTERED codepoint sequence (WB4-style: the legacy break
  // iterator's `<ignore>` class loops on every DFA state, so Cf characters are deleted before
  // segmentation -- this is what lets a pure-format rider chain bridge mid punctuation, e.g.
  // "A<sigma>-<ZWJ>b" is one word exactly like "A<sigma>-b").
  // ---------------------------------------------------------------------------------------

  /** Class of `cp` per the generated (start, end, class) triples; 0 = word boundary. */
  private[comet] def sigmaClassOf(cp: Int): Int = {
    val r = generated.classRanges
    var lo = 0
    var hi = r.length / 3 - 1
    while (lo <= hi) {
      val mid = (lo + hi) >>> 1
      if (cp < r(3 * mid)) hi = mid - 1
      else if (cp > r(3 * mid + 1)) lo = mid + 1
      else return r(3 * mid + 2)
    }
    0
  }

  private def isLetterBase(cls: Int): Boolean =
    cls == ClassALetterCased || cls == ClassALetter || cls == ClassSuppCased ||
      cls == ClassSuppLetter

  private def isDigitBase(cls: Int): Boolean =
    cls == ClassNumeric || cls == ClassNumericCased

  /** First position at/beyond `start` (step -1/+1) that isn't ClassExtend; -1 off the array. */
  private def skipExtends(cps: Array[Int], start: Int, step: Int): Int = {
    var k = start
    while (k >= 0 && k < cps.length && sigmaClassOf(cps(k)) == ClassExtend) k += step
    if (k < 0 || k >= cps.length) -1 else k
  }

  /**
   * As [[skipExtends]] but also skips ClassExtendCased, reporting whether one was walked: a cased
   * mark (U+0345) crossed while looking for a base is itself cased whenever the landing validates
   * the run.
   */
  private def skipExtendsTrackingCased(cps: Array[Int], start: Int, step: Int): (Int, Boolean) = {
    var k = start
    var sawCased = false
    while (k >= 0 && k < cps.length && {
        val c = sigmaClassOf(cps(k))
        c == ClassExtend || c == ClassExtendCased
      }) {
      if (sigmaClassOf(cps(k)) == ClassExtendCased) sawCased = true
      k += step
    }
    (if (k < 0 || k >= cps.length) -1 else k, sawCased)
  }

  private val AnchorNone = 0
  private val AnchorLetter = 1
  private val AnchorDigit = 2

  /**
   * What the supplementary-mark run at `k` (ClassSuppMn) ultimately hangs off, walking down
   * through further marks and supplementary chars: a letter-flavored base, a digit-flavored base,
   * or nothing word-forming. A cased mark riding the run belongs to the sigma's word only per
   * this anchor.
   */
  private def suppMnAnchor(cps: Array[Int], k: Int): Int = {
    var m = k - 1
    while (m >= 0 && {
        val c = sigmaClassOf(cps(m))
        c == ClassSuppMn || c == ClassExtend || c == ClassExtendCased
      }) {
      m -= 1
    }
    if (m < 0) return AnchorNone
    val a = sigmaClassOf(cps(m))
    if (a == ClassALetterCased || a == ClassALetter || a == ClassSuppCased ||
      a == ClassSuppLetter) {
      AnchorLetter
    } else if (a == ClassNumeric || a == ClassNumericCased || a == ClassSuppNum) {
      AnchorDigit
    } else {
      AnchorNone
    }
  }

  private def scanBackFindsCased(cps: Array[Int], i: Int, leadingFormat: Boolean): Boolean = {
    var lastLetter = true // the sigma itself is a letter
    var j = i - 1
    while (j >= 0) {
      sigmaClassOf(cps(j)) match {
        case ClassALetterCased | ClassNumericCased => return true
        case ClassALetter => lastLetter = true; j -= 1
        case ClassNumeric => lastLetter = false; j -= 1
        case ClassExtend =>
          // Non-cased marks attach only to a real base below them; anything else (mid
          // punctuation, danda, boundary, text start) leaves the run unattached.
          val k = skipExtends(cps, j, -1)
          if (k < 0) return false
          val b = sigmaClassOf(cps(k))
          val isContinuer = b == ClassALetterCased || b == ClassNumericCased ||
            b == ClassNumeric || b == ClassExtendCased || b == ClassALetter ||
            b == ClassSuppCased || b == ClassSuppLetter || b == ClassSuppNum
          if (!isContinuer) return false
          j = k
        case ClassSuppCased =>
          // Closes the preceding word, so the scan stops -- except at RAW text start (no
          // filtered-out leading format chars), where the DFA keeps it joined to what
          // follows.
          return j == 0 && !leadingFormat
        case ClassSuppLetter | ClassSuppMn | ClassSuppNum =>
          // Attach/close and never themselves cased; nothing beyond is reachable.
          return false
        case ClassExtendCased =>
          // Cased combining mark (U+0345): cased when its run hangs off a base -- a BMP
          // letter/digit, a word-forming supplementary char (which closes a word right
          // below the mark, merging the mark into the sigma's segment), or an ANCHORED
          // supplementary mark.
          val (k, _) = skipExtendsTrackingCased(cps, j - 1, -1)
          if (k < 0) return false
          val b = sigmaClassOf(cps(k))
          if (b == ClassALetterCased || b == ClassNumeric || b == ClassNumericCased ||
            b == ClassALetter || b == ClassSuppCased || b == ClassSuppLetter ||
            b == ClassSuppNum) {
            return true
          }
          if (b == ClassSuppMn) return suppMnAnchor(cps, k) != AnchorNone
          return false
        case ClassDanda =>
          // Backward across a danda: the word part before it must end in letters (grammar:
          // letters, optional danda, then number+word chains) -- or carry a riding cased
          // mark on a word-forming base, or be a cased supplementary char at text start --
          // and the danda itself chains only into digits after it.
          if (lastLetter) return false
          val (k, sawCasedMark) = skipExtendsTrackingCased(cps, j - 1, -1)
          if (k < 0) return false
          val b = sigmaClassOf(cps(k))
          if (b == ClassALetterCased) return true
          if (b == ClassSuppCased) return sawCasedMark || (k == 0 && !leadingFormat)
          if (b == ClassSuppLetter) return sawCasedMark
          if (b == ClassSuppMn) {
            return sawCasedMark && suppMnAnchor(cps, k) == AnchorLetter
          }
          if (b != ClassALetter) return false
          if (sawCasedMark) return true
          lastLetter = true
          j = k
        case cls @ (ClassMidLetter | ClassMidNum | ClassMidNumLet) =>
          // `<mid-letter><let>` / `<mid-num><digit>` require a genuine letter/digit base
          // before the punctuation; scanning backward legitimately walks marks-then-base
          // (marks trail their base). A cased mark walked over rides whatever the
          // punctuation hangs off, including a context-matching anchored supplementary
          // mark or supplementary digit.
          val mwOk = cls == ClassMidLetter || cls == ClassMidNumLet
          val mnOk = cls == ClassMidNum || cls == ClassMidNumLet
          val (realPos, sawCasedMark) = skipExtendsTrackingCased(cps, j - 1, -1)
          if (realPos < 0) return false
          val b = sigmaClassOf(cps(realPos))
          if (lastLetter && mwOk && sawCasedMark && b == ClassSuppMn &&
            suppMnAnchor(cps, realPos) == AnchorLetter) {
            return true
          }
          if (!lastLetter && mnOk && sawCasedMark &&
            (b == ClassSuppNum ||
              (b == ClassSuppMn && suppMnAnchor(cps, realPos) == AnchorDigit))) {
            return true
          }
          val bridgeValid = (lastLetter && mwOk && isLetterBase(b)) ||
            (!lastLetter && mnOk && isDigitBase(b))
          if (!bridgeValid) return false
          if (sawCasedMark) return true
          j = realPos
        case _ => return false
      }
    }
    false
  }

  private def scanFwdFindsCased(cps: Array[Int], i: Int): Boolean = {
    var lastLetter = true
    var j = i + 1
    while (j < cps.length) {
      sigmaClassOf(cps(j)) match {
        case ClassALetterCased | ClassNumericCased => return true
        case ClassALetter => lastLetter = true; j += 1
        case ClassNumeric => lastLetter = false; j += 1
        case ClassExtend =>
          // A mark run trailing the anchor is properly attached in text order, so the run
          // stays open past it, including onto mid punctuation on its far side.
          val k = skipExtends(cps, j, 1)
          if (k < 0) return false
          val b = sigmaClassOf(cps(k))
          val isContinuer = b == ClassALetterCased || b == ClassNumericCased ||
            b == ClassNumeric || b == ClassExtendCased || b == ClassALetter ||
            b == ClassSuppCased || b == ClassSuppLetter || b == ClassSuppMn ||
            b == ClassSuppNum || b == ClassDanda || b == ClassMidLetter ||
            b == ClassMidNum || b == ClassMidNumLet
          if (!isContinuer) return false
          j = k
        case ClassSuppCased | ClassExtendCased =>
          // Attaches to the current word, so the scan sees it (cased).
          return true
        case ClassSuppLetter | ClassSuppMn | ClassSuppNum =>
          // Attach to the current word and close it; never themselves cased, and nothing
          // beyond is reachable.
          return false
        case ClassDanda =>
          // The danda attaches only to a word part that ends in letters (reached after
          // digits the word is already closed) and continues only into a digit -- unless
          // that digit is itself cased (a Roman numeral), which resolves the scan.
          if (!lastLetter) return false
          if (j + 1 < cps.length && sigmaClassOf(cps(j + 1)) == ClassNumericCased) {
            return true
          } else if (j + 1 < cps.length && sigmaClassOf(cps(j + 1)) == ClassNumeric) {
            lastLetter = false
            j += 2
          } else {
            return false
          }
        case cls @ (ClassMidLetter | ClassMidNum | ClassMidNumLet) =>
          // `<mid-letter><let>` / `<mid-num><digit>` require a genuine letter/digit base
          // IMMEDIATELY after the punctuation -- unlike the backward scan, marks here are
          // never skipped past: a mark directly after the punctuation is attached to the
          // punctuation, not a base, so it blocks the bridge. (Format chars are already
          // filtered out, which is what lets "A<sigma>-<ZWJ>b" bridge exactly like
          // "A<sigma>-b".)
          val mwOk = cls == ClassMidLetter || cls == ClassMidNumLet
          val mnOk = cls == ClassMidNum || cls == ClassMidNumLet
          if (j + 1 >= cps.length) return false
          val b = sigmaClassOf(cps(j + 1))
          if ((lastLetter && mwOk && isLetterBase(b)) ||
            (!lastLetter && mnOk && isDigitBase(b))) {
            j += 1
          } else {
            return false
          }
        case _ => return false
      }
    }
    false
  }

  private lazy val lowerMap: java.util.HashMap[Integer, String] = {
    val m = new java.util.HashMap[Integer, String](generated.lowerCps.length * 2)
    var i = 0
    while (i < generated.lowerCps.length) {
      m.put(generated.lowerCps(i), generated.lowerRepls(i))
      i += 1
    }
    m
  }

  /**
   * Lowercase `s` exactly as the native matcher will, from the generated tables: per codepoint,
   * U+03A3 takes its contextual final/non-final form via the ported `isFinalCased` scan over the
   * format-filtered sequence; every other codepoint takes its table replacement (or itself).
   */
  private[comet] def mirrorLowercase(s: String): String = {
    val raw = s.codePoints().toArray
    var filtered: Array[Int] = null
    var filteredIdx: Array[Int] = null
    val sb = new java.lang.StringBuilder(s.length)
    var i = 0
    while (i < raw.length) {
      val cp = raw(i)
      if (cp == 0x03a3) {
        if (filtered == null) {
          val buf = new Array[Int](raw.length)
          filteredIdx = new Array[Int](raw.length)
          var n = 0
          var k = 0
          while (k < raw.length) {
            filteredIdx(k) = n
            if (sigmaClassOf(raw(k)) != ClassFormat) {
              buf(n) = raw(k)
              n += 1
            }
            k += 1
          }
          filtered = java.util.Arrays.copyOf(buf, n)
        }
        val fi = filteredIdx(i)
        val leadingFormat = sigmaClassOf(raw(0)) == ClassFormat
        val isFinal = scanBackFindsCased(filtered, fi, leadingFormat) &&
          !scanFwdFindsCased(filtered, fi)
        sb.append((if (isFinal) 0x03c2 else 0x03c3).toChar)
      } else {
        val repl = lowerMap.get(cp)
        if (repl != null) sb.append(repl) else sb.appendCodePoint(cp)
      }
      i += 1
    }
    sb.toString
  }
}
