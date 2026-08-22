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

package org.apache.comet.contrib.delta

import java.util.Locale

import org.scalatest.funsuite.AnyFunSuite

import org.apache.comet.serde.operator.JvmCaseTables

/**
 * Self-validating proof that `JvmCaseTables.mirrorLowercase` -- a line-for-line mirror of the
 * native `JvmCaseTables::lowercase` in `native/core/src/parquet/schema_adapter.rs`, run over the
 * tables `JvmCaseTables` generates from the RUNNING JVM -- reproduces this JVM's
 * `String.toLowerCase(Locale.ROOT)`. Because both the tables and the expectations come from the
 * same running JVM, this suite passes on ANY supported JDK by construction.
 *
 * Needs no `SparkSession`, so it extends `AnyFunSuite` directly rather than `CometDeltaTestBase`.
 */
class JvmLowercaseParitySuite extends AnyFunSuite {

  private def hex(s: String): String =
    s.codePoints().toArray.map(cp => f"U+$cp%04X").mkString(" ")

  private def assertMirrors(input: String): Option[String] = {
    val expected = input.toLowerCase(Locale.ROOT)
    val actual = JvmCaseTables.mirrorLowercase(input)
    if (expected == actual) None
    else Some(s"[${hex(input)}] jvm=[${hex(expected)}] mirror=[${hex(actual)}]")
  }

  test("generated tables are well-formed for the wire") {
    val t = JvmCaseTables.generated
    assert(t.lowerCps.length == t.lowerRepls.length, "lowercase arrays must stay index-aligned")
    assert(t.lowerCps.nonEmpty, "every Unicode version maps at least ASCII A-Z")
    // Strictly increasing codepoints (generation sweeps in order; native builds a map).
    t.lowerCps.sliding(2).foreach {
      case Array(a, b) => assert(a < b, f"lowercase codepoints out of order at U+$a%04X")
      case _ =>
    }
    t.lowerRepls.foreach(repl => assert(repl.nonEmpty, "empty lowercase replacement"))
    // Class ranges: (start, end, class) triples, sorted, disjoint, valid classes.
    assert(t.classRanges.length % 3 == 0, "class ranges must be (start, end, class) triples")
    var prevEnd = -1
    t.classRanges.grouped(3).foreach { case Array(start, end, cls) =>
      assert(start <= end && start > prevEnd, f"overlapping/unsorted range at U+$start%04X")
      assert(cls >= 1 && cls <= 15, s"class $cls outside the wire contract")
      assert(end <= 0x10ffff)
      prevEnd = end
    }
  }

  test("running JVM's toLowerCase(Locale.ROOT) is reproduced for every isolated codepoint") {
    // Proves the lowercase TABLE is exactly this JVM's non-identity set (nothing extra,
    // nothing missing); full 1,114,112-codepoint sweep, well under a second.
    val mismatches = (0 to 0x10ffff).view
      .filterNot(cp => cp >= 0xd800 && cp <= 0xdfff) // surrogate halves: not codepoints
      .flatMap(cp => assertMirrors(new String(Character.toChars(cp))))
      .take(20)
      .toList
    assert(
      mismatches.isEmpty,
      "the running JDK's toLowerCase(Locale.ROOT) diverges from the generated tables -- " +
        s"the JvmCaseTables generation is wrong for this JDK:\n${mismatches.mkString("\n")}")
  }

  test("running JVM's contextual sigma lowering is reproduced across the full codepoint sweep") {
    // Eight sigma contexts per codepoint X, exercising both scan directions of the ported
    // condition. Zero-mismatch calibrated on JDK 17, 21, and 25. Sequential (not `.par`): Scala
    // 2.13 moved parallel collections out of the standard library, breaking the spark-4.x
    // profiles; the sequential sweep already finishes in a few seconds.
    val mismatches = (0 to 0x10ffff)
      .filter(cp => !(cp >= 0xd800 && cp <= 0xdfff))
      .flatMap { cp =>
        val x = new String(Character.toChars(cp))
        Seq(
          s"A${x}Σ", // backward scan, cased letter beyond X
          s"AΣ$x", // forward scan, X terminal
          s"AΣ${x}B", // forward scan, cased letter beyond X
          s"A1${x}Σ", // backward scan, digit beyond X
          s"A${x}1Σ", // backward scan, digit between X and sigma
          s"${x}Σ", // X at string start
          s"${x}1Σ", // X at string start, digit before sigma
          s"A ${x}Σ" // X after a word boundary (space)
        ).flatMap(assertMirrors)
      }
      .seq
      .take(20)
      .toList
    assert(
      mismatches.isEmpty,
      "the running JDK's contextual sigma lowering diverges from the mirrored native " +
        s"algorithm:\n${mismatches.mkString("\n")}")
  }

  test("sigma corpus matches the running JVM exactly") {
    // Human-readable corpus pinning the interesting shapes by name.
    val corpus = Seq(
      "ΑΣ", // capital alpha + sigma: final
      "A1Σ", // digit keeps the word open: final
      "A_Σ", // underscore is mid-word: final
      "A-Σ", // dash is mid-word: final
      "A.Σ", // period is mid-word: final
      "A,Σ", // comma is mid-num only: non-final
      "A..Σ", // two mid-word marks in a row break the word: non-final
      "A1.Σ", // mid-word mark needs letters on both sides: non-final
      "A-1Σ", // dash before a digit breaks the word: non-final
      "A'Σ", // apostrophe is mid-word: final
      "AΣB", // cased letter after: non-final
      "Σ", // isolated: non-final
      "ΣA", // nothing cased before: non-final
      "ΣΣ", // sigma before sigma is cased: "σς"
      "AΣ", // simplest final
      "ABΣ",
      "A Σ", // space breaks the word: non-final
      "AΣ B",
      "A1Σ, B",
      "İΣ", // dotted capital I expands to two chars and is cased: final
      "AΣİ", // cased letter after: non-final
      "ΟΔΥΣΣΕΥΣ", // ΟΔΥΣΣΕΥΣ: only the last is final
      "ÁΣ", // precomposed accented letter: final
      "AΣ́", // combining acute after sigma is transparent: final
      "1Σ", // digit alone is not cased: non-final
      "Σ1",
      "A1Σ1", // trailing digit stays in the word: non-final... but nothing cased after
      "x'Σ",
      "A‍Σ", // zero-width joiner (Cf) is transparent: final
      "A­Σ", // soft hyphen is mid-word: final
      "3.Σ", // mid mark between digit and sigma: non-final
      "AアΣ", // katakana forms its own word: non-final
      "一Σ", // kanji forms its own word: non-final
      "A]Σ",
      "A[Σ",
      "Σ.",
      "AΣ.",
      "AΣ-",
      "AΣ_b", // mid-word joins a cased letter after: non-final
      "AΣ_",
      "AΣ'b",
      "AΣ.b",
      "AΣ1b",
      "AΣ1.",
      "a1ς", // already-lowercase controls
      "a1σ",
      "𐐀Σ", // leading Deseret capital: joined at string start, final
      "A𐐀Σ", // supplementary closes the preceding word: non-final
      "AΣ𐐀", // cased supplementary visible forward: non-final
      "𠀀Σ", // supplementary Han: not cased, non-final
      "AΣ𠀀", // and invisible forward: final
      "A।" + "1Σ", // danda chains into a number: final
      "AΣ।1B", // danda then number then cased letter: non-final
      "AΣ।B", // danda then letter: word closed, final
      "ͅΣ", // base-less cased mark: non-final
      "A ͅΣ", // cased mark attached to a space: non-final
      "1ͅΣ", // cased mark attached to a digit: final
      "A-ͅΣ", // cased mark attached to a dash: non-final
      "AΣͅ", // cased mark forward: non-final
      "A゙Σ", // combining kana voicing mark is transparent: final
      "KelvinΣ", // Kelvin sign is cased: final
      "col_a1Σ", // realistic column names
      "COL_A1Σ",
      "sales_2024.q1Σ",
      "AΣ-\u200Db", // sigma, dash, ZWJ, letter: the format filter bridges to the cased 'b'
      "AΣ-\u200D\u200Db", // and any number of pure-format riders
      "AΣ-\u0301b", // an Mn mark after the dash blocks the bridge: final
      "AΣ-\u200D\u0301b", // an Mn anywhere in the rider chain blocks it too
      "AΣ\u200D-b", // format rider trailing the sigma bridges onward
      "A-\u200DΣ", // and backward: "A-<ZWJ>Σ" behaves exactly like "A-Σ"
      "\u200D\uD801\uDC00Σ", // leading format defeats the supplementary text-start join
      "A\uD804\uDF74\u0345Σ", // anchored supplementary mark carries the cased mark
      "\uD804\uDF74\u0345Σ", // unanchored at text start: non-final
      "A\uD804\uDF74\u0345-Σ", // and across a mid-letter bridge when letter-anchored
      "A\uD835\uDFD3\u0345,1Σ", // supplementary digit carries the mark across mid-num
      "A\uD835\uDFD3\u0345-Σ", // but never across mid-letter
      "\uD801\uDC00।1Σ"
    ) // cased supplementary at text start chains through the danda
    val mismatches = corpus.flatMap(assertMirrors)
    assert(
      mismatches.isEmpty,
      s"corpus diverges from the running JDK:\n${mismatches.mkString("\n")}")
  }

  // One or more representatives per word-class `JvmCaseTables.classify` assigns, plus plain
  // word-boundary characters, shared by the pair sweep and the fuzz test below: cased/letterish/
  // digit/mid-word/mid-num/supplementary/danda/mark-cased codepoints, including the Unicode-14
  // scripts whose version-gated assignment first exposed the JDK 17 vs 21 divergence (U+A7C0,
  // U+1C89). A path-dependent DFA quirk that only shows up across multiple adjacent specials
  // would appear as a mismatch between two of these class shapes and the real JVM.
  private val classRepresentativeCodepoints: Array[Int] = Array(0x0041, 0x0061, 0x0391, 0x03c3,
    0x0130, 0x0131, 0x212a, 0xa7c0, 0x1c89, 0x2160, 0x2170, 0x0301, 0x3099, 0x0903, 0x20dd,
    0x0031, 0x0660, 0x00b2, 0x002d, 0x005f, 0x00ad, 0x2027, 0x002c, 0x066b, 0x002e, 0x0022,
    0x0027, 0x10570, 0x10400, 0x0964, 0x0965, 0x0345, 0x0020, 0x30a2, 0x4e00, 0x20000, 0x0021,
    0x00a0, 0x200d, 0x11374, 0x1d7ce, 0xe0049, 0x99992)

  // Exactly one representative per shipped word-break class, plus two plain boundary
  // characters, for the ordered-triple sweep: every 3-deep class sequence between the cased
  // anchor and the sigma is exercised in both scan directions.
  private val perClassRepresentatives: Array[Int] = Array(0x0041, // ClassALetterCased
    0x05d0, // ClassALetter (Hebrew alef: letter-base, not cased)
    0x0031, // ClassNumeric
    0x2160, // ClassNumericCased
    0x002d, // ClassMidLetter
    0x002c, // ClassMidNum
    0x002e, // ClassMidNumLet
    0x0301, // ClassExtend
    0x0345, // ClassExtendCased
    0x200d, // ClassFormat
    0x0964, // ClassDanda
    0x10400, // ClassSuppCased
    0x20000, // ClassSuppLetter
    0x11374, // ClassSuppMn
    0x1d7ce, // ClassSuppNum
    0x0020, // boundary (space)
    0x0021 // boundary (symbol)
  )

  test("ordered pairs of class-representative codepoints around a sigma match the running JVM") {
    val r = classRepresentativeCodepoints.map(cp => new String(Character.toChars(cp)))
    val templates: Array[(String, String) => String] = Array(
      (x, y) => s"A$x${y}Σ",
      (x, y) => s"AΣ$x$y",
      (x, y) => s"A${x}Σ$y",
      (x, y) => s"$x${y}Σ",
      (x, y) => s"Σ$x$y",
      (x, y) => s"A${x}1${y}Σ")
    val mismatches = scala.collection.mutable.ListBuffer.empty[String]
    var total = 0
    var i = 0
    while (i < r.length && mismatches.size < 20) {
      var j = 0
      while (j < r.length && mismatches.size < 20) {
        templates.foreach { tmpl =>
          total += 1
          assertMirrors(tmpl(r(i), r(j))).foreach(mismatches += _)
        }
        j += 1
      }
      i += 1
    }
    info(
      s"pair sweep: ${r.length} representatives x ${r.length} x ${templates.length} " +
        s"templates = $total strings tested")
    assert(total <= 100000, s"pair sweep exceeded the string-count budget: $total")
    assert(
      mismatches.isEmpty,
      s"multi-special ordered-pair sweep diverges from the running JDK " +
        s"($total strings tested):\n${mismatches.mkString("\n")}")
  }

  test("ordered triples of per-class codepoints around a sigma match the running JVM") {
    // Multi-rider chains: every ordered TRIPLE of class representatives in four templates
    // (trailing the sigma, preceding it, and sandwiched between the sigma and a cased
    // letter), plus every ordered pair sandwiched the same way. This is the shape family
    // where the pre-UAX#29 hand-rolled rider logic diverged from Java's BreakIterator
    // (e.g. sigma, '-', U+200D, letter).
    val r = perClassRepresentatives.map(cp => new String(Character.toChars(cp)))
    val mismatches = scala.collection.mutable.ListBuffer.empty[String]
    var total = 0
    var i = 0
    while (i < r.length && mismatches.size < 20) {
      var j = 0
      while (j < r.length && mismatches.size < 20) {
        total += 1
        assertMirrors(s"AΣ${r(i)}${r(j)}B").foreach(mismatches += _)
        var k = 0
        while (k < r.length && mismatches.size < 20) {
          total += 3
          assertMirrors(s"AΣ${r(i)}${r(j)}${r(k)}").foreach(mismatches += _)
          assertMirrors(s"A${r(i)}${r(j)}${r(k)}Σ").foreach(mismatches += _)
          assertMirrors(s"AΣ${r(i)}${r(j)}${r(k)}B").foreach(mismatches += _)
          k += 1
        }
        j += 1
      }
      i += 1
    }
    info(
      s"triple sweep: ${r.length} per-class representatives, 3 triple templates + 1 pair " +
        s"template = $total strings tested")
    assert(total <= 500000, s"triple sweep exceeded the string-count budget: $total")
    assert(
      mismatches.isEmpty,
      s"multi-rider ordered-triple sweep diverges from the running JDK " +
        s"($total strings tested):\n${mismatches.mkString("\n")}")
  }

  test("seeded fuzz: multi-special strings around a sigma match the running JVM") {
    // Fixed seed for determinism. Draws codepoints predominantly from the class-representative
    // pool above and otherwise uniformly from the full codepoint space, with exactly one sigma
    // placed at a random position in every generated string.
    val seed = 0x516d41ceL
    val rnd = new scala.util.Random(seed)
    val pool = classRepresentativeCodepoints
    def randomCodepoint(): Int = {
      if (rnd.nextBoolean()) {
        pool(rnd.nextInt(pool.length))
      } else {
        var cp = 0
        do {
          cp = rnd.nextInt(0x110000)
        } while (cp >= 0xd800 && cp <= 0xdfff)
        cp
      }
    }
    val fuzzCount = 40000
    val mismatches = scala.collection.mutable.ListBuffer.empty[String]
    var n = 0
    while (n < fuzzCount && mismatches.size < 20) {
      val len = 3 + rnd.nextInt(6) // 3..8 codepoints
      val sigmaAt = rnd.nextInt(len)
      val sb = new java.lang.StringBuilder()
      var k = 0
      while (k < len) {
        sb.appendCodePoint(if (k == sigmaAt) 0x03a3 else randomCodepoint())
        k += 1
      }
      assertMirrors(sb.toString).foreach(mismatches += _)
      n += 1
    }
    info(s"seeded fuzz (seed=0x${seed.toHexString}): $n strings tested")
    assert(
      mismatches.isEmpty,
      s"seeded multi-special fuzz diverges from the running JDK " +
        s"($n strings, seed=0x${seed.toHexString}):\n${mismatches.mkString("\n")}")
  }
}
