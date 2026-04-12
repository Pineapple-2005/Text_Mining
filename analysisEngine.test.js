import test from "node:test";
import assert from "node:assert/strict";

import { analyzeDocuments, cosine, tfidf, tokenize } from "./analysisEngine.js";

test("tokenize normalizes casing and punctuation", () => {
  assert.deepEqual(tokenize("Alpha, beta! ALPHA?"), ["alpha", "beta", "alpha"]);
});

test("cosine similarity remains high for overlapping pairwise documents", () => {
  const [a, b] = tfidf([
    "python automation for reports and analytics",
    "python analytics workflow for incident reports",
  ]);

  assert.ok(cosine(a, b) > 0.5);
});

test("analyzeDocuments builds every document pair and sorts strongest match first", () => {
  const result = analyzeDocuments(
    [
      { id: "doc-1", label: "Doc 1", text: "curse energy anomaly report urban stress" },
      { id: "doc-2", label: "Doc 2", text: "urban stress anomaly report curse pattern" },
      { id: "doc-3", label: "Doc 3", text: "botany greenhouse watering sunlight leaves" },
    ],
    "Research"
  );

  assert.equal(result.summary.docCount, 3);
  assert.equal(result.summary.pairCount, 3);
  assert.equal(result.pairResults[0].docA.id, "doc-1");
  assert.equal(result.pairResults[0].docB.id, "doc-2");
  assert.equal(result.summary.strongestPair.id, "doc-1-doc-2");
  assert.ok(result.summary.weakestPair.pct <= result.summary.strongestPair.pct);
});

test("analyzeDocuments rejects analysis with fewer than two populated documents", () => {
  assert.throws(
    () =>
      analyzeDocuments([{ id: "solo", label: "Solo", text: "only one document" }], "General"),
    /At least two documents/
  );
});

test("analyzeDocuments supports one-to-many comparisons for a chosen anchor document", () => {
  const result = analyzeDocuments(
    [
      { id: "source", label: "Source", text: "domain expansion cursed energy sequence intent" },
      { id: "match-1", label: "Match 1", text: "cursed energy intent sequence domain control" },
      { id: "match-2", label: "Match 2", text: "botany leaves sunlight roots watering greenhouse" },
      { id: "off-anchor", label: "Off Anchor", text: "sequence intent control cursed energy" },
    ],
    "Screening",
    { scope: "one-to-many", anchorId: "source" }
  );

  assert.equal(result.summary.scope, "one-to-many");
  assert.equal(result.summary.anchorDocument.id, "source");
  assert.equal(result.summary.pairCount, 3);
  assert.ok(result.pairResults.every((pair) => pair.docA.id === "source" || pair.docB.id === "source"));
});
