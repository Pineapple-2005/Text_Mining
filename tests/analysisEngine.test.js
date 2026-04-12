import test from "node:test";
import assert from "node:assert/strict";

import {
  aggregatePassageEvidence,
  alignTokenSequences,
  analyzeByMode,
  canonicalToken,
  comparePassages,
  filteredNgramSet,
  findCandidatePassages,
  longestCommonTokenRun,
  makeNgrams,
  normalizeSentence,
  normalizeTokens,
  segmentText,
  stemToken,
  sentenceSplit,
  tokenSimilarity,
  tokenize,
} from "../analysisEngine.js";

const FIXTURES = {
  highPlagiarismPair: {
    a: "The river delta restoration plan protects wetland habitat and strengthens flood control. The monitoring team tracks salinity levels every week. Local agencies publish shoreline repair updates for residents.",
    b: "The river delta restoration plan protects wetland habitat and strengthens flood control. The monitoring team tracks salinity levels every week. Local agencies publish shoreline repair updates for residents.",
  },
  thematicButNotCopiedPair: {
    a: "The climate adaptation report recommends wetland recovery, flood barriers, and weekly water quality monitoring in the estuary.",
    b: "The coastal resilience brief focuses on marsh restoration, stronger storm defenses, and recurring checks on estuary conditions.",
  },
  highGeneralSimilarityPair: {
    a: "Machine learning teams need clean datasets, careful feature selection, model evaluation, and reproducible experiments.",
    b: "Reliable machine learning work depends on clean datasets, feature selection, model evaluation, and reproducible experiments.",
  },
  lowSimilarityPair: {
    a: "Basketball rotations changed after halftime and the team found open corner shots.",
    b: "Fermentation time, hydration, and oven temperature determine the texture of sourdough bread.",
  },
  punctuationAndCasePair: {
    a: "Data Pipelines Improve Accuracy! Shared Metrics Matter.",
    b: "data pipelines improve accuracy. shared metrics matter?",
  },
  nearExactSentencePair: {
    a: "Data pipelines improve forecasting accuracy. Shared metrics matter for reliable planning.",
    b: "data pipelines improve forecasting accuracy! shared metrics matter for reliable planning?",
  },
  sharedPassagePair: {
    a: "The audit found process drift in three regions before the team rewrote onboarding, reviewed approvals, and retrained every field coordinator for consistency.",
    b: "After interviews, the audit found process drift in three regions before the team rewrote onboarding, reviewed approvals, and retrained every field coordinator for consistency across offices.",
  },
  boilerplatePair: {
    a: "This policy is provided as is and subject to change at any time by the service team.",
    b: "This policy is provided as is and subject to change at any time by the support team.",
  },
  shortTextPair: {
    a: "alpha beta",
    b: "alpha beta",
  },
  emptyPair: {
    a: "",
    b: "",
  },
};

test("plagiarism mode scores copied text higher than thematic similarity", () => {
  const copied = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.highPlagiarismPair.a,
    docB: FIXTURES.highPlagiarismPair.b,
    modeLabel: "Cursed Echo Screening",
  });
  const thematic = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.thematicButNotCopiedPair.a,
    docB: FIXTURES.thematicButNotCopiedPair.b,
    modeLabel: "Cursed Echo Screening",
  });

  assert.ok(copied.pct > thematic.pct, `expected ${copied.pct} > ${thematic.pct}`);
  assert.ok(copied.modeSignals.some(([label]) => label === "Repeated phrases"));
  assert.ok(copied.modeSignals.some(([label]) => label === "Exact sentence matches"));
});

test("plagiarism mode detects sentence reuse in identical documents", () => {
  const result = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.highPlagiarismPair.a,
    docB: FIXTURES.highPlagiarismPair.b,
    modeLabel: "Cursed Echo Screening",
  });

  const sentenceReuse = result.modeSignals.find(([label]) => label === "Sentence reuse");
  const exactSentenceMatches = result.modeSignals.find(([label]) => label === "Exact sentence matches");

  assert.ok(result.pct >= 70, `expected a high plagiarism score, got ${result.pct}`);
  assert.ok(sentenceReuse, "expected sentence reuse signal");
  assert.ok(exactSentenceMatches, "expected exact sentence match signal");
  assert.notEqual(exactSentenceMatches[1], "0");
  assert.ok(result.suspiciousPassages.length > 0, "expected suspicious passages to be localized");
});

test("sentence normalization catches punctuation-only sentence reuse", () => {
  const result = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.nearExactSentencePair.a,
    docB: FIXTURES.nearExactSentencePair.b,
    modeLabel: "Cursed Echo Screening",
  });

  assert.equal(normalizeSentence("Shared Metrics Matter."), normalizeSentence("shared metrics matter?"));
  assert.equal(result.modeSignals.find(([label]) => label === "Exact sentence matches")[1], "2");
});

test("light synonym and stem normalization improve paraphrase alignment", () => {
  const source = tokenize("The audit found process drift in three regions before the team rewrote onboarding and retrained every field coordinator.");
  const submitted = tokenize("The audit found process drift across three regions before the team revised onboarding and retrained each field coordinator.");
  const alignment = alignTokenSequences(source, submitted);

  assert.equal(stemToken("revised"), "revis");
  assert.equal(canonicalToken("rewrote"), "revise");
  assert.equal(canonicalToken("each"), "each");
  assert.equal(tokenSimilarity("rewrote", "revised"), 0.9);
  assert.deepEqual(normalizeTokens(["rewrote", "every", "coordinators"]), ["revise", "each", "coordinator"]);
  assert.ok(alignment.score >= 0.8, `expected strong alignment, got ${alignment.score}`);
});

test("copied text exposes direct-reuse signals only in plagiarism mode", () => {
  const plagiarism = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.highPlagiarismPair.a,
    docB: FIXTURES.highPlagiarismPair.b,
    modeLabel: "Cursed Echo Screening",
  });
  const general = analyzeByMode({
    mode: "general",
    docA: FIXTURES.highPlagiarismPair.a,
    docB: FIXTURES.highPlagiarismPair.b,
    modeLabel: "Domain Clash Analysis",
  });

  assert.ok(
    plagiarism.modeSignals.some(([label]) => label === "Exact sentence matches"),
    "expected plagiarism mode to expose direct sentence reuse",
  );
  assert.ok(
    general.modeSignals.every(([label]) => label !== "Exact sentence matches"),
    "expected general mode to omit direct reuse signals",
  );
  assert.equal(general.modeSignals[0][0], "Shared top terms");
  assert.equal(general.ai.match_label, "Special Grade Resonance");
});

test("long shared passages are surfaced even without exact full-sentence matches", () => {
  const result = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.sharedPassagePair.a,
    docB: FIXTURES.sharedPassagePair.b,
    modeLabel: "Cursed Echo Screening",
  });

  const run = longestCommonTokenRun(tokenize(FIXTURES.sharedPassagePair.a), tokenize(FIXTURES.sharedPassagePair.b));
  assert.ok(run.length >= 10, `expected a substantial shared passage, got ${run.length}`);
  assert.ok(result.modeSignals.some(([label]) => label === "Longest shared passage"));
  assert.ok(result.suspiciousPassages.length > 0, "expected localized suspicious passages");
  assert.ok(result.suspiciousPassages[0].longestSharedRun >= 10);
});

test("near-copy paraphrases get elevated by sequence alignment", () => {
  const result = analyzeByMode({
    mode: "plagiarism",
    docA: "The audit found process drift in three regions before the team rewrote onboarding and retrained every field coordinator.",
    docB: "The audit found process drift across three regions before the team revised onboarding and retrained each field coordinator.",
    modeLabel: "Cursed Echo Screening",
  });

  const alignmentSignal = result.modeSignals.find(([label]) => label === "Sequence alignment");
  assert.ok(alignmentSignal, "expected sequence alignment signal");
  assert.ok(parseInt(alignmentSignal[1], 10) >= 75, `expected high alignment, got ${alignmentSignal[1]}`);
  assert.ok(["near-copy", "paraphrase-risk"].includes(result.suspiciousPassages[0].kind));
});

test("candidate passage finder returns localized suspicious pairs", () => {
  const candidates = findCandidatePassages(
    FIXTURES.sharedPassagePair.a,
    FIXTURES.sharedPassagePair.b,
  );

  assert.ok(candidates.length > 0, "expected at least one candidate passage");
  assert.ok(["exact", "near-copy", "paraphrase-risk"].includes(candidates[0].kind));
  assert.equal(typeof candidates[0].sourceText, "string");
  assert.equal(typeof candidates[0].submittedText, "string");
});

test("passage comparison classifies exact sentence matches", () => {
  const [segmentA] = segmentText(FIXTURES.highPlagiarismPair.a);
  const [segmentB] = segmentText(FIXTURES.highPlagiarismPair.b);
  const comparison = comparePassages(segmentA, segmentB);

  assert.equal(comparison.kind, "exact");
  assert.equal(comparison.exactMatch, true);
  assert.ok(comparison.score >= 0.8, `expected a very high passage score, got ${comparison.score}`);
});

test("passage evidence aggregation reports strongest local signal", () => {
  const candidates = findCandidatePassages(
    FIXTURES.highPlagiarismPair.a,
    FIXTURES.highPlagiarismPair.b,
  );
  const evidence = aggregatePassageEvidence(candidates, Math.min(
    tokenize(FIXTURES.highPlagiarismPair.a).length,
    tokenize(FIXTURES.highPlagiarismPair.b).length,
  ));

  assert.ok(evidence.strongestPassageScore > 0);
  assert.ok(evidence.suspiciousPassageCount >= 1);
  assert.ok(evidence.evidenceCoverage > 0);
});

test("general mode reports stable structure for similar technical documents", () => {
  const result = analyzeByMode({
    mode: "general",
    docA: FIXTURES.highGeneralSimilarityPair.a,
    docB: FIXTURES.highGeneralSimilarityPair.b,
    modeLabel: "Domain Clash Analysis",
  });

  assert.equal(result.modeSignals.length, 4);
  assert.equal(typeof result.ai.verdict, "string");
  assert.equal(typeof result.ai.recommendation, "string");
  assert.ok(result.pct >= 45, `expected at least moderate similarity, got ${result.pct}`);
});

test("tokenization ignores punctuation and case differences", () => {
  const result = analyzeByMode({
    mode: "general",
    docA: FIXTURES.punctuationAndCasePair.a,
    docB: FIXTURES.punctuationAndCasePair.b,
    modeLabel: "Domain Clash Analysis",
  });

  assert.deepEqual(tokenize(FIXTURES.punctuationAndCasePair.a), tokenize(FIXTURES.punctuationAndCasePair.b));
  assert.equal(result.pct, 100);
});

test("boilerplate-heavy phrases are filtered out of 3-gram overlap", () => {
  const grams = filteredNgramSet(tokenize(FIXTURES.boilerplatePair.a), 3);
  assert.ok(!grams.has("as is and"));
  assert.ok(!grams.has("and subject to"));
});

test("short texts do not break phrase overlap calculations", () => {
  const result = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.shortTextPair.a,
    docB: FIXTURES.shortTextPair.b,
    modeLabel: "Cursed Echo Screening",
  });

  assert.deepEqual(makeNgrams(tokenize(FIXTURES.shortTextPair.a), 3), []);
  assert.equal(result.modeSignals.find(([label]) => label === "Repeated phrases")[1], "0");
});

test("empty inputs return stable low scores without crashing", () => {
  const plagiarism = analyzeByMode({
    mode: "plagiarism",
    docA: FIXTURES.emptyPair.a,
    docB: FIXTURES.emptyPair.b,
    modeLabel: "Cursed Echo Screening",
  });
  const general = analyzeByMode({
    mode: "general",
    docA: FIXTURES.emptyPair.a,
    docB: FIXTURES.emptyPair.b,
    modeLabel: "Domain Clash Analysis",
  });

  assert.equal(plagiarism.pct, 0);
  assert.equal(general.pct, 0);
  assert.equal(sentenceSplit(FIXTURES.emptyPair.a).length, 0);
});

test("unrelated documents remain low-similarity in general mode", () => {
  const result = analyzeByMode({
    mode: "general",
    docA: FIXTURES.lowSimilarityPair.a,
    docB: FIXTURES.lowSimilarityPair.b,
    modeLabel: "Domain Clash Analysis",
  });

  assert.ok(result.pct < 45, `expected low similarity, got ${result.pct}`);
  assert.ok(result.onlyA.length > 0);
  assert.ok(result.onlyB.length > 0);
});
