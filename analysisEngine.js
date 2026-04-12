const tokenize = (t) =>
  t
    .toLowerCase()
    .replaceAll(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 0);

const tf = (tokens) => {
  const f = {};
  tokens.forEach((t) => {
    f[t] = (f[t] || 0) + 1;
  });
  const mx = Math.max(...Object.values(f), 1);
  Object.keys(f).forEach((k) => {
    f[k] /= mx;
  });
  return f;
};

const tfidf = (docs) => {
  const tok = docs.map(tokenize);
  const N = docs.length;
  const idf = {};

  new Set(tok.flat()).forEach((w) => {
    const df = tok.filter((d) => d.includes(w)).length;

    if (N === 2) {
      idf[w] = df === 2 ? 1.0 : 1.2;
    } else {
      idf[w] = Math.log((N + 1) / (df + 1)) + 1;
    }
  });

  return tok.map((tokens) => {
    const f = tf(tokens);
    const v = {};
    Object.keys(f).forEach((t) => {
      v[t] = f[t] * (idf[t] || 1);
    });
    return v;
  });
};

const cosine = (a, b) => {
  const terms = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0;
  let mA = 0;
  let mB = 0;

  terms.forEach((t) => {
    const x = a[t] || 0;
    const y = b[t] || 0;
    dot += x * y;
    mA += x * x;
    mB += y * y;
  });

  return mA && mB ? dot / (Math.sqrt(mA) * Math.sqrt(mB)) : 0;
};

const top = (v, n = 10) =>
  Object.entries(v)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([w]) => w);

const unique = (arr) => [...new Set(arr)];

const normalizePct = (value) => Math.max(0, Math.min(100, Math.round(value)));

const clip01 = (value) => Math.max(0, Math.min(1, value));

const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
  "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
]);

const SYNONYM_GROUPS = [
  ["across", "in"],
  ["each", "every"],
  ["revise", "rework", "rewrite", "rewrote", "revised"],
  ["student", "learner", "pupil", "students", "learners", "pupils"],
  ["test", "exam"],
  ["comment", "comments", "feedback"],
];

const SYNONYM_MAP = new Map(
  SYNONYM_GROUPS.flatMap((group) => group.map((word) => [word, group[0]])),
);

const normalizeSentence = (text) =>
  text
    .toLowerCase()
    .replaceAll(/[^a-z0-9\s]/g, " ")
    .replaceAll(/\s+/g, " ")
    .trim();

const stemToken = (token) => {
  if (token.length <= 4) return token;
  if (token.endsWith("ing") && token.length > 6) return token.slice(0, -3);
  if (token.endsWith("ed") && token.length > 5) return token.slice(0, -2);
  if (token.endsWith("es") && token.length > 5) return token.slice(0, -2);
  if (token.endsWith("s") && token.length > 4) return token.slice(0, -1);
  return token;
};

const canonicalToken = (token) => SYNONYM_MAP.get(stemToken(token)) || SYNONYM_MAP.get(token) || stemToken(token);

const normalizeTokens = (tokens) => tokens.map(canonicalToken);

const sentenceSplit = (text) =>
  text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

const makeNgrams = (tokens, size = 3) => {
  if (tokens.length < size) return [];
  const grams = [];
  for (let i = 0; i <= tokens.length - size; i += 1) {
    grams.push(tokens.slice(i, i + size).join(" "));
  }
  return grams;
};

const ratio = (numerator, denominator) => (denominator ? numerator / denominator : 0);

const hasEnoughContent = (gram) => {
  const parts = gram.split(" ");
  const contentWords = parts.filter((part) => part.length >= 4 && !STOPWORDS.has(part));
  return contentWords.length >= 2;
};

const filteredNgramSet = (tokens, size = 3) =>
  new Set(makeNgrams(tokens, size).filter(hasEnoughContent));

const weightedTokenScore = (token) => {
  if (STOPWORDS.has(token)) return 0.2;
  if (token.length >= 8) return 1.25;
  if (token.length >= 5) return 1;
  return 0.7;
};

const segmentText = (text) =>
  sentenceSplit(text).map((raw, index) => ({
    id: index,
    raw,
    normalized: normalizeSentence(raw),
    tokens: tokenize(raw),
  }));

const ngramOverlapScore = (tokensA, tokensB, size = 3) => {
  const ngramsA = filteredNgramSet(tokensA, size);
  const ngramsB = filteredNgramSet(tokensB, size);
  const overlap = [...ngramsA].filter((gram) => ngramsB.has(gram));

  return {
    overlap,
    score: ratio(overlap.length, Math.max(Math.min(ngramsA.size, ngramsB.size), 1)),
  };
};

function longestCommonTokenRun(tokensA, tokensB) {
  if (!tokensA.length || !tokensB.length) return { length: 0, text: "" };

  const dp = Array(tokensB.length + 1).fill(0);
  let bestLength = 0;
  let bestEndIndex = -1;

  for (let i = 1; i <= tokensA.length; i += 1) {
    for (let j = tokensB.length; j >= 1; j -= 1) {
      if (tokensA[i - 1] === tokensB[j - 1]) {
        dp[j] = dp[j - 1] + 1;
        if (dp[j] > bestLength) {
          bestLength = dp[j];
          bestEndIndex = i - 1;
        }
      } else {
        dp[j] = 0;
      }
    }
  }

  if (bestLength === 0) return { length: 0, text: "" };

  return {
    length: bestLength,
    text: tokensA.slice(bestEndIndex - bestLength + 1, bestEndIndex + 1).join(" "),
  };
}

function tokenSimilarity(tokenA, tokenB) {
  if (tokenA === tokenB) return 1;
  const canonA = canonicalToken(tokenA);
  const canonB = canonicalToken(tokenB);
  if (canonA === canonB) return 0.9;
  if (tokenA.length >= 5 && tokenB.length >= 5 && (tokenA.startsWith(tokenB.slice(0, 4)) || tokenB.startsWith(tokenA.slice(0, 4)))) {
    return 0.7;
  }
  return 0;
}

function alignTokenSequences(tokensA, tokensB) {
  if (!tokensA.length || !tokensB.length) {
    return { score: 0, matchedWeight: 0, maxWeight: Math.max(tokensA.length, tokensB.length, 1), substitutions: 0 };
  }

  const dp = Array.from({ length: tokensA.length + 1 }, () => Array(tokensB.length + 1).fill(0));

  for (let i = 1; i <= tokensA.length; i += 1) {
    for (let j = 1; j <= tokensB.length; j += 1) {
      const sim = tokenSimilarity(tokensA[i - 1], tokensB[j - 1]) * weightedTokenScore(tokensA[i - 1]);
      dp[i][j] = Math.max(
        dp[i - 1][j],
        dp[i][j - 1],
        dp[i - 1][j - 1] + sim,
      );
    }
  }

  let i = tokensA.length;
  let j = tokensB.length;
  let substitutions = 0;
  while (i > 0 && j > 0) {
    const sim = tokenSimilarity(tokensA[i - 1], tokensB[j - 1]) * weightedTokenScore(tokensA[i - 1]);
    if (Math.abs(dp[i][j] - (dp[i - 1][j - 1] + sim)) < 0.0001 && sim > 0) {
      if (tokensA[i - 1] !== tokensB[j - 1]) substitutions += 1;
      i -= 1;
      j -= 1;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      i -= 1;
    } else {
      j -= 1;
    }
  }

  const maxWeight = Math.max(
    tokensA.reduce((sum, token) => sum + weightedTokenScore(token), 0),
    tokensB.reduce((sum, token) => sum + weightedTokenScore(token), 0),
    1,
  );

  return {
    score: clip01(dp[tokensA.length][tokensB.length] / maxWeight),
    matchedWeight: dp[tokensA.length][tokensB.length],
    maxWeight,
    substitutions,
  };
}

function comparePassages(segmentA, segmentB) {
  const normalizedA = normalizeTokens(segmentA.tokens);
  const normalizedB = normalizeTokens(segmentB.tokens);
  const semanticScore = cosine(tf(normalizedA), tf(normalizedB));
  const ngram = ngramOverlapScore(normalizedA, normalizedB, 3);
  const longestRun = longestCommonTokenRun(normalizedA, normalizedB);
  const alignment = alignTokenSequences(segmentA.tokens, segmentB.tokens);
  const exactMatch = segmentA.normalized.length > 0 && segmentA.normalized === segmentB.normalized;
  const passageDensity = ratio(longestRun.length, Math.max(Math.min(segmentA.tokens.length, segmentB.tokens.length), 1));

  const score = clip01(
    semanticScore * 0.1
      + ngram.score * 0.2
      + alignment.score * 0.35
      + (exactMatch ? 0.2 : 0)
      + passageDensity * 0.15,
  );

  let kind = "thematic";
  if (exactMatch) {
    kind = "exact";
  } else if (alignment.score >= 0.78 || longestRun.length >= 8 || score >= 0.7) {
    kind = "near-copy";
  } else if (score >= 0.45) {
    kind = "paraphrase-risk";
  }

  return {
    sourceId: segmentA.id,
    submittedId: segmentB.id,
    sourceText: segmentA.raw,
    submittedText: segmentB.raw,
    score,
    kind,
    exactMatch,
    repeatedPhrases: ngram.overlap.slice(0, 5),
    longestSharedRun: longestRun.length,
    longestSharedText: longestRun.text,
    alignmentScore: alignment.score,
    substitutions: alignment.substitutions,
    semanticScore,
  };
}

function findCandidatePassages(docA, docB) {
  const segmentsA = segmentText(docA);
  const segmentsB = segmentText(docB);
  const candidates = [];

  segmentsA.forEach((segmentA) => {
    segmentsB.forEach((segmentB) => {
      const candidate = comparePassages(segmentA, segmentB);
      if (candidate.score >= 0.45 || candidate.exactMatch || candidate.longestSharedRun >= 8) {
        candidates.push(candidate);
      }
    });
  });

  return candidates.sort((a, b) => b.score - a.score).slice(0, 5);
}

function aggregatePassageEvidence(candidates, tokenFloor) {
  if (!candidates.length) {
    return {
      strongestPassageScore: 0,
      suspiciousPassageCount: 0,
      evidenceCoverage: 0,
    };
  }

  const strongestPassageScore = candidates[0].score;
  const suspiciousPassageCount = candidates.filter((candidate) => candidate.score >= 0.6).length;
  const coveredTokens = candidates.reduce((sum, candidate) => sum + candidate.longestSharedRun, 0);

  return {
    strongestPassageScore,
    suspiciousPassageCount,
    evidenceCoverage: ratio(coveredTokens, Math.max(tokenFloor, 1)),
  };
}

function summarizeSignal(strength, pct) {
  if (pct >= 75) return { strength, confidence: "high" };
  if (pct >= 45) return { strength: `${strength} with some variance`, confidence: "medium" };
  return { strength: `limited ${strength}`, confidence: "low" };
}

function scoreLabelDetails(pct) {
  if (pct >= 70) return { t: "Special Grade Resonance", c: "var(--grade-special)" };
  if (pct >= 45) return { t: "Grade 1 Resonance", c: "var(--grade-mid)" };
  return { t: "Grade 4 Resonance", c: "var(--grade-low)" };
}

function buildGenericAnalysis({ pct, overlap, onlyA, onlyB, modeLabel }) {
  const matchLabel = scoreLabelDetails(pct).t;
  const overlapLead = overlap.slice(0, 5).join(", ");
  const onlyALead = onlyA.slice(0, 4).join(", ");
  const onlyBLead = onlyB.slice(0, 4).join(", ");

  let resonanceState = "low and unstable";
  let confidence = "low";
  if (pct >= 70) {
    resonanceState = "high and coherent";
    confidence = "high";
  } else if (pct >= 45) {
    resonanceState = "partial with notable variance";
    confidence = "medium";
  }

  const sharedText = overlap.length
    ? `Shared cursed signatures include ${overlapLead}. These recurring terms strongly influence the resonance score.`
    : "Very little direct cursed vocabulary overlaps, so this comparison relies more on broad thematic intent than exact term echoes.";

  const gapText = onlyA.length || onlyB.length
    ? `Document A channels ${onlyALead || "distinct terminology"}, while Document B channels ${onlyBLead || "distinct terminology"}. The gap suggests different focus zones.`
    : "No major term-level divergence was detected between the two documents.";

  const recommendationText = pct >= 70
    ? `The ritual is stable for ${modeLabel.toLowerCase()}. Keep the structure and terminology aligned, then run a final review pass for phrasing consistency.`
    : `For ${modeLabel.toLowerCase()}, align key terms and core concepts more tightly. Rework sections so both texts express the same intent and technical focus.`;

  return {
    verdict: `${matchLabel}: the cursed resonance between documents is ${resonanceState}.`,
    match_label: matchLabel,
    strength: sharedText,
    gap: gapText,
    recommendation: recommendationText,
    confidence,
  };
}

function analyzePlagiarismMode({ semanticScore, docA, docB }) {
  const tokensA = tokenize(docA);
  const tokensB = tokenize(docB);
  const normalizedTokensA = normalizeTokens(tokensA);
  const normalizedTokensB = normalizeTokens(tokensB);
  const ngram = ngramOverlapScore(normalizedTokensA, normalizedTokensB, 3);
  const repeatedPhrases = ngram.overlap;
  const phraseOverlap = ngram.score;
  const alignment = alignTokenSequences(tokensA, tokensB);

  const sentencesA = sentenceSplit(docA).map(normalizeSentence).filter(Boolean);
  const sentencesB = sentenceSplit(docB).map(normalizeSentence).filter(Boolean);
  const sentenceSetB = new Set(sentencesB);
  const exactSentenceOverlap = sentencesA.filter((sentence) => sentenceSetB.has(sentence)).length;
  const sentenceOverlap = ratio(exactSentenceOverlap, Math.max(Math.min(sentencesA.length, sentencesB.length), 1));

  const longestRun = longestCommonTokenRun(tokensA, tokensB);
  const passageOverlap = ratio(longestRun.length, Math.max(Math.min(tokensA.length, tokensB.length), 1));
  const candidatePassages = findCandidatePassages(docA, docB);
  const passageEvidence = aggregatePassageEvidence(candidatePassages, Math.min(tokensA.length, tokensB.length));

  const suspicion = clip01(
    semanticScore * 0.1
      + phraseOverlap * 0.25
      + sentenceOverlap * 0.15
      + passageOverlap * 0.2
      + alignment.score * 0.15
      + passageEvidence.strongestPassageScore * 0.2
      + passageEvidence.evidenceCoverage * 0.05,
  );
  const pct = normalizePct(suspicion * 100);
  const matchLabel = scoreLabelDetails(pct).t;
  const { strength, confidence } = summarizeSignal("echo risk", pct);

  return {
    pct,
    score: suspicion,
    overlap: repeatedPhrases.slice(0, 10),
    onlyA: unique(tokensA.filter((token) => !tokensB.includes(token))).slice(0, 10),
    onlyB: unique(tokensB.filter((token) => !tokensA.includes(token))).slice(0, 10),
    suspiciousPassages: candidatePassages,
    ai: {
      verdict: `${matchLabel}: ${strength} between the source and submitted scroll.`,
      match_label: matchLabel,
      strength: candidatePassages.length
        ? `The strongest flagged passage is classified as ${candidatePassages[0].kind} with ${(candidatePassages[0].score * 100).toFixed(0)}% passage confidence.`
        : repeatedPhrases.length
        ? `Repeated phrase fragments include ${repeatedPhrases.slice(0, 5).join(", ")}. Shared phrasing carries more weight here than general topic similarity.`
        : "Little direct phrase reuse was detected, so the screening result is driven more by general resemblance than repeated wording.",
      gap: exactSentenceOverlap
        ? `${exactSentenceOverlap} sentence-level match${exactSentenceOverlap === 1 ? "" : "es"} were detected after normalization, increasing the likelihood of direct reuse.`
        : candidatePassages.length
          ? `${candidatePassages.length} suspicious passage pair${candidatePassages.length === 1 ? "" : "s"} were localized for manual review.`
        : longestRun.length >= 6
          ? `No full sentence matches were found, but a contiguous shared passage of ${longestRun.length} tokens suggests close structural reuse.`
          : "No exact sentence matches were found, which lowers the chance of straight copy-paste reuse.",
      recommendation: pct >= 70
        ? "Review the shared passages manually. The overlap is strong enough to justify a closer plagiarism check with source citations."
        : "This looks more like thematic similarity than direct copying, but any shared phrases should still be reviewed in context.",
      confidence,
    },
    modeSignals: [
      ["Phrase overlap", `${normalizePct(phraseOverlap * 100)}%`],
      ["Sentence reuse", `${normalizePct(sentenceOverlap * 100)}%`],
      ["Sequence alignment", `${normalizePct(alignment.score * 100)}%`],
      ["Longest shared passage", `${longestRun.length.toLocaleString()} tokens`],
      ["Suspicious passages", candidatePassages.length.toLocaleString()],
      ["Strongest passage", `${normalizePct(passageEvidence.strongestPassageScore * 100)}%`],
      ["Repeated phrases", repeatedPhrases.length.toLocaleString()],
      ["Exact sentence matches", exactSentenceOverlap.toLocaleString()],
    ],
  };
}

function analyzeGeneralMode({ pct, overlap, onlyA, onlyB, modeLabel }) {
  return {
    pct,
    score: pct / 100,
    ai: buildGenericAnalysis({ pct, overlap, onlyA, onlyB, modeLabel }),
    modeSignals: [
      ["Shared top terms", overlap.length.toLocaleString()],
      ["Document A unique terms", onlyA.length.toLocaleString()],
      ["Document B unique terms", onlyB.length.toLocaleString()],
      ["Similarity score", `${pct}%`],
    ],
  };
}

function analyzeByMode({ mode, docA, docB, modeLabel }) {
  const [vA, vB] = tfidf([docA, docB]);
  const semanticScore = cosine(vA, vB);
  const tA = top(vA, 10);
  const tB = top(vB, 10);
  const sA = new Set(tA);
  const sB = new Set(tB);
  const overlap = unique(tA.filter((word) => sB.has(word)));
  const onlyA = unique(tA.filter((word) => !sB.has(word)));
  const onlyB = unique(tB.filter((word) => !sA.has(word)));
  const basePct = normalizePct(semanticScore * 100);

  const commonPayload = {
    score: semanticScore,
    pct: basePct,
    topA: tA,
    topB: tB,
    overlap,
    onlyA,
    onlyB,
    tokA: tokenize(docA).length,
    tokB: tokenize(docB).length,
    vocab: new Set([...tokenize(docA), ...tokenize(docB)]).size,
  };

  let modeResult;
  switch (mode) {
    case "plagiarism":
      modeResult = analyzePlagiarismMode({ semanticScore, docA, docB });
      break;
    case "general":
    default:
      modeResult = analyzeGeneralMode({ pct: basePct, overlap, onlyA, onlyB, modeLabel });
      break;
  }

  return {
    ...commonPayload,
    ...modeResult,
  };
}

export {
  analyzeByMode,
  analyzeGeneralMode,
  analyzePlagiarismMode,
  aggregatePassageEvidence,
  comparePassages,
  cosine,
  filteredNgramSet,
  findCandidatePassages,
  alignTokenSequences,
  canonicalToken,
  longestCommonTokenRun,
  makeNgrams,
  normalizeSentence,
  normalizePct,
  normalizeTokens,
  segmentText,
  stemToken,
  sentenceSplit,
  tfidf,
  tokenSimilarity,
  tokenize,
};
