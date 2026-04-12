export const tokenize = (text) =>
  text
    .toLowerCase()
    .replaceAll(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 0);

const tf = (tokens) => {
  const frequencies = {};

  tokens.forEach((token) => {
    frequencies[token] = (frequencies[token] || 0) + 1;
  });

  const maxFrequency = Math.max(...Object.values(frequencies), 1);

  Object.keys(frequencies).forEach((token) => {
    frequencies[token] /= maxFrequency;
  });

  return frequencies;
};

export const tfidf = (docs) => {
  const tokenizedDocs = docs.map(tokenize);
  const docCount = docs.length;
  const inverseDocFrequency = {};

  new Set(tokenizedDocs.flat()).forEach((word) => {
    const docFrequency = tokenizedDocs.filter((doc) => doc.includes(word)).length;
    inverseDocFrequency[word] = Math.log((docCount + 1) / (docFrequency + 1)) + 1;
  });

  return tokenizedDocs.map((tokens) => {
    const frequencies = tf(tokens);
    const vector = {};

    Object.keys(frequencies).forEach((token) => {
      vector[token] = frequencies[token] * (inverseDocFrequency[token] || 1);
    });

    return vector;
  });
};

export const cosine = (a, b) => {
  const terms = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  terms.forEach((term) => {
    const x = a[term] || 0;
    const y = b[term] || 0;
    dot += x * y;
    magnitudeA += x * x;
    magnitudeB += y * y;
  });

  return magnitudeA && magnitudeB ? dot / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB)) : 0;
};

export const topTerms = (vector, limit = 10) =>
  Object.entries(vector)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([word]) => word);

const unique = (arr) => [...new Set(arr)];

export function scoreLabelDetails(pct) {
  if (pct >= 70) return { t: "Special Grade Resonance", c: "var(--grade-special)" };
  if (pct >= 45) return { t: "Grade 1 Resonance", c: "var(--grade-mid)" };
  return { t: "Grade 4 Resonance", c: "var(--grade-low)" };
}

export function buildLocalAnalysis({ pct, overlap, onlyA, onlyB, modeLabel, labelA, labelB }) {
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
    ? `${labelA} channels ${onlyALead || "distinct terminology"}, while ${labelB} channels ${onlyBLead || "distinct terminology"}. The gap suggests different focus zones.`
    : `No major term-level divergence was detected between ${labelA} and ${labelB}.`;

  const recommendationText = pct >= 70
    ? `The ritual is stable for ${modeLabel.toLowerCase()}. Keep the structure and terminology aligned, then run a final review pass for phrasing consistency.`
    : `For ${modeLabel.toLowerCase()}, align key terms and core concepts more tightly. Rework sections so both texts express the same intent and technical focus.`;

  return {
    verdict: `${matchLabel}: the cursed resonance between ${labelA} and ${labelB} is ${resonanceState}.`,
    match_label: matchLabel,
    strength: sharedText,
    gap: gapText,
    recommendation: recommendationText,
    confidence,
  };
}

const buildPairResult = ({ docA, docB, vectorA, vectorB, modeLabel }) => {
  const topA = topTerms(vectorA, 10);
  const topB = topTerms(vectorB, 10);
  const termsA = new Set(topA);
  const termsB = new Set(topB);
  const overlap = unique(topA.filter((word) => termsB.has(word)));
  const onlyA = unique(topA.filter((word) => !termsB.has(word)));
  const onlyB = unique(topB.filter((word) => !termsA.has(word)));
  const score = cosine(vectorA, vectorB);
  const pct = Math.round(score * 100);

  return {
    id: `${docA.id}-${docB.id}`,
    docA,
    docB,
    score,
    pct,
    topA,
    topB,
    overlap,
    onlyA,
    onlyB,
    ai: buildLocalAnalysis({
      pct,
      overlap,
      onlyA,
      onlyB,
      modeLabel,
      labelA: docA.label,
      labelB: docB.label,
    }),
  };
};

export function analyzeDocuments(documents, modeLabel, options = {}) {
  const { scope = "pairwise", anchorId } = options;
  const activeDocuments = documents
    .map((doc, index) => ({ ...doc, order: index }))
    .filter((doc) => doc.text.trim());

  if (activeDocuments.length < 2) {
    throw new Error("At least two documents with text are required before starting the ritual.");
  }

  const vectors = tfidf(activeDocuments.map((doc) => doc.text));
  const tokenCounts = activeDocuments.map((doc) => tokenize(doc.text).length);
  const vocabulary = new Set(activeDocuments.flatMap((doc) => tokenize(doc.text))).size;
  const enrichedDocuments = activeDocuments.map((doc, index) => ({
    ...doc,
    vector: vectors[index],
    topTerms: topTerms(vectors[index], 10),
    tokenCount: tokenCounts[index],
  }));
  const pairResults = [];

  if (scope === "one-to-many") {
    const anchorDocument =
      enrichedDocuments.find((doc) => doc.id === anchorId) ||
      enrichedDocuments[0];

    enrichedDocuments
      .filter((doc) => doc.id !== anchorDocument.id)
      .forEach((doc) => {
        pairResults.push(
          buildPairResult({
            docA: anchorDocument,
            docB: doc,
            vectorA: anchorDocument.vector,
            vectorB: doc.vector,
            modeLabel,
          })
        );
      });
  } else {
    for (let i = 0; i < enrichedDocuments.length; i += 1) {
      for (let j = i + 1; j < enrichedDocuments.length; j += 1) {
        pairResults.push(
          buildPairResult({
            docA: enrichedDocuments[i],
            docB: enrichedDocuments[j],
            vectorA: enrichedDocuments[i].vector,
            vectorB: enrichedDocuments[j].vector,
            modeLabel,
          })
        );
      }
    }
  }

  pairResults.sort((a, b) => b.score - a.score);

  const averageScore = pairResults.reduce((sum, pair) => sum + pair.score, 0) / pairResults.length;
  const strongestPair = pairResults[0];
  const weakestPair = pairResults[pairResults.length - 1];
  const anchorDocument =
    scope === "one-to-many"
      ? enrichedDocuments.find((doc) => doc.id === (anchorId || enrichedDocuments[0]?.id)) || enrichedDocuments[0]
      : null;

  return {
    documents: enrichedDocuments,
    pairResults,
    summary: {
      scope,
      docCount: activeDocuments.length,
      pairCount: pairResults.length,
      avgPct: Math.round(averageScore * 100),
      vocab: vocabulary,
      strongestPair,
      weakestPair,
      anchorDocument,
    },
  };
}
