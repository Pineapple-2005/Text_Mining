# Smart Document Analyzer

Smart Document Analyzer is a browser-based React app for comparing two documents with a shared TF-IDF and cosine-similarity core plus a stronger passage-based suspicious-reuse screening mode.

## What It Does

- Compares two text inputs and calculates a similarity score.
- Extracts text from `.txt`, `.md`, `.pdf`, and `.docx` files.
- Uses local analysis summaries to explain overlap, differences, and recommendations.
- Provides two focused modes: suspicious-reuse screening and TF-IDF comparison.

## Analysis Modes

### Cursed Echo Screening

Use this mode for `source text vs submitted text`.

It emphasizes:
- repeated 3-word phrases
- normalized sentence reuse
- sequence alignment for lightly edited paraphrases
- longest shared passages
- localized suspicious passage pairs
- general similarity as a supporting signal

This mode is meant to help flag paraphrasing or suspicious reuse for manual review.

### Domain Clash Analysis

Use this mode for `general document vs general document`.

It emphasizes:
- TF-IDF similarity
- cosine similarity
- shared top terms
- unique terms in each document

## How It Works

1. Each document is tokenized and normalized.
2. TF-IDF vectors are built for both documents.
3. Cosine similarity provides the shared semantic baseline.
4. `Cursed Echo Screening` adds repeated-phrase, sentence-normalization, sequence-alignment, and passage-localization checks.
5. `Domain Clash Analysis` reports the baseline TF-IDF and cosine comparison.
6. The app summarizes shared signals, divergences, and practical guidance.

## Scoring Summary

- `Cursed Echo Screening` blends semantic similarity with repeated phrase overlap, normalized sentence reuse, sequence alignment, longest shared passage evidence, and localized suspicious passage review.
- `Domain Clash Analysis` uses the baseline TF-IDF plus cosine-similarity comparison.

## Engine Overview

The shared engine lives in `analysisEngine.js`.

High-level flow:

1. `tokenize()` normalizes raw text into lowercase word tokens.
2. `tfidf()` and `cosine()` generate the shared semantic baseline for both modes.
3. `analyzeByMode()` builds common comparison data, then routes to either:
   - `analyzeGeneralMode()` for broad document similarity
   - `analyzePlagiarismMode()` for suspicious-reuse analysis
4. `analyzePlagiarismMode()` combines:
   - filtered 3-gram overlap
   - normalized sentence reuse
   - sequence alignment for light paraphrases
   - longest shared contiguous token run
   - localized suspicious passage comparison
5. The UI renders the score, explanation, mode signals, and suspicious passage review panel.

## Engine Flow

### Cursed Echo Screening

Input:
- `Source Scroll`
- `Submitted Scroll`

Process:
1. Tokenize both texts.
2. Build TF-IDF vectors.
3. Compute cosine similarity for the baseline similarity score.
4. Generate 3-word phrases from both texts.
5. Check which phrase fragments appear in both documents.
6. Split both texts into sentences.
7. Normalize sentence punctuation and casing before checking sentence reuse.
8. Normalize light token variations with stemming and synonym-style mapping.
9. Align token sequences so lightly edited paraphrases still score as suspicious.
10. Compare sentence passages to localize exact, near-copy, and paraphrase-risk matches.
11. Combine semantic similarity, phrase overlap, sentence reuse, sequence alignment, and localized passage evidence into the final screening score.

Output:
- paraphrase or suspicious-reuse score
- repeated phrase count
- exact sentence match count
- sequence alignment signal
- suspicious passage pair list
- summary explaining whether the similarity looks suspicious or just thematic

### Domain Clash Analysis

Input:
- `Document A`
- `Document B`

Process:
1. Tokenize both texts.
2. Build TF-IDF vectors.
3. Compute cosine similarity between the two vectors.
4. Extract top weighted terms from each document.
5. Compare which important terms are shared and which are unique.
6. Generate a summary based on similarity score and term overlap.

Output:
- overall similarity score
- shared top terms
- terms unique to each document
- summary explaining the overall similarity and differences

## Sample Test Sentences

Use these pairs in `Cursed Echo Screening` to sanity-check the detector.

### 1. Exact Copy

Source:

`The research team archived the survey data after cleaning the responses and removing duplicate entries.`

Submitted:

`The research team archived the survey data after cleaning the responses and removing duplicate entries.`

Expected result:
- `exact`
- very high score
- `1` exact sentence match
- strong suspicious reuse verdict

### 2. Punctuation and Case Changes Only

Source:

`Shared metrics matter for reliable planning. Data pipelines improve forecasting accuracy.`

Submitted:

`shared metrics matter for reliable planning! data pipelines improve forecasting accuracy?`

Expected result:
- `exact`
- very high score
- `2` exact sentence matches after normalization
- very high sequence alignment

### 3. Lightly Edited Near-Copy

Source:

`The audit found process drift in three regions before the team rewrote onboarding and retrained every field coordinator.`

Submitted:

`The audit found process drift across three regions before the team revised onboarding and retrained each field coordinator.`

Expected result:
- `near-copy` or high `paraphrase-risk`
- high score
- `0` exact sentence matches
- high sequence alignment
- at least one suspicious passage pair

### 4. Long Shared Passage

Source:

`The committee recommended weekly monitoring, stronger vendor documentation, and a formal approval log for all emergency purchases.`

Submitted:

`After reviewing the delays, the committee recommended weekly monitoring, stronger vendor documentation, and a formal approval log for all emergency purchases across every district office.`

Expected result:
- `near-copy`
- high score
- high longest shared passage signal
- localized suspicious passage evidence

### 5. Paraphrase Risk

Source:

`Students who review feedback immediately after an exam usually correct misunderstandings faster and retain the material longer.`

Submitted:

`Learners tend to fix misconceptions more quickly and remember content longer when they go over comments right after a test.`

Expected result:
- `paraphrase-risk`
- medium score
- `0` exact sentence matches
- moderate alignment and passage evidence

### 6. Thematic but Not Copied

Source:

`The city climate plan focuses on wetland restoration, flood barriers, and long-term water quality tracking.`

Submitted:

`The resilience report emphasizes marsh recovery, storm protection, and recurring checks on coastal conditions.`

Expected result:
- `thematic`
- low to medium score
- no exact sentence matches
- weak or no suspicious passage localization

### 7. Clearly Unrelated

Source:

`Basketball rotations changed after halftime and the team generated better looks from the corner.`

Submitted:

`Fermentation time and oven temperature strongly affect the texture of sourdough bread.`

Expected result:
- low score
- no suspicious passages
- no suspicious reuse verdict

### 8. Boilerplate-Heavy Overlap

Source:

`This policy is provided as is and may be updated at any time by the service team.`

Submitted:

`This policy is provided as is and may be updated at any time by the support team.`

Expected result:
- overlap exists
- score lower than a real copied passage
- repeated phrase signal softened by boilerplate filtering
- manual review still recommended if this appears inside a larger match

## Current Limitation

Image OCR is not enabled in this local build. If you upload an image file, the app will show a message telling you to use PDF, DOCX, TXT, or pasted text instead.

## Local Setup

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Create a production build:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## Project Files

- `smart_document_analyzer.jsx` - main application logic and UI.
- `analysisEngine.js` - shared analysis logic for both modes, including passage-based plagiarism screening.
- `tests/analysisEngine.test.js` - validation cases for scoring, alignment, passage localization, and edge cases.
- `main.jsx` - React entry point.
- `index.html` - app shell.
- `vite.config.js` - Vite configuration.

## Notes

- The app is set up as a Vite project.
- The remote GitHub repository is configured through `origin`.
- Image OCR can be re-added later with a backend or OCR library.
- Paraphrase or plagiarism-style screening is heuristic and should be reviewed manually before making final academic or compliance decisions.
