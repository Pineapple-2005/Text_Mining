# Smart Document Analyzer

Smart Document Analyzer is a browser-based React app for comparing multiple documents with TF-IDF and cosine similarity. It supports resume matching, research comparison, plagiarism screening, and general document analysis, including both pairwise ranking and one-vs-many review.

## What It Does

- Compares two or more text inputs and ranks every pairwise similarity score.
- Supports a one-vs-many workflow so one anchor document can be checked against a batch.
- Extracts text from `.txt`, `.md`, `.pdf`, and `.docx` files.
- Uses a local analysis summary to explain overlap, differences, and recommendations.
- Provides preset comparison modes for common document workflows.
- Lets you add and remove document panels without changing the upload workflow.

## How It Works

1. Each document is tokenized and cleaned with a stop-word filter.
2. TF-IDF vectors are built for every populated document.
3. Cosine similarity is used to produce ranked pairwise match scores or anchor-vs-batch results.
4. The app highlights the strongest match, aggregate stats, and practical guidance.

## Current Limitation

Image OCR is not enabled in this local build. If you upload an image file, the app will show a message telling you to use PDF, DOCX, TXT, or pasted text instead.

## Local Setup

Install dependencies:

```bash
npm install
```

Run the regression tests:

```bash
npm test
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

- `analysisEngine.js` - reusable tokenization, TF-IDF, cosine, and multi-document analysis helpers.
- `analysisEngine.test.js` - regression coverage for the core analysis engine.
- `smart_document_analyzer.jsx` - main application logic and UI.
- `main.jsx` - React entry point.
- `index.html` - app shell.
- `vite.config.js` - Vite configuration.

## Notes

- The app is set up as a Vite project.
- The remote GitHub repository is configured through `origin`.
- Image OCR can be re-added later with a backend or OCR library.
