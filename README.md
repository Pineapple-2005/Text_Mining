# Smart Document Analyzer

Smart Document Analyzer is a browser-based React app for comparing two documents with TF-IDF and cosine similarity. It supports resume matching, research comparison, plagiarism screening, and general document analysis.

## What It Does

- Compares two text inputs and calculates a similarity score.
- Extracts text from `.txt`, `.md`, `.pdf`, and `.docx` files.
- Uses a local analysis summary to explain overlap, differences, and recommendations.
- Provides preset comparison modes for common document workflows.

## How It Works

1. Each document is tokenized and cleaned with a stop-word filter.
2. TF-IDF vectors are built for both documents.
3. Cosine similarity is used to produce the match score.
4. The app summarizes shared terms, unique terms, and practical guidance.

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
- `main.jsx` - React entry point.
- `index.html` - app shell.
- `vite.config.js` - Vite configuration.

## Notes

- The app is set up as a Vite project.
- The remote GitHub repository is configured through `origin`.
- Image OCR can be re-added later with a backend or OCR library.