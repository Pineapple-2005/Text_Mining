import { useState, useRef, useCallback, useEffect } from "react";
import * as mammoth from "mammoth";
import "./smart_document_analyzer.css";

/* eslint-disable react/prop-types */

/* --- ML Core ------------------------------------------------------------- */
const STOP = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
  "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
  "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
  "not", "no", "nor", "so", "yet", "both", "either", "neither", "each", "few", "more",
  "most", "other", "some", "such", "than", "then", "that", "this", "these", "those",
  "it", "its", "we", "our", "you", "your", "he", "his", "she", "her", "they", "their",
  "i", "me", "my", "us", "as", "if", "about", "above", "after", "before", "between",
  "into", "through", "during", "also", "just", "very", "too", "well", "there", "here",
  "what", "which", "who", "whom", "when", "where", "how", "all", "any", "much", "many",
  "own", "same", "only", "once", "s", "t", "re", "ve", "ll", "d", "m", "us", "its"
]);

const tokenize = (t) =>
  t
    .toLowerCase()
    .replaceAll(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP.has(w));

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
    idf[w] = Math.log((N + 1) / (df + 1)) + 1;
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

function scoreLabelDetails(pct) {
  if (pct >= 70) return { t: "Special Grade Resonance", c: "var(--grade-special)" };
  if (pct >= 45) return { t: "Grade 1 Resonance", c: "var(--grade-mid)" };
  return { t: "Grade 4 Resonance", c: "var(--grade-low)" };
}

function buildLocalAnalysis({ pct, overlap, onlyA, onlyB, modeLabel }) {
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

/* --- File Extraction ----------------------------------------------------- */
async function extractPDF(file) {
  if (!globalThis.pdfjsLib) {
    await new Promise((ok, err) => {
      const s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
      s.onload = ok;
      s.onerror = err;
      document.head.appendChild(s);
    });
    globalThis.pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }

  const ab = await file.arrayBuffer();
  const pdf = await globalThis.pdfjsLib.getDocument({ data: ab }).promise;
  let text = "";

  for (let i = 1; i <= pdf.numPages; i += 1) {
    const pg = await pdf.getPage(i);
    const c = await pg.getTextContent();
    text += `${c.items.map((x) => x.str).join(" ")} `;
  }

  return text.trim();
}

async function extractImageOCR(_file) {
  throw new Error("Image OCR is not available in this local build. Upload PDF, DOCX, TXT, or paste text instead.");
}

const IMG_TYPES = new Set(["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"]);

async function extractFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (ext === "txt" || ext === "md") return file.text();
  if (ext === "pdf") return extractPDF(file);
  if (ext === "docx") {
    const ab = await file.arrayBuffer();
    const r = await mammoth.extractRawText({ arrayBuffer: ab });
    return r.value;
  }
  if (IMG_TYPES.has(ext)) return extractImageOCR(file);
  throw new Error(`Unsupported file: .${ext}. Supported: PDF, DOCX, TXT, JPG, PNG, WEBP, TIFF`);
}

/* --- Config -------------------------------------------------------------- */
const MODES = [
  {
    id: "resume",
    label: "Sorcerer Recruitment",
    a: "Sorcerer Profile",
    b: "Mission Brief"
  },
  {
    id: "research",
    label: "Technique Research Duel",
    a: "Field Study A",
    b: "Field Study B"
  },
  {
    id: "plagiarism",
    label: "Cursed Echo Screening",
    a: "Source Scroll",
    b: "Submitted Scroll"
  },
  {
    id: "general",
    label: "Domain Clash Analysis",
    a: "Document A",
    b: "Document B"
  },
];

const SAMPLES = {
  resume: {
    a: "First-year sorcerer with hands-on experience in anomaly triage, cursed object cataloging, and threat response. Skilled in Python tooling, text analytics, and incident reports. Built lightweight APIs for mission logs, coordinated with field teams, and maintained cloud-hosted datasets with strong documentation discipline.",
    b: "Tokyo branch seeks a mission analyst who can process cursed incident reports, build Python automation, and maintain searchable archives. Experience with NLP pipelines, REST APIs, and cloud data operations is required. The role supports active response teams and post-mission intelligence reviews."
  },
  research: {
    a: "This study maps cursed residue intensity near high-density urban wards. Using time-series anomaly tracking and lexical clustering, the team observed recurring spikes around abandoned infrastructure. Findings suggest that social stress indicators correlate with unstable energy signatures.",
    b: "We present a longitudinal analysis of urban cursed activity using text-mined mission reports and temporal hotspot mapping. The strongest signal appears around infrastructure neglect and unresolved incident chains. Results support early-warning models for curse manifestation control."
  },
  plagiarism: {
    a: "Domain expansion techniques require precise verbal structure and disciplined energy routing. Sorcerers who stabilize both sequence and intent can maintain a domain for longer durations while limiting collateral collapse.",
    b: "Executing domain expansion depends on consistent phrasing and controlled flow of cursed energy. Practitioners that keep sequence and intent synchronized tend to sustain domains longer and reduce structural failure."
  },
  general: { a: "", b: "" }
};

const PHASE_COPY = {
  "Computing TF-IDF vectors": "Tracing cursed signatures",
  "Measuring cosine similarity": "Synchronizing domain vectors",
  "Generating analysis summary": "Compiling technique dossier",
};

const DOMAIN_PROFILES = [
  {
    id: "gojo",
    label: "Gojo Profile",
    aura: "Limitless Azure",
    cadence: "calm precision",
  },
  {
    id: "sukuna",
    label: "Sukuna Profile",
    aura: "Malevolent Crimson",
    cadence: "aggressive impact",
  },
];

const DOMAIN_SCENES = {
  gojo: {
    landingKicker: "Six Eyes Sync",
    landingTitle: "Limitless-grade document resonance with polished tactical calm",
    landingBody:
      "Map document intent through a serene but high-fidelity domain: cool gradients, precise motion, and clean readability while preserving deterministic NLP output.",
    chips: ["Infinity cadence", "Blue void harmonics", "Measured reveal flow"],
    analyzerHeadline: "Precision resonance in a Limitless-styled control room",
    analyzerBody:
      "Balanced motion and high-clarity contrast for long reading sessions, research comparisons, and dossier triage.",
  },
  sukuna: {
    landingKicker: "Shrine Dominance",
    landingTitle: "Malevolent high-impact visuals for relentless comparison rituals",
    landingBody:
      "Switch to a feral domain profile with hotter contrast, sharper motion, and intense pulse effects while preserving the same trusted analyzer outputs.",
    chips: ["Shrine pressure", "Crimson slash rhythm", "Aggressive reveal bursts"],
    analyzerHeadline: "High-pressure resonance with shrine-driven visual force",
    analyzerBody:
      "Fast visual feedback and heavier atmospheric energy tuned for dramatic review passes and quick divergence spotting.",
  },
};

const LANDING_FEATURES = [
  {
    title: "Deterministic Core",
    body: "TF-IDF vectors and cosine similarity stay unchanged, so visual upgrades never alter analysis math.",
  },
  {
    title: "Dual Domain Profiles",
    body: "Toggle between Gojo and Sukuna palettes plus motion behavior for different review moods.",
  },
  {
    title: "Cursed Pulse Engine",
    body: "Click-triggered pulse bursts with optional audio modulation to make interactions feel alive.",
  },
  {
    title: "Anime Reveal Choreography",
    body: "Results unlock in staged cards so verdict, strengths, and overlap feel cinematic but still readable.",
  },
];

const DOMAIN_REFERENCES = {
  gojo: [
    {
      title: "Infinity Control",
      text: "Perfect for careful sorting of overlapping terms before final mission drafts.",
    },
    {
      title: "Hollow Purple Finish",
      text: "Use final recommendation output as your one-shot refinement pass for cleaner alignment.",
    },
    {
      title: "Reverse Technique Recovery",
      text: "When match confidence drops, use divergent traces to restore shared intent quickly.",
    },
  ],
  sukuna: [
    {
      title: "Malevolent Shrine Spread",
      text: "Rapidly expose mismatched phrase clusters and slice through noisy wording.",
    },
    {
      title: "Cleave and Dismantle",
      text: "Isolate unique terms in each document and cut non-essential variation fast.",
    },
    {
      title: "Black Flash Timing",
      text: "Use staged result reveals to sharpen focus exactly when high-value signals land.",
    },
  ],
};

const PULSE_BARS = [0, 1, 2, 3, 4, 5, 6, 7];

const scoreLabel = (p) => scoreLabelDetails(p);

/* --- Sub-components ------------------------------------------------------ */
function ScoreBar({ pct }) {
  const { c, t } = scoreLabel(pct);
  return (
    <div className="score-shell">
      <div
        className="score-ring"
        style={{ "--pct": pct, "--score-color": c }}
      >
        <div className="score-ring__inner">
          <div className="score-ring__value">
            {pct}
            <span>%</span>
          </div>
          <div className="score-ring__tier">{t}</div>
        </div>
      </div>
      <div className="score-track" aria-hidden="true">
        <div
          className="score-track__fill"
          style={{ width: `${pct}%`, background: c }}
        />
      </div>
    </div>
  );
}

function Tag({ word, variant = "neutral" }) {
  return <span className={`term-tag term-tag--${variant}`}>{word}</span>;
}

function UploadPanel({ label, value, fileName, onChange, onFile, accent }) {
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [ocrMode, setOcrMode] = useState(false);
  const [errMsg, setErrMsg] = useState("");
  const ref = useRef();

  const handle = useCallback(
    async (f) => {
      if (!f) return;
      setErrMsg("");
      setBusy(true);

      const ext = f.name.split(".").pop().toLowerCase();
      const isImg = IMG_TYPES.has(ext);
      setOcrMode(isImg);

      try {
        const t = await extractFile(f);
        onFile(t, f.name, ext);
      } catch (e) {
        setErrMsg(e.message);
        setOcrMode(false);
      }

      setBusy(false);
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDrag(false);
      const f = e.dataTransfer.files[0];
      if (f) handle(f);
    },
    [handle]
  );

  return (
    <div className="upload-card" style={{ "--panel-accent": accent }}>
      <div className="upload-card__head">
        <div className="upload-card__title">
          <span className="upload-card__dot" />
          <span>{label}</span>
        </div>
        {fileName ? (
          <div className="upload-card__meta">
            {ocrMode ? <span className="ocr-pill">OCR</span> : null}
            <span className="upload-card__filename" title={fileName}>{fileName}</span>
            <button
              type="button"
              className="ghost-icon-btn"
              onClick={() => {
                setOcrMode(false);
                setErrMsg("");
                onFile("", "", "");
              }}
              aria-label={`Clear ${label}`}
            >
              x
            </button>
          </div>
        ) : null}
      </div>

      <button
        type="button"
        className={`drop-zone ${drag ? "is-dragging" : ""}`}
        onClick={() => ref.current.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <input
          ref={ref}
          type="file"
          accept=".txt,.pdf,.docx,.md,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff"
          className="hidden-input"
          onChange={(e) => {
            const f = e.target.files[0];
            if (f) handle(f);
            e.target.value = "";
          }}
        />

        {busy ? (
          <div className="drop-zone__busy">
            <span className="spinner" aria-hidden="true" />
            <span>{ocrMode ? "Attempting OCR extraction" : "Extracting document text"}</span>
          </div>
        ) : (
          <div>
            <div className="drop-zone__headline">
              {fileName ? "Drop a replacement file" : "Drop file or click to browse"}
            </div>
            <div className="drop-zone__hint">PDF, DOCX, TXT, MD, JPG, PNG, WEBP</div>
          </div>
        )}
      </button>

      {errMsg ? <div className="panel-error">{errMsg}</div> : null}

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={11}
        className="source-input"
        placeholder={`Paste ${label.toLowerCase()} text or upload a file above...`}
      />
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section className="detail-section">
      <div className="detail-section__head">
        <span>{title}</span>
        <div className="detail-section__line" />
      </div>
      {children}
    </section>
  );
}

function useRevealStage(result) {
  const [revealStage, setRevealStage] = useState(0);
  const revealTimeoutsRef = useRef([]);

  useEffect(() => {
    revealTimeoutsRef.current.forEach((id) => globalThis.clearTimeout(id));
    revealTimeoutsRef.current = [];

    if (!result) {
      setRevealStage(0);
      return () => {
        revealTimeoutsRef.current.forEach((id) => globalThis.clearTimeout(id));
        revealTimeoutsRef.current = [];
      };
    }

    setRevealStage(1);

    [180, 360, 540, 720].forEach((delay, idx) => {
      const id = globalThis.setTimeout(() => {
        setRevealStage(idx + 2);
      }, delay);
      revealTimeoutsRef.current.push(id);
    });

    return () => {
      revealTimeoutsRef.current.forEach((id) => globalThis.clearTimeout(id));
      revealTimeoutsRef.current = [];
    };
  }, [result]);

  return revealStage;
}

function useCursedPulse(domainProfile, audioPulseEnabled) {
  const [pulseBursts, setPulseBursts] = useState([]);
  const [pulseLevel, setPulseLevel] = useState(0);

  const pulseIdRef = useRef(0);
  const pulseTimeoutsRef = useRef([]);
  const pulseDecayRef = useRef(null);
  const audioContextRef = useRef(null);

  const removePulseBurst = useCallback((pulseId) => {
    setPulseBursts((prev) => prev.filter((pulse) => pulse.id !== pulseId));
  }, []);

  useEffect(
    () => () => {
      pulseTimeoutsRef.current.forEach((id) => globalThis.clearTimeout(id));
      pulseTimeoutsRef.current = [];

      if (pulseDecayRef.current) {
        globalThis.clearTimeout(pulseDecayRef.current);
      }

      pulseDecayRef.current = null;

      const ctx = audioContextRef.current;
      if (ctx && typeof ctx.close === "function") {
        ctx.close().catch(() => {});
      }
    },
    []
  );

  const playPulseTone = useCallback(
    (strength) => {
      const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!Ctx) return;

      if (!audioContextRef.current) {
        audioContextRef.current = new Ctx();
      }

      const ctx = audioContextRef.current;
      if (ctx.state === "suspended") {
        ctx.resume().catch(() => {});
      }

      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const filter = ctx.createBiquadFilter();
      const gain = ctx.createGain();

      const isGojo = domainProfile === "gojo";
      const base = isGojo ? 230 : 150;

      osc.type = isGojo ? "triangle" : "sawtooth";
      osc.frequency.setValueAtTime(base + strength * 90, now);
      osc.frequency.exponentialRampToValueAtTime(base * 0.7, now + 0.24);

      filter.type = "bandpass";
      filter.frequency.setValueAtTime(isGojo ? 980 : 620, now);
      filter.Q.setValueAtTime(isGojo ? 5.2 : 7.8, now);

      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.045 + strength * 0.03, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.28);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.3);
    },
    [domainProfile]
  );

  const triggerPulse = useCallback(
    (strengthOrEvent, maybeEvent) => {
      const hasStrength = typeof strengthOrEvent === "number";
      const strength = hasStrength ? strengthOrEvent : 0.8;
      const evt = hasStrength ? maybeEvent : strengthOrEvent;

      const id = pulseIdRef.current + 1;
      pulseIdRef.current = id;

      const x = evt?.clientX ?? globalThis.innerWidth * (domainProfile === "gojo" ? 0.56 : 0.48);
      const y = evt?.clientY ?? globalThis.innerHeight * 0.34;

      setPulseBursts((prev) => [...prev.slice(-6), { id, x, y, strength }]);

      const cleanupId = globalThis.setTimeout(() => {
        removePulseBurst(id);
      }, 840);
      pulseTimeoutsRef.current.push(cleanupId);

      setPulseLevel(Math.min(1, 0.4 + strength * 0.58));
      if (pulseDecayRef.current) {
        globalThis.clearTimeout(pulseDecayRef.current);
      }

      pulseDecayRef.current = globalThis.setTimeout(() => {
        setPulseLevel(0);
      }, 240);

      if (audioPulseEnabled) {
        playPulseTone(strength);
      }
    },
    [audioPulseEnabled, domainProfile, playPulseTone, removePulseBurst]
  );

  return { pulseBursts, pulseLevel, triggerPulse };
}

/* --- Main App ------------------------------------------------------------ */
export default function App() { // NOSONAR
  const [mode, setMode] = useState("resume");
  const [docA, setDocA] = useState(SAMPLES.resume.a);
  const [docB, setDocB] = useState(SAMPLES.resume.b);
  const [metaA, setMetaA] = useState({ name: "", ext: "" });
  const [metaB, setMetaB] = useState({ name: "", ext: "" });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [phase, setPhase] = useState("");
  const [err, setErr] = useState("");
  const [domainProfile, setDomainProfile] = useState("gojo");
  const [audioPulseEnabled, setAudioPulseEnabled] = useState(false);

  const analyzerRef = useRef(null);

  const modeMeta = MODES.find((m) => m.id === mode) || MODES[0];
  const domainScene = DOMAIN_SCENES[domainProfile] || DOMAIN_SCENES.gojo;
  const domainRefs = DOMAIN_REFERENCES[domainProfile] || DOMAIN_REFERENCES.gojo;
  const revealStage = useRevealStage(result);
  const { pulseBursts, pulseLevel, triggerPulse } = useCursedPulse(domainProfile, audioPulseEnabled);

  const scrollToAnalyzer = useCallback(() => {
    const reduceMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    analyzerRef.current?.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "start",
    });
  }, []);

  const switchMode = (id) => {
    setMode(id);
    setDocA(SAMPLES[id]?.a || "");
    setDocB(SAMPLES[id]?.b || "");
    setMetaA({ name: "", ext: "" });
    setMetaB({ name: "", ext: "" });
    setResult(null);
    setErr("");
  };

  const analyze = useCallback(async () => {
    if (!docA.trim() || !docB.trim()) {
      setErr("Both document fields are required before starting the ritual.");
      return;
    }

    setErr("");
    setLoading(true);
    setResult(null);

    try {
      setPhase("Computing TF-IDF vectors");
      await new Promise((r) => setTimeout(r, 260));
      const [vA, vB] = tfidf([docA, docB]);

      setPhase("Measuring cosine similarity");
      await new Promise((r) => setTimeout(r, 220));
      const score = cosine(vA, vB);
      const tA = top(vA, 10);
      const tB = top(vB, 10);

      const sA = new Set(tA);
      const sB = new Set(tB);

      const overlap = unique(tA.filter((w) => sB.has(w)));
      const onlyA = unique(tA.filter((w) => !sB.has(w)));
      const onlyB = unique(tB.filter((w) => !sA.has(w)));

      setPhase("Generating analysis summary");
      await new Promise((r) => setTimeout(r, 220));

      const pct = Math.round(score * 100);
      const ai = buildLocalAnalysis({
        pct,
        overlap,
        onlyA,
        onlyB,
        modeLabel: modeMeta.label,
      });

      setResult({
        score,
        pct,
        topA: tA,
        topB: tB,
        overlap,
        onlyA,
        onlyB,
        ai,
        tokA: tokenize(docA).length,
        tokB: tokenize(docB).length,
        vocab: new Set([...tokenize(docA), ...tokenize(docB)]).size,
      });

      triggerPulse(0.9);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Analysis failed. Please try again.");
    }

    setLoading(false);
    setPhase("");
  }, [docA, docB, modeMeta.label, triggerPulse]);

  const startAnalyze = useCallback(
    (evt) => {
      triggerPulse(1.08, evt);
      analyze();
    },
    [analyze, triggerPulse]
  );

  const pct = result?.pct ?? 0;
  const scoreColor = result ? scoreLabel(pct).c : "var(--ink-muted)";
  const helperText = loading && phase
    ? `${PHASE_COPY[phase] || phase}...`
    : "Image OCR is currently disabled in local mode. Use PDF, DOCX, TXT, or pasted text.";

  const pulseButtonText = audioPulseEnabled ? "Audio pulse armed" : "Audio pulse muted";

  return (
    <div className="jjk-app" data-domain={domainProfile}>
      <div className="atmosphere" aria-hidden="true">
        <div className="mist mist--one" />
        <div className="mist mist--two" />
        <div className="mist mist--three" />
        <div className="grid-fog" />
      </div>

      <div className="pulse-layer" aria-hidden="true">
        {pulseBursts.map((pulse) => (
          <span
            key={pulse.id}
            className="curse-pulse"
            style={{
              left: `${pulse.x}px`,
              top: `${pulse.y}px`,
              "--pulse-strength": pulse.strength,
            }}
          />
        ))}
      </div>

      <header className="jjk-header">
        <div className="brand-block">
          <img src="/images/jjk-seal-main.svg" alt="Cursed seal" className="brand-seal" />
          <div>
            <h1>Cursed Document Analyzer</h1>
            <p>TF-IDF core, cosine resonance, and domain-grade interpretation</p>
          </div>
        </div>

        <div className="header-tools">
          <div className="domain-switch" role="tablist" aria-label="Domain profile">
            {DOMAIN_PROFILES.map((profile) => (
              <button
                key={profile.id}
                type="button"
                role="tab"
                aria-selected={domainProfile === profile.id}
                className={`domain-chip ${domainProfile === profile.id ? "is-active" : ""}`}
                onClick={(evt) => {
                  setDomainProfile(profile.id);
                  triggerPulse(0.78, evt);
                }}
              >
                <span>{profile.label}</span>
                <small>{profile.aura}</small>
              </button>
            ))}
          </div>

          <button
            type="button"
            className={`audio-toggle ${audioPulseEnabled ? "is-active" : ""}`}
            onClick={(evt) => {
              setAudioPulseEnabled((prev) => !prev);
              triggerPulse(0.95, evt);
            }}
            aria-pressed={audioPulseEnabled}
          >
            {pulseButtonText}
          </button>

          <div className="pulse-meter" style={{ "--pulse-level": pulseLevel }} aria-hidden="true">
            {PULSE_BARS.map((bar) => (
              <span key={`bar-${bar}`} style={{ "--bar-index": bar }} />
            ))}
          </div>
        </div>

        <div className="mode-switch" role="tablist" aria-label="Analysis mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              role="tab"
              aria-selected={mode === m.id}
              className={`mode-pill ${mode === m.id ? "is-active" : ""}`}
              onClick={(evt) => {
                switchMode(m.id);
                triggerPulse(0.65, evt);
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
      </header>

      <main className="jjk-main">
        <section className="landing-stage">
          <div className="landing-copy">
            <p className="landing-kicker">{domainScene.landingKicker}</p>
            <h2>{domainScene.landingTitle}</h2>
            <p>{domainScene.landingBody}</p>

            <div className="landing-cta-row">
              <button
                type="button"
                className="landing-cta landing-cta--primary"
                onClick={(evt) => {
                  triggerPulse(1.15, evt);
                  scrollToAnalyzer();
                }}
              >
                Enter Analyzer Domain
              </button>
              <button
                type="button"
                className="landing-cta landing-cta--ghost"
                onClick={(evt) => triggerPulse(1.2, evt)}
              >
                Trigger Cursed Pulse
              </button>
            </div>

            <div className="landing-chip-row">
              {domainScene.chips.map((chip) => (
                <span key={chip}>{chip}</span>
              ))}
            </div>
          </div>

          <div className="landing-visual" aria-hidden="true">
            <div className="landing-orbit landing-orbit--outer" />
            <div className="landing-orbit landing-orbit--inner" />
            <img src="/images/jjk-seal-main.svg" alt="" className="landing-seal landing-seal--core" />
            <img src="/images/jjk-seal-blue.svg" alt="" className="landing-seal landing-seal--blue" />
            <img src="/images/jjk-seal-red.svg" alt="" className="landing-seal landing-seal--red" />
          </div>
        </section>

        <section className="landing-grid">
          {LANDING_FEATURES.map((feature) => (
            <article key={feature.title} className="landing-tile">
              <h3>{feature.title}</h3>
              <p>{feature.body}</p>
            </article>
          ))}
        </section>

        <section className="landing-grid landing-grid--references">
          {domainRefs.map((ref) => (
            <article key={ref.title} className="landing-tile landing-tile--reference">
              <h3>{ref.title}</h3>
              <p>{ref.text}</p>
            </article>
          ))}
        </section>

        <section className="hero-card" id="analyzer" ref={analyzerRef}>
          <div className="hero-copy">
            <p className="hero-eyebrow">{domainScene.landingKicker}</p>
            <h2>{domainScene.analyzerHeadline}</h2>
            <p>{domainScene.analyzerBody}</p>
            <div className="hero-chips">
              <span>Style cadence: {DOMAIN_PROFILES.find((d) => d.id === domainProfile)?.cadence}</span>
              <span>Dynamic pulse choreography</span>
              <span>Deterministic NLP fidelity</span>
            </div>
          </div>

          <div className="hero-art" aria-hidden="true">
            <img src="/images/jjk-seal-blue.svg" alt="" className="hero-art__seal hero-art__seal--blue" />
            <img src="/images/jjk-seal-red.svg" alt="" className="hero-art__seal hero-art__seal--red" />
            <img src="/images/jjk-seal-main.svg" alt="" className="hero-art__seal hero-art__seal--main" />
          </div>
        </section>

        <section className="inputs-grid">
          <UploadPanel
            label={modeMeta.a}
            value={docA}
            fileName={metaA.name}
            onChange={(v) => {
              setDocA(v);
              setResult(null);
            }}
            onFile={(t, n, e) => {
              setDocA(t);
              setMetaA({ name: n || "", ext: e || "" });
              setResult(null);
            }}
            accent="var(--azure-core)"
          />

          <UploadPanel
            label={modeMeta.b}
            value={docB}
            fileName={metaB.name}
            onChange={(v) => {
              setDocB(v);
              setResult(null);
            }}
            onFile={(t, n, e) => {
              setDocB(t);
              setMetaB({ name: n || "", ext: e || "" });
              setResult(null);
            }}
            accent="var(--crimson-core)"
          />
        </section>

        {err ? <div className="global-error">{err}</div> : null}

        <div className="action-row">
          <button
            type="button"
            className={`analyze-btn ${loading ? "is-loading" : ""}`}
            onClick={startAnalyze}
            disabled={loading}
          >
            {loading ? (
              <span className="analyze-btn__loading">
                <span className="spinner" aria-hidden="true" />
                <span>Channeling analysis</span>
              </span>
            ) : (
              "Activate Domain Analysis"
            )}
          </button>

          <span className="phase-copy">{helperText}</span>
        </div>

        {result ? (
          <section className="results-stack is-active">
            <article className={`result-card reveal-card reveal-card--1 verdict-grid ${revealStage >= 1 ? "is-visible" : ""}`}>
              <div>
                <div className="small-label">Curse Resonance Score</div>
                <ScoreBar pct={pct} />
                <div className="score-meta" style={{ color: scoreColor }}>
                  {result.ai.match_label || scoreLabel(pct).t}
                  <span className="dot-sep">|</span>
                  <span>{result.ai.confidence || "low"} confidence</span>
                </div>
              </div>

              <div>
                <div className="small-label">Verdict</div>
                <p className="verdict-copy">{result.ai.verdict}</p>
                <p className="verdict-mode">Mode: {modeMeta.label}</p>
              </div>
            </article>

            <article className={`result-card reveal-card reveal-card--2 summary-grid ${revealStage >= 2 ? "is-visible" : ""}`}>
              {[
                {
                  label: "Resonant Signatures",
                  text: result.ai.strength,
                  cls: "summary-block--good",
                },
                {
                  label: "Divergent Traces",
                  text: result.ai.gap,
                  cls: "summary-block--warn",
                },
                {
                  label: "Technique Refinement",
                  text: result.ai.recommendation,
                  cls: "summary-block--focus",
                },
              ].map((item) => (
                <div key={item.label} className={`summary-block ${item.cls}`}>
                  <div className="summary-block__title">{item.label}</div>
                  <p>{item.text}</p>
                </div>
              ))}
            </article>

            <article className={`result-card reveal-card reveal-card--3 ${revealStage >= 3 ? "is-visible" : ""}`}>
              <Section title="Cursed Term Signatures">
                <div className="term-columns">
                  {[
                    { label: modeMeta.a, words: result.topA, other: new Set(result.topB) },
                    { label: modeMeta.b, words: result.topB, other: new Set(result.topA) },
                  ].map(({ label, words, other }) => (
                    <div key={label}>
                      <div className="term-column__title">{label}</div>
                      <div className="term-list">
                        {words.map((w) => (
                          <Tag key={w} word={w} variant={other.has(w) ? "shared" : "neutral"} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="legend-row">
                  <Tag word="shared term" variant="shared" />
                  <span>appears in both documents</span>
                </div>
              </Section>
            </article>

            <article className={`result-card reveal-card reveal-card--4 ${revealStage >= 4 ? "is-visible" : ""}`}>
              <Section title="Curse Overlap Breakdown">
                <div className="overlap-grid">
                  {[
                    { label: `Shared (${result.overlap.length})`, words: result.overlap, variant: "shared" },
                    {
                      label: `Only in ${modeMeta.a} (${result.onlyA.length})`,
                      words: result.onlyA,
                      variant: "unique",
                    },
                    {
                      label: `Only in ${modeMeta.b} (${result.onlyB.length})`,
                      words: result.onlyB,
                      variant: "unique",
                    },
                  ].map(({ label, words, variant }) => (
                    <div key={label}>
                      <div className="term-column__title">{label}</div>
                      <div className="term-list">
                        {words.length ? (
                          words.map((w) => <Tag key={w} word={w} variant={variant} />)
                        ) : (
                          <span className="none-copy">No terms detected</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </Section>
            </article>

            <article className={`result-card reveal-card reveal-card--5 ${revealStage >= 5 ? "is-visible" : ""}`}>
              <Section title="Engine Ritual Details">
                <div className="stats-grid">
                  {[
                    ["Algorithm", "TF-IDF + Cosine Similarity"],
                    ["Document A tokens", result.tokA.toLocaleString()],
                    ["Document B tokens", result.tokB.toLocaleString()],
                    ["Shared vocabulary", `${result.vocab.toLocaleString()} terms`],
                    ["Raw cosine score", result.score.toFixed(6)],
                    ["Scale", "0.000 (no match) to 1.000 (identical)"],
                  ].map(([k, v]) => (
                    <div key={k} className="stat-item">
                      <div className="stat-item__key">{k}</div>
                      <div className="stat-item__value">{v}</div>
                    </div>
                  ))}
                </div>
              </Section>
            </article>
          </section>
        ) : null}
      </main>
    </div>
  );
}