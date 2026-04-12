import { useState, useRef, useCallback, useEffect } from "react";
import * as mammoth from "mammoth";
import { analyzeByMode } from "./analysisEngine.js";
import "./smart_document_analyzer.css";

/* eslint-disable react/prop-types */

function scoreLabelDetails(pct) {
  if (pct >= 70) return { t: "Special Grade Resonance", c: "var(--grade-special)" };
  if (pct >= 45) return { t: "Grade 1 Resonance", c: "var(--grade-mid)" };
  return { t: "Grade 4 Resonance", c: "var(--grade-low)" };
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
    analyzerHeadline: "Precision resonance in a Limitless-styled control room",
    analyzerBody:
      "Balanced motion and high-clarity contrast for long reading sessions, paraphrase screening, and TF-IDF dossier triage.",
  },
  sukuna: {
    analyzerHeadline: "High-pressure resonance with shrine-driven visual force",
    analyzerBody:
      "Fast visual feedback and heavier atmospheric energy tuned for dramatic review passes and quick divergence spotting.",
  },
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
        ctx.close().catch(() => { });
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
        ctx.resume().catch(() => { });
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
  const [mode, setMode] = useState("plagiarism");
  const [docA, setDocA] = useState(SAMPLES.plagiarism.a);
  const [docB, setDocB] = useState(SAMPLES.plagiarism.b);
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

      setPhase("Measuring cosine similarity");
      await new Promise((r) => setTimeout(r, 220));

      setPhase("Generating analysis summary");
      await new Promise((r) => setTimeout(r, 220));
      const analysisResult = analyzeByMode({
        mode,
        docA,
        docB,
        modeLabel: modeMeta.label,
      });
      setResult(analysisResult);

      triggerPulse(0.9);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Analysis failed. Please try again.");
    }

    setLoading(false);
    setPhase("");
  }, [docA, docB, mode, modeMeta.label, triggerPulse]);

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
              <Section title="Mode Signal Breakdown">
                <div className="stats-grid">
                  {(result.modeSignals || []).map(([k, v]) => (
                    <div key={k} className="stat-item">
                      <div className="stat-item__key">{k}</div>
                      <div className="stat-item__value">{v}</div>
                    </div>
                  ))}
                </div>
              </Section>
            </article>

            <article className={`result-card reveal-card reveal-card--4 ${revealStage >= 4 ? "is-visible" : ""}`}>
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

            <article className={`result-card reveal-card reveal-card--5 ${revealStage >= 5 ? "is-visible" : ""}`}>
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

            <article className={`result-card reveal-card reveal-card--6 ${revealStage >= 6 ? "is-visible" : ""}`}>
              {mode === "plagiarism" ? (
                <Section title="Suspicious Passage Review">
                  {result.suspiciousPassages?.length ? (
                    <div className="passage-stack">
                      {result.suspiciousPassages.map((passage, index) => (
                        <article key={`${passage.sourceId}-${passage.submittedId}-${index}`} className="passage-card">
                          <div className="passage-card__meta">
                            <span className={`passage-kind passage-kind--${passage.kind}`}>{passage.kind}</span>
                            <span>{Math.round(passage.score * 100)}% confidence</span>
                            <span>{passage.longestSharedRun} shared tokens</span>
                          </div>

                          <div className="passage-columns">
                            <div className="passage-column">
                              <div className="term-column__title">{modeMeta.a}</div>
                              <p>{passage.sourceText}</p>
                            </div>
                            <div className="passage-column">
                              <div className="term-column__title">{modeMeta.b}</div>
                              <p>{passage.submittedText}</p>
                            </div>
                          </div>

                          <div className="passage-evidence">
                            <span>Longest shared text: {passage.longestSharedText || "No long contiguous span detected"}</span>
                            {passage.repeatedPhrases?.length ? (
                              <span>Repeated phrases: {passage.repeatedPhrases.join(", ")}</span>
                            ) : (
                              <span>Repeated phrases: none retained after boilerplate filtering</span>
                            )}
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <p className="none-copy">No suspicious passage pairs were localized.</p>
                  )}
                </Section>
              ) : (
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
              )}
            </article>

            <article className={`result-card reveal-card reveal-card--7 ${revealStage >= 6 ? "is-visible" : ""}`}>
              <Section title="Engine Ritual Details">
                <div className="stats-grid">
                  {[
                    [
                      "Algorithm",
                      mode === "plagiarism"
                        ? "TF-IDF + phrase overlap + sentence normalization + passage localization"
                        : "TF-IDF + Cosine Similarity",
                    ],
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
