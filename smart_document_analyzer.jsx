import { useState, useRef, useCallback, useEffect } from "react";
import * as mammoth from "mammoth";
import * as THREE from "three";

/* ─── ML Core ─────────────────────────────────────────────────────────── */
const STOP = new Set([
  "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
  "from","is","are","was","were","be","been","being","have","has","had","do",
  "does","did","will","would","could","should","may","might","shall","can",
  "not","no","nor","so","yet","both","either","neither","each","few","more",
  "most","other","some","such","than","then","that","this","these","those",
  "it","its","we","our","you","your","he","his","she","her","they","their",
  "i","me","my","us","as","if","about","above","after","before","between",
  "into","through","during","also","just","very","too","well","there","here",
  "what","which","who","whom","when","where","how","all","any","much","many",
  "own","same","only","once","s","t","re","ve","ll","d","m","us","its"
]);
const tokenize = t => t.toLowerCase().replaceAll(/[^a-z0-9\s]/g," ").split(/\s+/).filter(w=>w.length>2&&!STOP.has(w));
const tf = tokens => { const f={}; tokens.forEach(t=>{f[t]=(f[t]||0)+1;}); const mx=Math.max(...Object.values(f),1); Object.keys(f).forEach(k=>{f[k]/=mx;}); return f; };
const tfidf = docs => {
  const tok=docs.map(tokenize), N=docs.length, idf={};
  new Set(tok.flat()).forEach(w=>{const df=tok.filter(d=>d.includes(w)).length; idf[w]=Math.log((N+1)/(df+1))+1;});
  return tok.map(tokens=>{const f=tf(tokens),v={}; Object.keys(f).forEach(t=>{v[t]=f[t]*(idf[t]||1);}); return v;});
};
const cosine = (a,b) => { const T=new Set([...Object.keys(a),...Object.keys(b)]); let dot=0,mA=0,mB=0; T.forEach(t=>{const x=a[t]||0,y=b[t]||0; dot+=x*y;mA+=x*x;mB+=y*y;}); return mA&&mB?dot/(Math.sqrt(mA)*Math.sqrt(mB)):0; };
const top = (v,n=10) => Object.entries(v).sort((a,b)=>b[1]-a[1]).slice(0,n).map(([w])=>w);
const unique = arr => [...new Set(arr)];

function scoreLabelDetails(pct) {
  if (pct >= 70) return { t: "RESONANT TECHNIQUE", c: "var(--green)" };
  if (pct >= 45) return { t: "PARTIAL RESONANCE",  c: "var(--gold)"  };
  return            { t: "DIVERGENT ENERGY",       c: "var(--crimson)" };
}

function buildLocalAnalysis({ pct, overlap, onlyA, onlyB, modeLabel }) {
  const matchLabel = scoreLabelDetails(pct).t;
  const sharedText = overlap.length
    ? `Both scrolls resonate through ${overlap.slice(0,5).join(", ")} — their cursed techniques share the same root vocabulary, amplifying the binding score.`
    : "These scrolls share little direct cursed vocabulary. The resonance arises from broader thematic overlap rather than exact term binding.";
  const gapText = onlyA.length || onlyB.length
    ? `Scroll I channels ${onlyA.slice(0,4).join(", ") || "distinct techniques"} while Scroll II invokes ${onlyB.slice(0,4).join(", ") || "a separate cursed domain"} — their energies are not fully aligned.`
    : "No significant divergence detected in the binding vow matrix.";
  const recommendationText = pct >= 70
    ? `The resonance is strong. Maintain aligned cursed terminology for ${modeLabel.toLowerCase()}. Refine any remaining divergences before final domain expansion.`
    : `Strengthen the binding vow by aligning key terminology for ${modeLabel.toLowerCase()}. Realign conceptual structure to amplify shared resonance.`;

  return {
    verdict: `${matchLabel} — the scrolls are ${pct >= 70 ? "deeply bound" : pct >= 45 ? "partially resonant" : "weakly linked"} through shared cursed energy and overlapping technique vocabulary.`,
    match_label: matchLabel,
    strength: sharedText,
    gap: gapText,
    recommendation: recommendationText,
    confidence: pct >= 70 ? "HIGH" : pct >= 45 ? "MODERATE" : "LOW",
  };
}

/* ─── File Extraction ─────────────────────────────────────────────────── */
async function extractPDF(file) {
  if (!window.pdfjsLib) {
    await new Promise((ok,err)=>{
      const s=document.createElement("script");
      s.src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
      s.onload=ok; s.onerror=err; document.head.appendChild(s);
    });
    window.pdfjsLib.GlobalWorkerOptions.workerSrc=
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }
  const ab=await file.arrayBuffer();
  const pdf=await window.pdfjsLib.getDocument({data:ab}).promise;
  let text="";
  for(let i=1;i<=pdf.numPages;i++){
    const pg=await pdf.getPage(i);
    const c=await pg.getTextContent();
    text+=c.items.map(x=>x.str).join(" ")+" ";
  }
  return text.trim();
}

async function extractImageOCR(file) {
  throw new Error("Image OCR is not available in this local build. Upload PDF, DOCX, TXT, or paste text instead.");
}

const IMG_TYPES=new Set(["jpg","jpeg","png","gif","webp","bmp","tiff"]);
async function extractFile(file) {
  const ext=file.name.split(".").pop().toLowerCase();
  if(ext==="txt"||ext==="md") return file.text();
  if(ext==="pdf") return extractPDF(file);
  if(ext==="docx"){const ab=await file.arrayBuffer();const r=await mammoth.extractRawText({arrayBuffer:ab});return r.value;}
  if(IMG_TYPES.has(ext)) return extractImageOCR(file);
  throw new Error(`Unsupported file: .${ext}. Supported: PDF, DOCX, TXT, JPG, PNG, WEBP, TIFF`);
}

/* ─── Config ──────────────────────────────────────────────────────────── */
const MODES=[
  {id:"resume",    label:"Sorcerer Eval",    a:"Sorcerer Dossier",  b:"Mission Scroll"},
  {id:"research",  label:"Technique Clash",  a:"Technique Alpha",   b:"Technique Beta"},
  {id:"plagiarism",label:"Cursed Trace",     a:"Original Grimoire", b:"Suspect Script"},
  {id:"general",   label:"Domain Scan",      a:"Cursed Script I",   b:"Cursed Script II"},
];
const SAMPLES={
  resume:{
    a:`Experienced software engineer with 5 years of expertise in Python, machine learning, and data analysis. Proficient in TensorFlow, PyTorch, scikit-learn, and deep learning frameworks. Built scalable REST APIs using Django and FastAPI. Led agile development teams and delivered high-performance backend systems. Strong background in SQL, PostgreSQL, AWS cloud infrastructure, and Docker containerization.`,
    b:`We are seeking a Software Engineer experienced in Python and machine learning. Hands-on expertise with TensorFlow or PyTorch is required. You will design REST APIs and work with AWS or GCP cloud services. Experience with SQL databases, Docker, and agile methodologies is expected. Strong communication and collaboration skills are essential.`
  },
  research:{
    a:`This study investigates the effects of climate change on coral reef biodiversity in tropical marine ecosystems. Using satellite imaging and underwater surveys, we documented significant bleaching events correlated with sea surface temperature anomalies. Results indicate a 35% decline in species diversity over the past decade.`,
    b:`We present a longitudinal analysis of ocean temperature rise and its consequences on reef-building corals. Our data reveals accelerated bleaching frequency tied to global warming. Biodiversity loss in affected zones reached critical thresholds. The study calls for strengthened marine conservation policies.`
  },
  plagiarism:{
    a:`Artificial intelligence is transforming industries by enabling machines to learn from data and make intelligent decisions. Machine learning algorithms process large datasets to identify patterns and improve predictions over time. Deep learning uses neural networks to model complex relationships in data.`,
    b:`Artificial intelligence is revolutionizing sectors by allowing computers to learn from data and make smart decisions. Machine learning techniques analyze large amounts of information to discover patterns and enhance forecasts. Deep learning employs neural networks to capture intricate relationships.`
  },
  general:{a:"",b:""}
};

/* ─── CSS Injection ───────────────────────────────────────────────────── */
const GLOBAL_CSS = `
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
  --bg:            #06060b;
  --bg-surface:    #0b0b14;
  --bg-panel:      #0f0f1c;
  --bg-elevated:   #151526;
  --bg-card:       #111122;
  --border:        rgba(255,255,255,0.055);
  --border-mid:    rgba(255,255,255,0.09);
  --border-accent: rgba(124,92,252,0.38);
  --text:          #ece8f8;
  --text-muted:    #8a87a8;
  --text-faint:    #44425e;
  --accent:        #7c5cfc;
  --accent-dim:    rgba(124,92,252,0.18);
  --accent-glow:   rgba(124,92,252,0.35);
  --blue:          #00d4ff;
  --blue-dim:      rgba(0,212,255,0.12);
  --blue-glow:     rgba(0,212,255,0.3);
  --crimson:       #ff2855;
  --crimson-dim:   rgba(255,40,85,0.12);
  --crimson-glow:  rgba(255,40,85,0.3);
  --gold:          #ffab30;
  --gold-dim:      rgba(255,171,48,0.12);
  --gold-glow:     rgba(255,171,48,0.3);
  --green:         #00e887;
  --green-dim:     rgba(0,232,135,0.12);
  --green-glow:    rgba(0,232,135,0.3);
  --font-display:  'Bebas Neue', Impact, sans-serif;
  --font-ui:       'Syne', system-ui, sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-ui);
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,92,252,0.45); border-radius: 2px; }

@keyframes orbDrift {
  0%   { transform: translate(0px, 0px) scale(1); }
  33%  { transform: translate(30px, -40px) scale(1.05); }
  66%  { transform: translate(-20px, 30px) scale(0.97); }
  100% { transform: translate(0px, 0px) scale(1); }
}
@keyframes orbDrift2 {
  0%   { transform: translate(0, 0) scale(1); }
  40%  { transform: translate(-40px, 25px) scale(1.08); }
  70%  { transform: translate(20px, -30px) scale(0.95); }
  100% { transform: translate(0, 0) scale(1); }
}
@keyframes orbDrift3 {
  0%   { transform: translate(0, 0); }
  50%  { transform: translate(15px, 40px); }
  100% { transform: translate(0, 0); }
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: none; }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
@keyframes auraExpand {
  0%   { transform: translate(-50%, -50%) scale(1);   opacity: 0.55; }
  100% { transform: translate(-50%, -50%) scale(2.6); opacity: 0; }
}
@keyframes borderPulse {
  0%, 100% { box-shadow: 0 0 0 1px var(--border-accent), 0 0 18px rgba(124,92,252,0.12); }
  50%       { box-shadow: 0 0 0 1px rgba(124,92,252,0.6), 0 0 32px rgba(124,92,252,0.22); }
}
@keyframes scanSweep {
  0%   { top: -2px; opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 0.6; }
  100% { top: 100%; opacity: 0; }
}
@keyframes titleReveal {
  from { opacity: 0; letter-spacing: 8px; }
  to   { opacity: 1; letter-spacing: 4px; }
}
@keyframes glitchFlash {
  0%, 90%, 100% { clip-path: none; transform: none; }
  92% { clip-path: inset(20% 0 50% 0); transform: translateX(-3px); }
  94% { clip-path: inset(60% 0 10% 0); transform: translateX(3px); }
  96% { clip-path: none; transform: none; }
}
@keyframes orbPulse {
  0%, 100% { transform: scale(1); opacity: 0.7; }
  50% { transform: scale(1.05); opacity: 1; }
}
@keyframes ringSpin {
  from { transform: translate(-50%, -50%) rotate(0deg); }
  to { transform: translate(-50%, -50%) rotate(360deg); }
}
@keyframes runeFloat {
  0%, 100% { transform: translateY(0px); opacity: 0.5; }
  50% { transform: translateY(-8px); opacity: 1; }
}
@keyframes expansionFlash {
  0% { opacity: 0; transform: scale(0.72); }
  20% { opacity: 0.9; }
  100% { opacity: 0; transform: scale(1.9); }
}
@keyframes expansionSweep {
  0% { transform: translateY(-110%) skewY(-6deg); opacity: 0; }
  25% { opacity: 0.7; }
  100% { transform: translateY(120%) skewY(-6deg); opacity: 0; }
}
@keyframes expansionRunes {
  0%, 100% { opacity: 0.2; transform: scale(0.96); }
  50% { opacity: 1; transform: scale(1.03); }
}
@keyframes constellationDrift {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(0, -6px, 0) scale(1.03); }
}
@keyframes constellationPulse {
  0%, 100% { opacity: 0.45; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.08); }
}
@keyframes reactorSpin {
  from { transform: translate(-50%, -50%) rotate(0deg); }
  to { transform: translate(-50%, -50%) rotate(360deg); }
}
@keyframes reactorCorePulse {
  0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.85; }
  50% { transform: translate(-50%, -50%) scale(1.08); opacity: 1; }
}

/* ── Orb background elements ── */
.jjk-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  pointer-events: none;
}
.jjk-orb-1 {
  width: 600px; height: 600px;
  background: radial-gradient(circle, rgba(124,92,252,0.22) 0%, transparent 70%);
  top: -200px; left: -150px;
  animation: orbDrift 20s ease-in-out infinite;
}
.jjk-orb-2 {
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(0,212,255,0.15) 0%, transparent 70%);
  top: 40%; right: -100px;
  animation: orbDrift2 25s ease-in-out infinite;
}
.jjk-orb-3 {
  width: 350px; height: 350px;
  background: radial-gradient(circle, rgba(255,40,85,0.12) 0%, transparent 70%);
  bottom: 10%; left: 30%;
  animation: orbDrift3 18s ease-in-out infinite;
}

/* ── Panel 3D hover ── */
.cursed-panel {
  position: relative;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  transition: transform 0.3s cubic-bezier(.25,.8,.25,1),
              box-shadow 0.3s cubic-bezier(.25,.8,.25,1),
              border-color 0.3s;
  transform-style: preserve-3d;
}
.cursed-panel:hover {
  border-color: var(--border-accent);
  transform: perspective(900px) rotateX(-1.5deg) translateY(-4px);
  box-shadow: 0 24px 64px rgba(0,0,0,0.6),
              0 0 0 1px rgba(124,92,252,0.25),
              0 8px 32px rgba(124,92,252,0.15);
}

/* ── Analysis card 3D ── */
.analysis-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 20px;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s;
  transform-style: preserve-3d;
  cursor: default;
}
.analysis-card:hover {
  transform: perspective(600px) rotateX(-2deg) rotateY(1deg) translateY(-6px) translateZ(10px);
  box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 30px var(--card-glow, rgba(124,92,252,0.15));
}

/* ── Domain button ── */
.domain-btn {
  font-family: var(--font-display);
  font-size: 22px;
  letter-spacing: 4px;
  color: #fff;
  background: linear-gradient(145deg, #7c5cfc 0%, #5a3ecf 50%, #3d28a8 100%);
  border: none;
  padding: 15px 52px 13px;
  border-radius: 5px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transform: perspective(400px) rotateX(4deg);
  box-shadow:
    0 6px 0 #2b1e7a,
    0 10px 30px rgba(124,92,252,0.45),
    inset 0 1px 0 rgba(255,255,255,0.15);
  transition: transform 0.12s, box-shadow 0.12s;
}
.domain-btn::before {
  content: '';
  position: absolute;
  top: -2px; left: -100%;
  width: 60%; height: calc(100% + 4px);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  transform: skewX(-15deg);
  transition: left 0.5s;
}
.domain-btn:hover::before { left: 160%; }
.domain-btn:hover {
  transform: perspective(400px) rotateX(4deg) translateY(-3px);
  box-shadow:
    0 9px 0 #2b1e7a,
    0 16px 40px rgba(124,92,252,0.6),
    inset 0 1px 0 rgba(255,255,255,0.18);
}
.domain-btn:active {
  transform: perspective(400px) rotateX(4deg) translateY(3px);
  box-shadow:
    0 2px 0 #2b1e7a,
    0 4px 16px rgba(124,92,252,0.4);
}
.domain-btn:disabled {
  background: var(--bg-elevated);
  color: var(--text-faint);
  box-shadow: none;
  transform: none;
  cursor: not-allowed;
}
.domain-btn:disabled::before { display: none; }

/* ── Mode tabs ── */
.mode-tab {
  font-family: var(--font-ui);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--text-faint);
  background: transparent;
  border: 1px solid transparent;
  padding: 7px 15px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.18s;
}
.mode-tab:hover {
  color: var(--text-muted);
  border-color: var(--border-mid);
  background: var(--bg-elevated);
}
.mode-tab.active {
  color: var(--text);
  background: var(--bg-elevated);
  border-color: var(--border-accent);
  box-shadow: 0 0 16px rgba(124,92,252,0.2), inset 0 1px 0 rgba(124,92,252,0.15);
}

/* ── Cursed tags ── */
.ctag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 11px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  border: 1px solid;
  line-height: 18px;
  transition: transform 0.15s, box-shadow 0.15s;
}
.ctag:hover {
  transform: translateY(-1px);
}
.ctag-shared {
  background: var(--green-dim);
  color: var(--green);
  border-color: rgba(0,232,135,0.25);
}
.ctag-shared:hover { box-shadow: 0 4px 12px rgba(0,232,135,0.2); }
.ctag-unique {
  background: var(--accent-dim);
  color: var(--accent);
  border-color: rgba(124,92,252,0.25);
}
.ctag-unique:hover { box-shadow: 0 4px 12px rgba(124,92,252,0.2); }
.ctag-neutral {
  background: rgba(255,255,255,0.03);
  color: var(--text-muted);
  border-color: var(--border-mid);
}

/* ── Textarea ── */
.cursed-textarea {
  font-family: var(--font-ui);
  font-size: 13px;
  line-height: 1.8;
  color: var(--text);
  background: rgba(0,0,0,0.35);
  border: 1px solid var(--border-mid);
  border-radius: 5px;
  padding: 14px 15px;
  resize: vertical;
  width: 100%;
  outline: none;
  transition: border-color 0.18s, box-shadow 0.18s, background 0.18s;
  caret-color: var(--accent);
}
.cursed-textarea::placeholder { color: var(--text-faint); }
.cursed-textarea:focus {
  border-color: var(--border-accent);
  background: rgba(0,0,0,0.5);
  box-shadow: 0 0 0 3px rgba(124,92,252,0.12), 0 0 20px rgba(124,92,252,0.1);
}
.cursed-textarea::-webkit-scrollbar { width: 3px; }
.cursed-textarea::-webkit-scrollbar-thumb { background: rgba(124,92,252,0.35); }

/* ── Drop zone ── */
.drop-zone {
  border: 1px dashed rgba(255,255,255,0.12);
  border-radius: 5px;
  padding: 18px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.18s, background 0.18s;
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.02), transparent 36%),
    rgba(0,0,0,0.18);
}
.drop-zone::before {
  content: "";
  position: absolute;
  inset: 10px;
  border-radius: 4px;
  border: 1px solid color-mix(in srgb, var(--panel-accent, var(--accent)) 18%, transparent);
  pointer-events: none;
  opacity: 0.75;
}
.drop-zone:hover, .drop-zone.dragging {
  border-color: color-mix(in srgb, var(--panel-accent, var(--accent)) 40%, transparent);
  background: color-mix(in srgb, var(--panel-accent, var(--accent)) 12%, rgba(0,0,0,0.2));
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--panel-accent, var(--accent)) 18%, transparent), 0 0 30px color-mix(in srgb, var(--panel-accent, var(--accent)) 10%, transparent);
}
.drop-zone .scan-line {
  position: absolute;
  left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  animation: scanSweep 3s ease-in-out infinite;
  pointer-events: none;
}
.holo-panel {
  position: relative;
  isolation: isolate;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.03), transparent 18%),
    linear-gradient(180deg, rgba(15,15,28,0.96), rgba(9,9,18,0.94));
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.06),
    0 24px 70px rgba(0,0,0,0.38);
}
.holo-panel::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 8px;
  pointer-events: none;
  background:
    linear-gradient(135deg, transparent 0 12%, color-mix(in srgb, var(--panel-accent) 22%, transparent) 12% 13%, transparent 13% 87%, color-mix(in srgb, var(--panel-accent) 22%, transparent) 87% 88%, transparent 88%),
    radial-gradient(circle at 15% 18%, color-mix(in srgb, var(--panel-accent) 14%, transparent), transparent 30%);
  opacity: 0.9;
}
.holo-panel::after {
  content: "";
  position: absolute;
  inset: 10px;
  border-radius: 6px;
  border: 1px solid color-mix(in srgb, var(--panel-accent) 16%, rgba(255,255,255,0.04));
  pointer-events: none;
}
.holo-topbar {
  position: absolute;
  top: 0;
  left: 18px;
  right: 18px;
  height: 1px;
  background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--panel-accent) 70%, transparent), transparent);
  box-shadow: 0 0 14px color-mix(in srgb, var(--panel-accent) 30%, transparent);
}
.holo-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  border: 1px solid color-mix(in srgb, var(--panel-accent) 20%, transparent);
  border-radius: 999px;
  background: color-mix(in srgb, var(--panel-accent) 8%, rgba(0,0,0,0.25));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.panel-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: color-mix(in srgb, var(--panel-accent) 76%, white);
  background: color-mix(in srgb, var(--panel-accent) 10%, rgba(255,255,255,0.02));
  border: 1px solid color-mix(in srgb, var(--panel-accent) 18%, transparent);
}

/* ── Aura rings ── */
.aura-ring {
  position: absolute;
  border-radius: 50%;
  border: 1px solid;
  top: 50%; left: 50%;
  animation: auraExpand 2.4s ease-out infinite;
  pointer-events: none;
}

/* ── Sigil animation ── */
.sigil-icon {
  animation: glitchFlash 8s ease-in-out infinite;
}

/* ── Section divider ── */
.section-label {
  font-family: var(--font-ui);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--text-faint);
}

/* ── Stat block ── */
.stat-block {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 12px 16px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.stat-block:hover {
  border-color: var(--border-mid);
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}

.hero-shell {
  position: relative;
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  background:
    radial-gradient(circle at 20% 20%, rgba(124,92,252,0.12), transparent 35%),
    radial-gradient(circle at 80% 30%, rgba(0,212,255,0.08), transparent 30%),
    linear-gradient(180deg, rgba(17,17,34,0.88), rgba(9,9,18,0.94));
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.05),
    0 30px 90px rgba(0,0,0,0.45);
}
.hero-grid {
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 32px;
  align-items: center;
}
.hero-orb-stage {
  position: relative;
  min-height: 360px;
  perspective: 1400px;
  transform-style: preserve-3d;
  border-radius: 16px;
  overflow: hidden;
  background:
    radial-gradient(circle at center, rgba(124,92,252,0.08), transparent 42%),
    radial-gradient(circle at 50% 70%, rgba(0,212,255,0.08), transparent 34%);
}
.hero-webgl-canvas {
  position: absolute;
  inset: 0;
}
.hero-webgl-canvas canvas {
  width: 100% !important;
  height: 100% !important;
  display: block;
}
.hero-orb-stage,
.hero-webgl-canvas,
.hero-orb-ring,
.hero-orb-ring-2,
.hero-shell::before {
  will-change: transform, opacity;
}
.hero-orb-ring,
.hero-orb-ring-2 {
  position: absolute;
  top: 50%;
  left: 50%;
  border-radius: 50%;
  border: 1px solid rgba(124,92,252,0.28);
  box-shadow: 0 0 20px rgba(124,92,252,0.14), inset 0 0 20px rgba(124,92,252,0.08);
}
.hero-orb-ring {
  width: 300px;
  height: 300px;
  animation: ringSpin 16s linear infinite;
}
.hero-orb-ring-2 {
  width: 360px;
  height: 360px;
  border-color: rgba(0,212,255,0.2);
  animation: ringSpin 24s linear infinite reverse;
}
.hero-orb-ring::before,
.hero-orb-ring-2::before {
  content: "";
  position: absolute;
  inset: 16px;
  border-radius: 50%;
  border: 1px dashed rgba(255,255,255,0.08);
}
.hero-orb-floor {
  position: absolute;
  top: 72%;
  left: 50%;
  width: 78%;
  height: 22%;
  transform: translateX(-50%);
  border-radius: 50%;
  background:
    radial-gradient(ellipse, rgba(0,212,255,0.18), transparent 60%),
    radial-gradient(ellipse, rgba(124,92,252,0.18), transparent 72%);
  filter: blur(20px);
  opacity: 0.85;
}
.hero-rune {
  position: absolute;
  font-family: var(--font-display);
  font-size: 11px;
  letter-spacing: 3px;
  color: var(--text-faint);
  text-shadow: 0 0 12px rgba(124,92,252,0.45);
  animation: runeFloat 4s ease-in-out infinite;
}
@media (max-width: 900px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }
  .hero-orb-stage {
    min-height: 300px;
    order: -1;
  }
}

.domain-expansion-overlay {
  position: fixed;
  inset: 0;
  z-index: 80;
  overflow: hidden;
  pointer-events: none;
}
.domain-expansion-overlay::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at center, rgba(124,92,252,0.22), transparent 34%),
    radial-gradient(circle at center, rgba(0,212,255,0.12), transparent 54%),
    rgba(6,6,11,0.68);
}
.domain-expansion-flash {
  position: absolute;
  top: 50%;
  left: 50%;
  width: min(72vw, 780px);
  aspect-ratio: 1;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow:
    0 0 40px rgba(124,92,252,0.35),
    0 0 100px rgba(0,212,255,0.14),
    inset 0 0 30px rgba(255,255,255,0.06);
  will-change: transform, opacity;
  animation: expansionFlash 0.9s ease-out forwards;
}
.domain-expansion-flash::before,
.domain-expansion-flash::after {
  content: "";
  position: absolute;
  inset: 8%;
  border-radius: 50%;
  border: 1px dashed rgba(255,255,255,0.09);
}
.domain-expansion-flash::after {
  inset: 20%;
  border-style: solid;
  border-color: rgba(0,212,255,0.18);
}
.domain-expansion-sweep {
  position: absolute;
  left: -10%;
  right: -10%;
  height: 26%;
  background: linear-gradient(180deg, transparent, rgba(255,255,255,0.08), transparent);
  filter: blur(6px);
  will-change: transform, opacity;
  animation: expansionSweep 0.9s ease-out forwards;
}
.domain-expansion-runes {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-family: var(--font-display);
  font-size: clamp(34px, 8vw, 88px);
  letter-spacing: 10px;
  color: rgba(236,232,248,0.12);
  text-shadow: 0 0 30px rgba(124,92,252,0.18);
  will-change: transform, opacity;
  animation: expansionRunes 0.9s ease-out forwards;
}
.domain-results-reveal {
  animation: fadeUp 0.6s both;
}
.constellation-shell {
  position: relative;
  padding: 24px;
  background: linear-gradient(180deg, rgba(15,15,28,0.96), rgba(10,10,20,0.94));
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
.constellation-stage {
  position: relative;
  min-height: 360px;
  border-radius: 10px;
  overflow: hidden;
  background:
    radial-gradient(circle at center, rgba(124,92,252,0.08), transparent 36%),
    radial-gradient(circle at 50% 50%, rgba(0,212,255,0.05), transparent 52%),
    rgba(6,6,11,0.58);
}
.constellation-grid {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 22px;
  align-items: center;
}
.constellation-core {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 88px;
  height: 88px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  background: radial-gradient(circle, rgba(255,255,255,0.92), rgba(124,92,252,0.32) 28%, rgba(8,10,20,0.96) 72%);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow:
    0 0 28px rgba(124,92,252,0.3),
    0 0 60px rgba(0,212,255,0.12),
    inset 0 0 16px rgba(255,255,255,0.08);
}
.constellation-core::after {
  content: "";
  position: absolute;
  inset: -18px;
  border-radius: 50%;
  border: 1px solid rgba(124,92,252,0.16);
}
.constellation-ring {
  position: absolute;
  top: 50%;
  left: 50%;
  border-radius: 50%;
  border: 1px dashed rgba(255,255,255,0.08);
  transform: translate(-50%, -50%);
}
.constellation-node {
  position: absolute;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 7px 11px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  white-space: nowrap;
  border: 1px solid;
  box-shadow: 0 0 18px rgba(0,0,0,0.22);
  animation: constellationDrift 4.5s ease-in-out infinite;
}
.constellation-node::before {
  content: "";
  position: absolute;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  animation: constellationPulse 2.2s ease-in-out infinite;
}
.constellation-node-shared {
  color: var(--green);
  border-color: rgba(0,232,135,0.26);
  background: rgba(0,232,135,0.08);
}
.constellation-node-shared::before {
  background: var(--green);
  box-shadow: 0 0 14px rgba(0,232,135,0.45);
}
.constellation-node-alpha {
  color: var(--accent);
  border-color: rgba(124,92,252,0.28);
  background: rgba(124,92,252,0.08);
}
.constellation-node-alpha::before {
  background: var(--accent);
  box-shadow: 0 0 14px rgba(124,92,252,0.45);
}
.constellation-node-beta {
  color: var(--blue);
  border-color: rgba(0,212,255,0.24);
  background: rgba(0,212,255,0.08);
}
.constellation-node-beta::before {
  background: var(--blue);
  box-shadow: 0 0 14px rgba(0,212,255,0.45);
}
.score-reactor {
  position: relative;
  width: 168px;
  height: 168px;
  transform-style: preserve-3d;
}
.score-reactor-ring,
.score-reactor-ring-2 {
  position: absolute;
  top: 50%;
  left: 50%;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  will-change: transform;
}
.score-reactor-ring {
  width: 168px;
  height: 168px;
  border: 1px solid var(--reactor-color);
  box-shadow: 0 0 24px color-mix(in srgb, var(--reactor-color) 24%, transparent);
  animation: reactorSpin 12s linear infinite;
}
.score-reactor-ring::before {
  content: "";
  position: absolute;
  inset: 12px;
  border-radius: 50%;
  border: 1px dashed color-mix(in srgb, var(--reactor-color) 45%, transparent);
}
.score-reactor-ring-2 {
  width: 126px;
  height: 126px;
  border: 1px solid color-mix(in srgb, var(--reactor-color) 55%, transparent);
  animation: reactorSpin 8s linear infinite reverse;
}
.score-reactor-ring-2::before,
.score-reactor-ring-2::after {
  content: "";
  position: absolute;
  inset: -10px;
  border-radius: 50%;
  border-top: 2px solid color-mix(in srgb, var(--reactor-color) 70%, transparent);
  border-left: 2px solid transparent;
  border-right: 2px solid transparent;
  border-bottom: 2px solid transparent;
}
.score-reactor-ring-2::after {
  inset: 10px;
  border-top-color: color-mix(in srgb, var(--reactor-color) 35%, transparent);
}
.score-reactor-core {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 94px;
  height: 94px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  background:
    radial-gradient(circle at 35% 30%, rgba(255,255,255,0.94), transparent 18%),
    radial-gradient(circle at center, color-mix(in srgb, var(--reactor-color) 45%, white) 0%, color-mix(in srgb, var(--reactor-color) 30%, transparent) 38%, rgba(8,10,20,0.94) 78%);
  border: 1px solid color-mix(in srgb, var(--reactor-color) 58%, rgba(255,255,255,0.08));
  box-shadow:
    0 0 34px color-mix(in srgb, var(--reactor-color) 34%, transparent),
    0 0 70px color-mix(in srgb, var(--reactor-color) 18%, transparent),
    inset 0 0 18px rgba(255,255,255,0.08);
  animation: reactorCorePulse 2.8s ease-in-out infinite;
}
.score-reactor-core::before {
  content: "";
  position: absolute;
  inset: -16px;
  border-radius: 50%;
  background: radial-gradient(circle, color-mix(in srgb, var(--reactor-color) 18%, transparent), transparent 70%);
  filter: blur(12px);
}
@media (max-width: 900px) {
  .constellation-grid {
    grid-template-columns: 1fr;
  }
  .constellation-stage {
    min-height: 320px;
  }
}
`;

/* ─── Background ──────────────────────────────────────────────────────── */
function CursedBackground() {
  return (
    <div style={{position:"fixed",inset:0,zIndex:0,overflow:"hidden",pointerEvents:"none"}}>
      <div style={{position:"absolute",inset:0,background:"radial-gradient(ellipse 80% 60% at 15% 40%, rgba(76,29,149,0.18) 0%, transparent 55%), radial-gradient(ellipse 70% 50% at 85% 15%, rgba(14,60,130,0.14) 0%, transparent 50%), var(--bg)"}}/>
      <div style={{position:"absolute",inset:0,backgroundImage:"radial-gradient(rgba(124,92,252,0.1) 1px, transparent 1px)",backgroundSize:"30px 30px"}}/>
      <div className="jjk-orb jjk-orb-1"/>
      <div className="jjk-orb jjk-orb-2"/>
      <div className="jjk-orb jjk-orb-3"/>
      <div style={{position:"absolute",inset:0,background:"radial-gradient(ellipse 100% 100% at 50% 50%, transparent 30%, rgba(6,6,11,0.75) 100%)"}}/>
    </div>
  );
}

/* ─── Sigil SVG ───────────────────────────────────────────────────────── */
function CursedSigil({size=36,color="var(--accent)"}) {
  return (
    <svg className="sigil-icon" width={size} height={size} viewBox="0 0 36 36" fill="none">
      <polygon points="18,1 34,9.5 34,26.5 18,35 2,26.5 2,9.5" stroke={color} strokeWidth="1.2" opacity="0.9"/>
      <polygon points="18,7 28,12.5 28,23.5 18,29 8,23.5 8,12.5" stroke={color} strokeWidth="0.7" opacity="0.45"/>
      <line x1="18" y1="1" x2="18" y2="35" stroke={color} strokeWidth="0.5" opacity="0.25"/>
      <line x1="2" y1="9.5" x2="34" y2="26.5" stroke={color} strokeWidth="0.5" opacity="0.2"/>
      <line x1="34" y1="9.5" x2="2" y2="26.5" stroke={color} strokeWidth="0.5" opacity="0.2"/>
      <circle cx="18" cy="18" r="3.5" fill={color} opacity="0.85"/>
      <circle cx="18" cy="18" r="6" stroke={color} strokeWidth="0.6" opacity="0.3"/>
    </svg>
  );
}

/* ─── Score Aura ──────────────────────────────────────────────────────── */
function ScoreAura({pct}) {
  const {t,c} = scoreLabelDetails(pct);
  return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:14}}>
      <div className="score-reactor" style={{"--reactor-color":c}}>
        <div className="score-reactor-ring"/>
        <div className="score-reactor-ring-2"/>
        <div className="score-reactor-core"/>
        <div style={{
          position:"absolute",top:"50%",left:"50%",transform:"translate(-50%, -50%)",
          zIndex:2,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"
        }}>
          <span style={{fontFamily:"var(--font-display)",fontSize:54,lineHeight:0.92,color:c,textShadow:`0 0 18px ${c}`}}>
            {pct}
          </span>
          <span style={{fontSize:12,color:c,opacity:0.75,letterSpacing:2,marginTop:4}}>%</span>
        </div>
      </div>
      <div style={{textAlign:"center"}}>
        <div style={{fontFamily:"var(--font-display)",fontSize:13,letterSpacing:3,color:c,textShadow:`0 0 12px ${c}`}}>{t}</div>
      </div>
    </div>
  );
}

/* ─── Cursed Tag ──────────────────────────────────────────────────────── */
function CTag({word,variant="neutral"}) {
  const cls = variant==="shared"?"ctag ctag-shared":variant==="unique"?"ctag ctag-unique":"ctag ctag-neutral";
  return <span className={cls}>{word}</span>;
}

function KeywordConstellation({ shared, onlyA, onlyB, labelA, labelB }) {
  const sharedNodes = shared.slice(0, 5).map((word, index) => {
    const angle = (index / Math.max(shared.slice(0, 5).length, 1)) * Math.PI * 2 - Math.PI / 2;
    const radius = 86;
    return {
      word,
      cls: "constellation-node constellation-node-shared",
      left: `calc(50% + ${Math.cos(angle) * radius}px)`,
      top: `calc(50% + ${Math.sin(angle) * radius}px)`,
      delay: `${index * 0.2}s`,
    };
  });

  const aNodes = onlyA.slice(0, 4).map((word, index) => ({
    word,
    cls: "constellation-node constellation-node-alpha",
    left: `${14 + (index % 2) * 12}%`,
    top: `${18 + index * 16}%`,
    delay: `${index * 0.18}s`,
  }));

  const bNodes = onlyB.slice(0, 4).map((word, index) => ({
    word,
    cls: "constellation-node constellation-node-beta",
    left: `${70 + (index % 2) * 8}%`,
    top: `${18 + index * 16}%`,
    delay: `${index * 0.18 + 0.12}s`,
  }));

  const nodes = [...sharedNodes, ...aNodes, ...bNodes];

  return (
    <div className="constellation-shell" style={{marginBottom:14}}>
      <Divider label="Keyword Constellation"/>
      <div className="constellation-grid">
        <div className="constellation-stage">
          <div className="constellation-ring" style={{width:140,height:140}}/>
          <div className="constellation-ring" style={{width:220,height:220}}/>
          <div className="constellation-ring" style={{width:300,height:300, borderStyle:"solid", opacity:0.35}}/>
          <div className="constellation-core"/>

          <div style={{position:"absolute",left:"12%",top:"10%",fontSize:10,letterSpacing:2,color:"var(--accent)",fontWeight:700}}>
            {labelA.toUpperCase()}
          </div>
          <div style={{position:"absolute",right:"12%",top:"10%",fontSize:10,letterSpacing:2,color:"var(--blue)",fontWeight:700}}>
            {labelB.toUpperCase()}
          </div>
          <div style={{position:"absolute",left:"50%",top:"50%",transform:"translate(-50%, -50%)",fontSize:9,letterSpacing:2,color:"var(--text-faint)",fontWeight:700}}>
            SHARED CORE
          </div>

          {nodes.map((node)=>(
            <div
              key={`${node.cls}-${node.word}`}
              className={node.cls}
              style={{left:node.left,top:node.top,animationDelay:node.delay}}
            >
              {node.word}
            </div>
          ))}
        </div>

        <div style={{display:"flex",flexDirection:"column",gap:14}}>
          <div>
            <div className="section-label" style={{marginBottom:10}}>Constellation Readout</div>
            <p style={{fontSize:12.5,color:"var(--text-muted)",lineHeight:1.8}}>
              Shared terms cluster at the center as the active resonance core. Unique vocabulary for each scroll drifts to opposing flanks, showing where the techniques align and where their domains split apart.
            </p>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr",gap:10}}>
            <div className="stat-block">
              <div style={{fontSize:9,fontWeight:700,letterSpacing:2,color:"var(--green)",textTransform:"uppercase",marginBottom:5}}>Shared Resonance</div>
              <div style={{fontSize:13,color:"var(--text-muted)",fontWeight:500}}>{shared.length} bound terms orbiting the core</div>
            </div>
            <div className="stat-block">
              <div style={{fontSize:9,fontWeight:700,letterSpacing:2,color:"var(--accent)",textTransform:"uppercase",marginBottom:5}}>{labelA}</div>
              <div style={{fontSize:13,color:"var(--text-muted)",fontWeight:500}}>{onlyA.length} unique terms on the left flank</div>
            </div>
            <div className="stat-block">
              <div style={{fontSize:9,fontWeight:700,letterSpacing:2,color:"var(--blue)",textTransform:"uppercase",marginBottom:5}}>{labelB}</div>
              <div style={{fontSize:13,color:"var(--text-muted)",fontWeight:500}}>{onlyB.length} unique terms on the right flank</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Scroll Panel (Upload) ───────────────────────────────────────────── */
function ScrollPanel({label,accent,value,fileName,fileType,onChange,onFile}) {
  const [drag,setDrag] = useState(false);
  const [busy,setBusy] = useState(false);
  const [errMsg,setErrMsg] = useState("");
  const ref = useRef();
  const accentVar = accent==="a" ? "var(--accent)" : "var(--blue)";

  const handle = useCallback(async f=>{
    if(!f) return;
    setErrMsg(""); setBusy(true);
    try { const t=await extractFile(f); onFile(t,f.name,f.name.split(".").pop().toLowerCase()); }
    catch(e) { setErrMsg(e.message); }
    setBusy(false);
  },[onFile]);

  const onDrop = useCallback(e=>{
    e.preventDefault(); setDrag(false);
    const f=e.dataTransfer.files[0]; if(f) handle(f);
  },[handle]);

  return (
    <div className="cursed-panel holo-panel" style={{padding:"22px 22px 20px","--panel-accent":accentVar}}>
      <div className="holo-topbar"/>

      {/* Header */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14}}>
        <div className="holo-badge">
          <div style={{width:5,height:5,background:accentVar,borderRadius:"50%",boxShadow:`0 0 8px ${accentVar}`}}/>
          <span style={{fontFamily:"var(--font-display)",fontSize:15,letterSpacing:3,color:accentVar,textShadow:`0 0 10px ${accentVar}60`}}>
            {label.toUpperCase()}
          </span>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <div className="panel-status">{fileName ? "Bound" : "Awaiting"}</div>
          {fileName && (
          <div style={{display:"flex",alignItems:"center",gap:7}}>
            <span style={{fontSize:10,color:"var(--text-faint)",maxWidth:130,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",letterSpacing:0.3}}>
              {fileName}
            </span>
            <button onClick={()=>{onFile("","","");setErrMsg("");}} style={{
              background:"rgba(255,255,255,0.05)",border:"1px solid var(--border-mid)",
              color:"var(--text-faint)",borderRadius:3,width:18,height:18,
              cursor:"pointer",fontSize:12,lineHeight:"16px",padding:0,
              display:"flex",alignItems:"center",justifyContent:"center",
              transition:"background 0.15s, color 0.15s"
            }}>×</button>
          </div>
          )}
        </div>
      </div>

      {/* Drop zone */}
      <div className={`drop-zone${drag?" dragging":""}`}
        style={{marginBottom:12}}
        onClick={()=>ref.current.click()}
        onDragOver={e=>{e.preventDefault();setDrag(true);}}
        onDragLeave={()=>setDrag(false)}
        onDrop={onDrop}>
        <input ref={ref} type="file"
          accept=".txt,.pdf,.docx,.md,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff"
          style={{display:"none"}}
          onChange={e=>{const f=e.target.files[0];if(f) handle(f);e.target.value="";}}/>
        <div className="scan-line"/>
        {busy ? (
          <div style={{display:"flex",alignItems:"center",justifyContent:"center",gap:10,padding:"6px 0"}}>
            <svg width="14" height="14" viewBox="0 0 14 14" style={{animation:"spin .7s linear infinite",flexShrink:0}}>
              <circle cx="7" cy="7" r="5.5" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="2"/>
              <path d="M7 1.5 A5.5 5.5 0 0 1 12.5 7" fill="none" stroke={accentVar} strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <span style={{fontSize:11,color:"var(--text-muted)",letterSpacing:1}}>EXTRACTING CURSED TEXT…</span>
          </div>
        ) : (
          <>
            <div style={{fontSize:11,fontWeight:600,letterSpacing:1.5,color:"var(--text-muted)",marginBottom:4}}>
              {fileName ? "DROP TO REPLACE SCROLL" : "DROP SCROLL OR CLICK TO BIND"}
            </div>
            <div style={{fontSize:10,color:"var(--text-faint)",letterSpacing:0.8}}>PDF · DOCX · TXT · JPG · PNG</div>
          </>
        )}
      </div>

      {errMsg && (
        <div style={{
          fontSize:11,color:"var(--crimson)",marginBottom:10,
          padding:"8px 12px",background:"var(--crimson-dim)",
          borderRadius:4,border:"1px solid rgba(255,40,85,0.2)",letterSpacing:0.3
        }}>{errMsg}</div>
      )}

      <textarea className="cursed-textarea" value={value} rows={10}
        onChange={e=>onChange(e.target.value)}
        placeholder={`Inscribe ${label.toLowerCase()} text, or bind a scroll above…`}
        style={{caretColor:accentVar}}
        onFocus={e=>{e.target.style.borderColor=accentVar;e.target.style.boxShadow=`0 0 0 3px ${accentVar}18, 0 0 20px ${accentVar}10`;}}
        onBlur={e=>{e.target.style.borderColor="rgba(255,255,255,0.09)";e.target.style.boxShadow="none";}}
      />
    </div>
  );
}

/* ─── Analysis Card ───────────────────────────────────────────────────── */
function AnalysisCard({icon,label,text,glowColor}) {
  return (
    <div className="analysis-card" style={{"--card-glow":glowColor}}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
        <div style={{width:28,height:28,borderRadius:4,background:glowColor?.replace("0.15","0.1")||"var(--accent-dim)",
          display:"flex",alignItems:"center",justifyContent:"center",
          boxShadow:`0 0 12px ${glowColor||"rgba(124,92,252,0.2)"}`,fontSize:14}}>
          {icon}
        </div>
        <span style={{fontFamily:"var(--font-display)",fontSize:14,letterSpacing:2.5,
          color:glowColor?.includes("0,232")?"var(--green)":
                glowColor?.includes("255,171")?"var(--gold)":"var(--accent)"}}>
          {label.toUpperCase()}
        </span>
      </div>
      <p style={{fontSize:12.5,color:"var(--text-muted)",lineHeight:1.8}}>{text}</p>
    </div>
  );
}

/* ─── Divider ─────────────────────────────────────────────────────────── */
function Divider({label}) {
  return (
    <div style={{display:"flex",alignItems:"center",gap:14,margin:"32px 0 22px"}}>
      <div style={{flex:1,height:"1px",background:"linear-gradient(90deg, transparent, var(--border-mid))"}}/>
      <span className="section-label" style={{whiteSpace:"nowrap"}}>{label}</span>
      <div style={{flex:1,height:"1px",background:"linear-gradient(90deg, var(--border-mid), transparent)"}}/>
    </div>
  );
}

/* ─── Main App ────────────────────────────────────────────────────────── */
function ReactiveOrbHero({modeLabel}) {
  const shellRef = useRef(null);
  const canvasWrapRef = useRef(null);
  const frameRef = useRef(0);
  const currentRef = useRef({x:0,y:0});
  const targetRef = useRef({x:0,y:0});

  useEffect(()=>{
    if (!canvasWrapRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    camera.position.set(0, 0.2, 5.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    canvasWrapRef.current.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    const coreGeometry = new THREE.IcosahedronGeometry(1.15, 12);
    const coreMaterial = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color("#6ea8ff"),
      emissive: new THREE.Color("#4b2ad6"),
      emissiveIntensity: 1.55,
      roughness: 0.18,
      metalness: 0.12,
      transmission: 0.24,
      transparent: true,
      opacity: 0.96,
      thickness: 1.2,
      clearcoat: 1,
      clearcoatRoughness: 0.18,
      iridescence: 0.45,
    });
    const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    group.add(coreMesh);

    const wireMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color("#8fe7ff"),
      wireframe: true,
      transparent: true,
      opacity: 0.28,
    });
    const wireMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1.32, 4), wireMaterial);
    group.add(wireMesh);

    const haloMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color("#7c5cfc"),
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const haloMesh = new THREE.Mesh(new THREE.TorusGeometry(1.95, 0.03, 24, 200), haloMaterial);
    haloMesh.rotation.x = Math.PI / 2;
    group.add(haloMesh);

    const haloMesh2 = new THREE.Mesh(new THREE.TorusGeometry(1.48, 0.025, 20, 160), haloMaterial.clone());
    haloMesh2.material.color = new THREE.Color("#00d4ff");
    haloMesh2.material.opacity = 0.14;
    haloMesh2.rotation.set(Math.PI / 2.6, 0, 0.45);
    group.add(haloMesh2);

    const ambient = new THREE.AmbientLight("#7c5cfc", 1.15);
    scene.add(ambient);

    const key = new THREE.PointLight("#7c5cfc", 26, 18, 2);
    key.position.set(2.8, 2.4, 3.6);
    scene.add(key);

    const fill = new THREE.PointLight("#00d4ff", 18, 16, 2);
    fill.position.set(-3.2, -1.2, 3.4);
    scene.add(fill);

    const rim = new THREE.PointLight("#ffffff", 8, 14, 2);
    rim.position.set(0, 0, 4.8);
    scene.add(rim);

    const resize = ()=>{
      const node = canvasWrapRef.current;
      if (!node) return;
      const width = node.clientWidth || 1;
      const height = node.clientHeight || 1;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvasWrapRef.current);

    const clock = new THREE.Clock();
    const animate = ()=>{
      const current = currentRef.current;
      const target = targetRef.current;
      current.x += (target.x - current.x) * 0.12;
      current.y += (target.y - current.y) * 0.12;

      const rotateX = -(current.y * 18);
      const rotateY = current.x * 22;
      const glowX = 50 + current.x * 30;
      const glowY = 40 + current.y * 25;
      const elapsed = clock.getElapsedTime();

      if (shellRef.current) {
        shellRef.current.style.setProperty("--orb-glow", `radial-gradient(circle at ${glowX}% ${glowY}%, rgba(124,92,252,0.22), transparent 28%)`);
      }

      group.rotation.x = THREE.MathUtils.degToRad(rotateX * 0.28) + Math.sin(elapsed * 0.45) * 0.05;
      group.rotation.y = THREE.MathUtils.degToRad(rotateY * 0.34) + elapsed * 0.28;
      coreMesh.rotation.x += 0.0025;
      coreMesh.rotation.y += 0.004;
      wireMesh.rotation.x -= 0.0028;
      wireMesh.rotation.z += 0.0035;
      haloMesh.rotation.z += 0.0038;
      haloMesh2.rotation.z -= 0.0042;
      coreMesh.position.x = current.x * 0.12;
      coreMesh.position.y = -current.y * 0.08;
      wireMesh.position.copy(coreMesh.position);
      haloMesh.position.x = current.x * 0.08;
      haloMesh.position.y = -current.y * 0.05;
      haloMesh2.position.copy(haloMesh.position);
      coreMaterial.emissiveIntensity = 1.35 + Math.max(Math.abs(current.x), Math.abs(current.y)) * 0.8 + Math.sin(elapsed * 1.8) * 0.08;

      renderer.render(scene, camera);

      frameRef.current = requestAnimationFrame(animate);
    };

    frameRef.current = requestAnimationFrame(animate);
    return ()=>{
      cancelAnimationFrame(frameRef.current);
      resizeObserver.disconnect();
      renderer.dispose();
      coreGeometry.dispose();
      coreMaterial.dispose();
      wireMesh.geometry.dispose();
      wireMaterial.dispose();
      haloMesh.geometry.dispose();
      haloMaterial.dispose();
      haloMesh2.geometry.dispose();
      haloMesh2.material.dispose();
      canvasWrapRef.current?.contains(renderer.domElement) && canvasWrapRef.current.removeChild(renderer.domElement);
    };
  },[]);

  const handleMove = useCallback((event)=>{
    const rect = event.currentTarget.getBoundingClientRect();
    targetRef.current = {
      x: (event.clientX - rect.left) / rect.width - 0.5,
      y: (event.clientY - rect.top) / rect.height - 0.5,
    };
  },[]);

  const handleLeave = useCallback(()=>{
    targetRef.current = {x:0,y:0};
  },[]);

  return (
    <section
      ref={shellRef}
      className="hero-shell"
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
      style={{
        "--orb-glow":"radial-gradient(circle at 50% 40%, rgba(124,92,252,0.18), transparent 28%)",
        marginBottom:28,padding:"30px 32px",animation:"fadeUp 0.5s 0.03s both"
      }}
    >
      <div style={{
        position:"absolute",inset:0,pointerEvents:"none",
        background:"var(--orb-glow)",
        opacity:0.8,
        transition:"opacity 0.25s ease"
      }}/>

      <div className="hero-grid" style={{position:"relative",zIndex:1}}>
        <div>
          <div style={{display:"inline-flex",alignItems:"center",gap:10,marginBottom:18}}>
            <div style={{width:7,height:7,borderRadius:"50%",background:"var(--accent)",boxShadow:"0 0 14px var(--accent)"}}/>
            <span className="section-label" style={{color:"var(--accent)"}}>Curse Core Interface</span>
          </div>
          <h2 style={{
            fontFamily:"var(--font-display)",
            fontSize:"clamp(34px, 5vw, 58px)",
            lineHeight:0.95,
            letterSpacing:2,
            color:"var(--text)",
            marginBottom:14,
            textShadow:"0 0 30px rgba(124,92,252,0.22)"
          }}>
            REACTIVE DOMAIN
            <br/>
            SPHERE
          </h2>
          <p style={{maxWidth:520,fontSize:13,lineHeight:1.85,color:"var(--text-muted)",marginBottom:22}}>
            The central curse core tracks movement like a living domain anchor. It gives the interface a true 3D focal point while the {modeLabel.toLowerCase()} workflow remains readable beneath it.
          </p>
          <div style={{display:"flex",flexWrap:"wrap",gap:10}}>
            {["Cursor-tracked tilt","Layered 3D glow","Domain expansion mood"].map(item=>(
              <span key={item} className="ctag ctag-neutral" style={{background:"rgba(255,255,255,0.02)"}}>
                {item}
              </span>
            ))}
          </div>
        </div>

        <div className="hero-orb-stage">
          <div ref={canvasWrapRef} className="hero-webgl-canvas"/>
          <div className="hero-orb-ring"/>
          <div className="hero-orb-ring-2"/>
          <div className="hero-orb-floor"/>
          {[
            {text:"CURSE", top:"12%", left:"15%", delay:"0s"},
            {text:"VECTOR", top:"22%", right:"10%", delay:"0.8s"},
            {text:"DOMAIN", bottom:"18%", left:"12%", delay:"1.3s"},
            {text:"SYNC", bottom:"12%", right:"16%", delay:"2s"},
          ].map(rune=>(
            <span key={rune.text} className="hero-rune" style={{...rune,animationDelay:rune.delay}}>
              {rune.text}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [mode,setMode]     = useState("resume");
  const [docA,setDocA]     = useState(SAMPLES.resume.a);
  const [docB,setDocB]     = useState(SAMPLES.resume.b);
  const [metaA,setMetaA]   = useState({name:"",ext:""});
  const [metaB,setMetaB]   = useState({name:"",ext:""});
  const [result,setResult] = useState(null);
  const [loading,setLoading]= useState(false);
  const [phase,setPhase]   = useState("");
  const [err,setErr]       = useState("");
  const [expanding,setExpanding] = useState(false);
  const M = MODES.find(m=>m.id===mode);

  useEffect(()=>{
    const style = document.createElement("style");
    style.id = "jjk-theme";
    style.textContent = GLOBAL_CSS;
    document.head.appendChild(style);
    document.title = "DOMAIN EXPANSION — Cursed Analysis Engine";
    return ()=>{ style.remove(); };
  },[]);

  const switchMode = id=>{
    setMode(id); setDocA(SAMPLES[id]?.a||""); setDocB(SAMPLES[id]?.b||"");
    setMetaA({name:"",ext:""}); setMetaB({name:"",ext:""});
    setResult(null); setErr("");
    setExpanding(false);
  };

  const analyze = useCallback(async()=>{
    if(!docA.trim()||!docB.trim()){setErr("Both scrolls must be inscribed before domain expansion.");return;}
    setErr(""); setLoading(true); setResult(null); setExpanding(true);
    try {
      setPhase("COMPUTING CURSED ENERGY VECTORS");
      await new Promise(r=>setTimeout(r,350));
      const [vA,vB] = tfidf([docA,docB]);
      setPhase("MEASURING TECHNIQUE RESONANCE");
      await new Promise(r=>setTimeout(r,250));
      const score = cosine(vA,vB);
      const tA=top(vA,10), tB=top(vB,10);
      const sA=new Set(tA), sB=new Set(tB);
      const overlap = unique(tA.filter(w=>sB.has(w)));
      const onlyA   = unique(tA.filter(w=>!sB.has(w)));
      const onlyB   = unique(tB.filter(w=>!sA.has(w)));
      setPhase("CONSTRUCTING BINDING VOW MATRIX");
      const ai = buildLocalAnalysis({ pct:Math.round(score*100), overlap, onlyA, onlyB, modeLabel:M.label });
      setResult({
        score, pct:Math.round(score*100), topA:tA, topB:tB,
        overlap, onlyA, onlyB, ai,
        tokA:tokenize(docA).length, tokB:tokenize(docB).length,
        vocab:new Set([...tokenize(docA),...tokenize(docB)]).size
      });
      setPhase("DOMAIN EXPANSION STABILIZING");
      await new Promise(r=>setTimeout(r,320));
    } catch(e) { setErr(e instanceof Error ? e.message : "Domain expansion failed. Reinscribe the scrolls and retry."); }
    setLoading(false); setPhase(""); setExpanding(false);
  },[docA,docB,M]);

  const pct = result?.pct??0;
  const {c:scoreColor} = result ? scoreLabelDetails(pct) : {c:"var(--accent)"};

  return (
    <div style={{fontFamily:"var(--font-ui)",background:"var(--bg)",color:"var(--text)",minHeight:"100vh",position:"relative"}}>
      <CursedBackground/>
      {expanding && (
        <div className="domain-expansion-overlay" aria-hidden="true">
          <div className="domain-expansion-sweep"/>
          <div className="domain-expansion-flash"/>
          <div className="domain-expansion-runes">DOMAIN EXPANSION</div>
        </div>
      )}

      {/* ── HEADER ── */}
      <header style={{
        position:"relative",zIndex:10,
        borderBottom:"1px solid var(--border)",
        background:"rgba(6,6,11,0.85)",
        backdropFilter:"blur(20px)",
        WebkitBackdropFilter:"blur(20px)",
      }}>
        <div style={{maxWidth:1080,margin:"0 auto",padding:"16px 32px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:16}}>
          {/* Brand */}
          <div style={{display:"flex",alignItems:"center",gap:14}}>
            <CursedSigil size={36}/>
            <div>
              <div style={{
                fontFamily:"var(--font-display)",
                fontSize:22,letterSpacing:4,
                color:"var(--text)",
                lineHeight:1,
                animation:"titleReveal 0.8s ease both",
                textShadow:"0 0 30px rgba(124,92,252,0.3)"
              }}>
                CURSED ANALYSIS ENGINE
              </div>
              <div style={{fontSize:9,letterSpacing:3,color:"var(--text-faint)",marginTop:3,fontWeight:600}}>
                DOMAIN EXPANSION PROTOCOL · TF-IDF · BINDING VOW MATRIX
              </div>
            </div>
          </div>

          {/* Mode Tabs */}
          <div style={{display:"flex",gap:4,padding:"5px",background:"rgba(0,0,0,0.4)",borderRadius:6,border:"1px solid var(--border)"}}>
            {MODES.map(m=>(
              <button key={m.id} className={`mode-tab${mode===m.id?" active":""}`}
                onClick={()=>switchMode(m.id)}>
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── BODY ── */}
      <main style={{position:"relative",zIndex:10,maxWidth:1080,margin:"0 auto",padding:"32px 32px 64px"}}>

        {/* ── Mode descriptor ── */}
        <div style={{marginBottom:24,animation:"fadeUp 0.5s both"}}>
          <div style={{display:"inline-flex",alignItems:"center",gap:8,
            padding:"6px 14px",
            background:"var(--accent-dim)",
            border:"1px solid rgba(124,92,252,0.22)",
            borderRadius:4}}>
            <div style={{width:4,height:4,borderRadius:"50%",background:"var(--accent)",boxShadow:"0 0 8px var(--accent)"}}/>
            <span style={{fontSize:10,fontWeight:700,letterSpacing:2,color:"var(--accent)"}}>
              ACTIVE PROTOCOL: {M.label.toUpperCase()}
            </span>
          </div>
        </div>

        <ReactiveOrbHero modeLabel={M.label}/>

        {/* ── Scroll Panels ── */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:24,marginBottom:24,animation:"fadeUp 0.5s 0.05s both"}}>
          <ScrollPanel label={M.a} accent="a" value={docA} fileName={metaA.name} fileType={metaA.ext}
            onChange={v=>{setDocA(v);setResult(null);}}
            onFile={(t,n,e)=>{if(t){setDocA(t);setMetaA({name:n,ext:e});}else setMetaA({name:"",ext:""});setResult(null);}}/>
          <ScrollPanel label={M.b} accent="b" value={docB} fileName={metaB.name} fileType={metaB.ext}
            onChange={v=>{setDocB(v);setResult(null);}}
            onFile={(t,n,e)=>{if(t){setDocB(t);setMetaB({name:n,ext:e});}else setMetaB({name:"",ext:""});setResult(null);}}/>
        </div>

        {/* ── Error ── */}
        {err && (
          <div style={{
            marginBottom:20,padding:"12px 16px",
            background:"var(--crimson-dim)",
            border:"1px solid rgba(255,40,85,0.25)",
            borderRadius:5,fontSize:12,color:"var(--crimson)",
            letterSpacing:0.4,animation:"fadeIn 0.3s both"
          }}>
            ⚠ {err}
          </div>
        )}

        {/* ── Action Row ── */}
        <div style={{display:"flex",alignItems:"center",gap:20,marginBottom:36,animation:"fadeUp 0.5s 0.1s both"}}>
          <button className="domain-btn" onClick={analyze} disabled={loading}>
            {loading ? (
              <span style={{display:"flex",alignItems:"center",gap:10,justifyContent:"center"}}>
                <svg width="16" height="16" viewBox="0 0 16 16" style={{animation:"spin .65s linear infinite",flexShrink:0}}>
                  <circle cx="8" cy="8" r="6.5" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="2"/>
                  <path d="M8 1.5 A6.5 6.5 0 0 1 14.5 8" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                ANALYZING…
              </span>
            ) : "EXPAND DOMAIN"}
          </button>
          {loading && phase && (
            <span style={{fontSize:10,color:"var(--text-faint)",letterSpacing:2,fontWeight:700,animation:"fadeIn .3s both"}}>
              {phase}
            </span>
          )}
          {!loading && !result && (
            <span style={{fontSize:11,color:"var(--text-faint)",letterSpacing:0.5}}>
              Bind PDF, DOCX, TXT scrolls — or inscribe directly
            </span>
          )}
        </div>

        {/* ── RESULTS ── */}
        {result && (
          <div className="domain-results-reveal">

            {/* ── Score + Verdict ── */}
            <div style={{
              display:"grid",gridTemplateColumns:"180px 1fr",gap:36,
              padding:"32px",
              background:"var(--bg-panel)",
              border:"1px solid var(--border)",
              borderRadius:8,
              marginBottom:20,
              position:"relative",overflow:"hidden",
              boxShadow:`0 0 0 1px rgba(124,92,252,0.1), 0 20px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)`
            }}>
              {/* Corner accent */}
              <div style={{position:"absolute",top:0,left:0,right:0,height:"1px",
                background:`linear-gradient(90deg, ${scoreColor}60, transparent 60%)`}}/>
              <div style={{position:"absolute",top:0,left:0,width:"1px",height:"100%",
                background:`linear-gradient(180deg, ${scoreColor}40, transparent 60%)`}}/>

              <ScoreAura pct={pct}/>

              <div style={{display:"flex",flexDirection:"column",justifyContent:"center",gap:16}}>
                <div>
                  <div className="section-label" style={{marginBottom:8}}>DOMAIN VERDICT</div>
                  <p style={{
                    fontFamily:"var(--font-display)",
                    fontSize:18,letterSpacing:1,
                    color:"var(--text)",lineHeight:1.6,
                    textShadow:"0 0 20px rgba(124,92,252,0.2)"
                  }}>
                    "{result.ai.verdict}"
                  </p>
                </div>
                <div style={{display:"flex",gap:8,alignItems:"center"}}>
                  <div style={{
                    padding:"5px 12px",
                    background:"rgba(0,0,0,0.4)",
                    border:`1px solid ${scoreColor}40`,
                    borderRadius:3,
                    fontSize:10,fontWeight:700,letterSpacing:2,color:scoreColor,
                    textShadow:`0 0 8px ${scoreColor}`
                  }}>
                    {result.ai.match_label}
                  </div>
                  <div style={{
                    padding:"5px 12px",
                    background:"rgba(0,0,0,0.3)",
                    border:"1px solid var(--border-mid)",
                    borderRadius:3,
                    fontSize:10,fontWeight:700,letterSpacing:2,color:"var(--text-muted)"
                  }}>
                    {result.ai.confidence} CONFIDENCE
                  </div>
                </div>
              </div>
            </div>

            <KeywordConstellation
              shared={result.overlap}
              onlyA={result.onlyA}
              onlyB={result.onlyB}
              labelA={M.a}
              labelB={M.b}
            />

            {/* ── Analysis Triptych ── */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14,marginBottom:20}}>
              <AnalysisCard
                icon="⚡"
                label="Binding Strength"
                text={result.ai.strength}
                glowColor="rgba(0,232,135,0.15)"
              />
              <AnalysisCard
                icon="⚔"
                label="Energy Gaps"
                text={result.ai.gap}
                glowColor="rgba(255,171,48,0.15)"
              />
              <AnalysisCard
                icon="◈"
                label="Cursed Directive"
                text={result.ai.recommendation}
                glowColor="rgba(124,92,252,0.15)"
              />
            </div>

            {/* ── Cursed Technique Index ── */}
            <div style={{
              padding:"24px",
              background:"var(--bg-panel)",
              border:"1px solid var(--border)",
              borderRadius:8,
              marginBottom:14,
            }}>
              <Divider label="Cursed Technique Index"/>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:24}}>
                {[
                  {label:M.a, words:result.topA, other:new Set(result.topB)},
                  {label:M.b, words:result.topB, other:new Set(result.topA)},
                ].map(({label,words,other})=>(
                  <div key={label}>
                    <div style={{fontSize:10,fontWeight:700,letterSpacing:2,color:"var(--text-faint)",textTransform:"uppercase",marginBottom:10}}>
                      {label}
                    </div>
                    <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
                      {words.map(w=>(
                        <CTag key={w} word={w} variant={other.has(w)?"shared":"neutral"}/>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div style={{marginTop:14,display:"flex",alignItems:"center",gap:8}}>
                <CTag word="shared term" variant="shared"/>
                <span style={{fontSize:10,color:"var(--text-faint)",letterSpacing:0.3}}>= resonates in both scrolls</span>
              </div>
            </div>

            {/* ── Binding Vow Matrix ── */}
            <div style={{
              padding:"24px",
              background:"var(--bg-panel)",
              border:"1px solid var(--border)",
              borderRadius:8,
              marginBottom:14,
            }}>
              <Divider label="Binding Vow Matrix"/>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:20}}>
                {[
                  {label:`Bound Terms (${result.overlap.length})`,    words:result.overlap, variant:"shared"},
                  {label:`Only in ${M.a} (${result.onlyA.length})`,   words:result.onlyA,   variant:"unique"},
                  {label:`Only in ${M.b} (${result.onlyB.length})`,   words:result.onlyB,   variant:"unique"},
                ].map(({label,words,variant})=>(
                  <div key={label}>
                    <div style={{fontSize:10,fontWeight:700,letterSpacing:1.5,color:"var(--text-faint)",textTransform:"uppercase",marginBottom:10}}>
                      {label}
                    </div>
                    <div style={{display:"flex",flexWrap:"wrap",gap:5}}>
                      {words.length ? words.map(w=><CTag key={w} word={w} variant={variant}/>) :
                        <span style={{fontSize:11,color:"var(--text-faint)",fontStyle:"italic"}}>None detected</span>
                      }
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Domain Parameters ── */}
            <div style={{
              padding:"24px",
              background:"var(--bg-panel)",
              border:"1px solid var(--border)",
              borderRadius:8,
            }}>
              <Divider label="Domain Parameters"/>
              <div style={{display:"flex",flexWrap:"wrap",gap:10}}>
                {[
                  ["Algorithm",       "TF-IDF · Cosine Similarity"],
                  ["Scroll I tokens", result.tokA.toLocaleString()],
                  ["Scroll II tokens",result.tokB.toLocaleString()],
                  ["Shared vocab",    result.vocab.toLocaleString()+" terms"],
                  ["Raw cosine",      result.score.toFixed(6)],
                  ["Scale",           "0.000 → 1.000"],
                ].map(([k,v])=>(
                  <div key={k} className="stat-block" style={{minWidth:140,flex:"1 0 auto"}}>
                    <div style={{fontSize:9,fontWeight:700,letterSpacing:2,color:"var(--text-faint)",textTransform:"uppercase",marginBottom:5}}>{k}</div>
                    <div style={{fontSize:13,color:"var(--text-muted)",fontWeight:500}}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

        {/* ── Empty state ── */}
        {!result && !loading && (
          <div style={{
            marginTop:16,
            padding:"48px 32px",
            border:"1px dashed rgba(124,92,252,0.12)",
            borderRadius:8,
            textAlign:"center",
            animation:"fadeIn 0.6s 0.2s both",
          }}>
            <CursedSigil size={48} color="rgba(124,92,252,0.2)"/>
            <div style={{
              fontFamily:"var(--font-display)",
              fontSize:16,letterSpacing:4,
              color:"var(--text-faint)",
              marginTop:16,marginBottom:8,
            }}>AWAITING DOMAIN EXPANSION</div>
            <div style={{fontSize:11,color:"var(--text-faint)",letterSpacing:0.5,opacity:0.7}}>
              Inscribe both scrolls and initiate the domain to reveal cursed resonance
            </div>
          </div>
        )}

      </main>
    </div>
  );
}
