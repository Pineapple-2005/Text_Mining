import { useState, useRef, useCallback, useEffect } from "react";
import * as mammoth from "mammoth";

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
  if (pct >= 70) return { t: "Strong Match", c: "var(--green)" };
  if (pct >= 45) return { t: "Moderate Match", c: "var(--amber)" };
  return { t: "Weak Match", c: "var(--red)" };
}

function buildLocalAnalysis({ pct, overlap, onlyA, onlyB, modeLabel }) {
  const matchLabel = scoreLabelDetails(pct).t;
  const sharedText = overlap.length
    ? `Both documents share ${overlap.slice(0, 5).join(", ")} and other terms, which supports the similarity score.`
    : "The documents share little direct vocabulary, so the score is driven mainly by broader thematic overlap rather than exact term reuse.";
  const gapText = onlyA.length || onlyB.length
    ? `Document A emphasizes ${onlyA.slice(0, 4).join(", ") || "different terminology"} while Document B emphasizes ${onlyB.slice(0, 4).join(", ") || "different terminology"}, so the content is not fully aligned.`
    : "There is no strong term-level gap between the documents.";
  const recommendationText = pct >= 70
    ? `Keep the current structure and terminology aligned for ${modeLabel.toLowerCase()}. Review any remaining wording differences before final use.`
    : `Add more shared terminology and align the key concepts for ${modeLabel.toLowerCase()}. Recheck the section structure and terminology so both documents emphasize the same ideas.`;

  return {
    verdict: `${matchLabel} for ${modeLabel.toLowerCase()}: the documents are ${pct >= 70 ? "closely aligned" : pct >= 45 ? "partially aligned" : "only loosely aligned"} based on shared vocabulary and topic overlap.`,
    match_label: matchLabel,
    strength: sharedText,
    gap: gapText,
    recommendation: recommendationText,
    confidence: pct >= 70 ? "high" : pct >= 45 ? "medium" : "low",
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
  {id:"resume",label:"Resume Matching",a:"Resume / CV",b:"Job Description"},
  {id:"research",label:"Research Comparison",a:"Abstract A",b:"Abstract B"},
  {id:"plagiarism",label:"Plagiarism Screening",a:"Original Document",b:"Submitted Document"},
  {id:"general",label:"General Analysis",a:"Document A",b:"Document B"},
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

/* ─── Score helpers ───────────────────────────────────────────────────── */
const scoreLabel = p => scoreLabelDetails(p);

/* ─── Sub-components ──────────────────────────────────────────────────── */
function ScoreBar({pct}){
  const {c}=scoreLabel(pct);
  return(
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:8}}>
        <span style={{fontSize:13,color:"var(--text-muted)",letterSpacing:.3}}>Similarity score</span>
        <span style={{fontSize:32,fontWeight:600,fontFamily:"var(--font-display)",color:"var(--text)",letterSpacing:-1}}>{pct}<span style={{fontSize:18,fontWeight:400,color:"var(--text-muted)"}}>%</span></span>
      </div>
      <div style={{height:4,background:"var(--bg-muted)",borderRadius:2,overflow:"hidden"}}>
        <div style={{height:"100%",width:`${pct}%`,background:c,borderRadius:2,transition:"width 1s cubic-bezier(.4,0,.2,1)"}}/>
      </div>
    </div>
  );
}

function Tag({word,variant="neutral"}){
  const styles={
    shared:{bg:"var(--green-bg)",color:"var(--green)",border:"var(--green-border)"},
    unique:{bg:"var(--accent-bg)",color:"var(--accent)",border:"var(--accent-border)"},
    neutral:{bg:"var(--bg-subtle)",color:"var(--text-muted)",border:"var(--border)"},
  };
  const s=styles[variant]||styles.neutral;
  return(
    <span style={{display:"inline-block",padding:"3px 10px",borderRadius:3,fontSize:12,
      background:s.bg,color:s.color,border:`1px solid ${s.border}`,letterSpacing:.2,lineHeight:"20px"}}>
      {word}
    </span>
  );
}

function UploadPanel({label,icon,value,fileName,fileType,onChange,onFile,accent}){
  const [drag,setDrag]=useState(false);
  const [busy,setBusy]=useState(false);
  const [ocrMode,setOcrMode]=useState(false);
  const [errMsg,setErrMsg]=useState("");
  const ref=useRef();

  const handle=useCallback(async f=>{
    if(!f) return;
    setErrMsg("");
    setBusy(true);
    const ext=f.name.split(".").pop().toLowerCase();
    const isImg=IMG_TYPES.has(ext);
    setOcrMode(isImg);
    try{const t=await extractFile(f); onFile(t,f.name,ext);}
    catch(e){setErrMsg(e.message); setOcrMode(false);}
    setBusy(false);
  },[onFile]);

  const onDrop=useCallback(e=>{e.preventDefault();setDrag(false);const f=e.dataTransfer.files[0];if(f)handle(f);},[handle]);

  return(
    <div style={{display:"flex",flexDirection:"column",gap:0}}>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",paddingBottom:10,borderBottom:"1px solid var(--border)",marginBottom:14}}>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <div style={{width:6,height:6,borderRadius:"50%",background:accent,flexShrink:0}}/>
          <span style={{fontSize:11,fontWeight:600,letterSpacing:1.5,color:"var(--text-muted)",textTransform:"uppercase"}}>{label}</span>
        </div>
        {fileName&&(
          <div style={{display:"flex",alignItems:"center",gap:6}}>
            {ocrMode&&<span style={{fontSize:10,padding:"2px 7px",background:"var(--accent-bg)",color:"var(--accent)",border:"1px solid var(--accent-border)",borderRadius:3,letterSpacing:.5}}>OCR</span>}
            <span style={{fontSize:11,color:"var(--text-faint)",maxWidth:140,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{fileName}</span>
            <button onClick={()=>{onFile("","","");setOcrMode(false);setErrMsg("");}} style={{background:"none",border:"none",cursor:"pointer",color:"var(--text-faint)",fontSize:14,lineHeight:1,padding:"0 2px"}}>×</button>
          </div>
        )}
      </div>

      <div onClick={()=>ref.current.click()}
        onDragOver={e=>{e.preventDefault();setDrag(true);}}
        onDragLeave={()=>setDrag(false)}
        onDrop={onDrop}
        style={{
          border:`1px dashed ${drag?"var(--accent)":"var(--border-strong)"}`,
          borderRadius:6,padding:"18px 16px",textAlign:"center",cursor:"pointer",
          background:drag?"var(--accent-bg)":"transparent",
          transition:"border-color .15s, background .15s",marginBottom:12
        }}>
        <input ref={ref} type="file" accept=".txt,.pdf,.docx,.md,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff"
          style={{display:"none"}} onChange={e=>{const f=e.target.files[0];if(f) handle(f);e.target.value="";}}/>
        {busy?(
          <div style={{display:"flex",alignItems:"center",justifyContent:"center",gap:10,padding:"4px 0"}}>
            <svg width="14" height="14" viewBox="0 0 14 14" style={{animation:"spin .7s linear infinite",flexShrink:0}}>
              <circle cx="7" cy="7" r="6" fill="none" stroke="var(--border-strong)" strokeWidth="2"/>
              <path d="M7 1 A6 6 0 0 1 13 7" fill="none" stroke={accent} strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <span style={{fontSize:12,color:"var(--text-muted)"}}>{ocrMode?"Running OCR on image...":"Extracting text..."}</span>
          </div>
        ):(
          <div>
            <div style={{fontSize:11,fontWeight:500,color:"var(--text-muted)",marginBottom:3}}>
              {fileName?"Drop a new file to replace":"Drop file or click to browse"}
            </div>
            <div style={{fontSize:11,color:"var(--text-faint)"}}>PDF · DOCX · TXT · JPG · PNG · WEBP</div>
          </div>
        )}
      </div>

      {errMsg&&<div style={{fontSize:11,color:"var(--red)",marginBottom:10,padding:"6px 10px",background:"var(--red-bg)",borderRadius:4,border:"1px solid var(--red-border)"}}>{errMsg}</div>}

      <textarea value={value} onChange={e=>onChange(e.target.value)} rows={11}
        placeholder={`Paste ${label.toLowerCase()} text, or upload a file above…`}
        style={{
          fontFamily:"var(--font-ui)",fontSize:13,lineHeight:1.75,color:"var(--text)",
          border:"1px solid var(--border)",borderRadius:6,padding:"13px 14px",
          background:"var(--bg-subtle)",resize:"vertical",width:"100%",boxSizing:"border-box",
          outline:"none",transition:"border-color .15s",caretColor:accent,
          placeholder:"var(--text-faint)"
        }}
        onFocus={e=>{e.target.style.borderColor=accent;e.target.style.background="var(--bg)";}}
        onBlur={e=>{e.target.style.borderColor="var(--border)";e.target.style.background="var(--bg-subtle)";}}/>
    </div>
  );
}

function Section({title,children,delay=0}){
  return(
    <div style={{animation:`fadeUp .4s ${delay}s both`}}>
      <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:16}}>
        <span style={{fontSize:11,fontWeight:600,letterSpacing:1.5,color:"var(--text-faint)",textTransform:"uppercase"}}>{title}</span>
        <div style={{flex:1,height:"1px",background:"var(--border)"}}/>
      </div>
      {children}
    </div>
  );
}

/* ─── Main App ────────────────────────────────────────────────────────── */
export default function App(){
  const [mode,setMode]=useState("resume");
  const [docA,setDocA]=useState(SAMPLES.resume.a);
  const [docB,setDocB]=useState(SAMPLES.resume.b);
  const [metaA,setMetaA]=useState({name:"",ext:""});
  const [metaB,setMetaB]=useState({name:"",ext:""});
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(false);
  const [phase,setPhase]=useState("");
  const [err,setErr]=useState("");
  const M=MODES.find(m=>m.id===mode);

  useEffect(()=>{
    /* Inject fonts + keyframes */
    const link1=document.createElement("link");
    link1.rel="stylesheet";
    link1.href="https://unpkg.com/@fontsource/lora@5/index.css";
    const link2=document.createElement("link");
    link2.rel="stylesheet";
    link2.href="https://unpkg.com/@fontsource/outfit@5/index.css";
    document.head.appendChild(link1);
    document.head.appendChild(link2);

    const style=document.createElement("style");
    style.textContent=`
      :root {
        --font-display:'Lora',Georgia,serif;
        --font-ui:'Outfit',system-ui,sans-serif;
        --bg:#ffffff;
        --bg-subtle:#f8f7f5;
        --bg-muted:#eeecea;
        --border:#e3e1dc;
        --border-strong:#c8c5be;
        --text:#18170f;
        --text-muted:#706d65;
        --text-faint:#a09d97;
        --accent:#1a4ed8;
        --accent-bg:#eff3ff;
        --accent-border:#bfcfff;
        --green:#166534;
        --green-bg:#f0fdf4;
        --green-border:#bbf7d0;
        --amber:#854d0e;
        --amber-bg:#fffbeb;
        --amber-border:#fde68a;
        --red:#991b1b;
        --red-bg:#fef2f2;
        --red-border:#fecaca;
      }
      @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
      @keyframes spin{to{transform:rotate(360deg)}}
      @keyframes fadeIn{from{opacity:0}to{opacity:1}}
      * { box-sizing:border-box; }
      body { background:var(--bg); }
      textarea::placeholder { color:var(--text-faint); }
      textarea::-webkit-scrollbar { width:4px; }
      textarea::-webkit-scrollbar-track { background:transparent; }
      textarea::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:2px; }
    `;
    document.head.appendChild(style);
    return()=>{
      link1.remove();
      link2.remove();
      style.remove();
    };
  },[]);

  const switchMode=id=>{
    setMode(id); setDocA(SAMPLES[id]?.a||""); setDocB(SAMPLES[id]?.b||"");
    setMetaA({name:"",ext:""}); setMetaB({name:"",ext:""});
    setResult(null); setErr("");
  };

  const analyze=useCallback(async()=>{
    if(!docA.trim()||!docB.trim()){setErr("Both document fields are required.");return;}
    setErr(""); setLoading(true); setResult(null);
    try{
      setPhase("Computing TF-IDF vectors");
      await new Promise(r=>setTimeout(r,300));
      const [vA,vB]=tfidf([docA,docB]);
      setPhase("Measuring cosine similarity");
      await new Promise(r=>setTimeout(r,200));
      const score=cosine(vA,vB);
      const tA=top(vA,10),tB=top(vB,10);
      const sA=new Set(tA),sB=new Set(tB);
      const overlap=unique(tA.filter(w=>sB.has(w)));
      const onlyA=unique(tA.filter(w=>!sB.has(w)));
      const onlyB=unique(tB.filter(w=>!sA.has(w)));

      setPhase("Generating analysis summary");
      const ai=buildLocalAnalysis({
        pct: Math.round(score*100),
        overlap,
        onlyA,
        onlyB,
        modeLabel: M.label,
      });

      setResult({
        score, pct:Math.round(score*100), topA:tA, topB:tB,
        overlap, onlyA, onlyB, ai,
        tokA:tokenize(docA).length, tokB:tokenize(docB).length,
        vocab:new Set([...tokenize(docA),...tokenize(docB)]).size
      });
    }catch(e){setErr(e instanceof Error ? e.message : "Analysis failed. Please try again.");}
    setLoading(false); setPhase("");
  },[docA,docB,M]);

  const pct=result?.pct??0;
  const {c:scoreColor}=result?scoreLabel(pct):{c:"var(--text-muted)"};

  return(
    <div style={{fontFamily:"var(--font-ui)",background:"var(--bg)",color:"var(--text)",minHeight:"100vh"}}>

      {/* ── Header ── */}
      <div style={{borderBottom:"1px solid var(--border)",padding:"20px 32px"}}>
        <div style={{maxWidth:1000,margin:"0 auto",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:16}}>
          <div>
            <h1 style={{fontFamily:"var(--font-display)",fontSize:20,fontWeight:600,margin:0,letterSpacing:-.3,color:"var(--text)"}}>
              Document Analyzer
            </h1>
            <p style={{fontSize:12,color:"var(--text-faint)",margin:"3px 0 0",letterSpacing:.3}}>
              TF-IDF · Cosine Similarity · AI Interpretation
            </p>
          </div>
          <div style={{display:"flex",gap:4,padding:"4px",background:"var(--bg-subtle)",borderRadius:7,border:"1px solid var(--border)"}}>
            {MODES.map(m=>(
              <button key={m.id} onClick={()=>switchMode(m.id)} style={{
                padding:"5px 13px",fontSize:12,cursor:"pointer",borderRadius:5,
                border:"none",fontFamily:"var(--font-ui)",fontWeight:mode===m.id?500:400,
                background:mode===m.id?"var(--bg)":"transparent",
                color:mode===m.id?"var(--text)":"var(--text-muted)",
                boxShadow:mode===m.id?"0 1px 3px rgba(0,0,0,0.08)":"none",
                transition:"all .15s"
              }}>{m.label}</button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{maxWidth:1000,margin:"0 auto",padding:"28px 32px"}}>

        {/* ── Document Panels ── */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:32,marginBottom:24}}>
          <UploadPanel label={M.a} value={docA} fileName={metaA.name} fileType={metaA.ext}
            onChange={v=>{setDocA(v);setResult(null);}}
            onFile={(t,n,e)=>{if(t){setDocA(t);setMetaA({name:n,ext:e});}else setMetaA({name:"",ext:""});setResult(null);}}
            accent="#1a4ed8"/>
          <UploadPanel label={M.b} value={docB} fileName={metaB.name} fileType={metaB.ext}
            onChange={v=>{setDocB(v);setResult(null);}}
            onFile={(t,n,e)=>{if(t){setDocB(t);setMetaB({name:n,ext:e});}else setMetaB({name:"",ext:""});setResult(null);}}
            accent="#0891b2"/>
        </div>

        {/* ── Action Row ── */}
        {err&&(
          <div style={{marginBottom:16,padding:"9px 14px",background:"var(--red-bg)",border:"1px solid var(--red-border)",borderRadius:5,fontSize:13,color:"var(--red)"}}>
            {err}
          </div>
        )}
        <div style={{display:"flex",alignItems:"center",gap:16,marginBottom:36,paddingBottom:28,borderBottom:"1px solid var(--border)"}}>
          <button onClick={analyze} disabled={loading} style={{
            padding:"10px 28px",fontSize:13,fontWeight:500,cursor:loading?"default":"pointer",
            borderRadius:5,border:"none",fontFamily:"var(--font-ui)",
            background:loading?"var(--bg-muted)":"var(--accent)",
            color:loading?"var(--text-faint)":"#fff",
            transition:"background .15s, opacity .15s",
            letterSpacing:.2
          }}>
            {loading?(
              <span style={{display:"flex",alignItems:"center",gap:8}}>
                <svg width="13" height="13" viewBox="0 0 13 13" style={{animation:"spin .7s linear infinite"}}>
                  <circle cx="6.5" cy="6.5" r="5.5" fill="none" stroke="var(--border-strong)" strokeWidth="2"/>
                  <path d="M6.5 1 A5.5 5.5 0 0 1 12 6.5" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                Analyzing…
              </span>
            ):"Analyze Documents"}
          </button>
          {loading&&phase&&(
            <span style={{fontSize:12,color:"var(--text-faint)",animation:"fadeIn .3s both"}}>
              {phase}…
            </span>
          )}
          {!loading&&!result&&(
            <span style={{fontSize:12,color:"var(--text-faint)"}}>Supports PDF, DOCX, TXT, and images (OCR)</span>
          )}
        </div>

        {/* ── Results ── */}
        {result&&(
          <div style={{display:"flex",flexDirection:"column",gap:28}}>

            {/* Score + Verdict */}
            <div style={{animation:"fadeUp .4s both",display:"grid",gridTemplateColumns:"240px 1fr",gap:32,paddingBottom:28,borderBottom:"1px solid var(--border)"}}>
              <div>
                <ScoreBar pct={pct}/>
                <div style={{marginTop:12,display:"flex",alignItems:"center",gap:8}}>
                  <span style={{fontSize:12,fontWeight:500,color:scoreColor}}>{result.ai.match_label||scoreLabel(pct).t}</span>
                  <span style={{fontSize:11,color:"var(--text-faint)"}}>·</span>
                  <span style={{fontSize:11,color:"var(--text-faint)",textTransform:"capitalize"}}>
                    {result.ai.confidence||"—"} confidence
                  </span>
                </div>
              </div>
              <div>
                <div style={{fontSize:11,fontWeight:600,letterSpacing:1.5,color:"var(--text-faint)",textTransform:"uppercase",marginBottom:10}}>Verdict</div>
                <p style={{fontFamily:"var(--font-display)",fontSize:16,fontStyle:"italic",color:"var(--text)",lineHeight:1.65,margin:0}}>
                  "{result.ai.verdict}"
                </p>
              </div>
            </div>

            {/* AI Analysis */}
            <div style={{animation:"fadeUp .4s .08s both",display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:20,paddingBottom:28,borderBottom:"1px solid var(--border)"}}>
              {[
                {label:"Strengths",text:result.ai.strength,borderColor:"var(--green)",textColor:"var(--green)"},
                {label:"Gaps",text:result.ai.gap,borderColor:"var(--amber)",textColor:"var(--amber)"},
                {label:"Recommendations",text:result.ai.recommendation,borderColor:"var(--accent)",textColor:"var(--accent)"},
              ].filter(s=>s.text).map(({label,text,borderColor,textColor})=>(
                <div key={label}>
                  <div style={{fontSize:11,fontWeight:600,letterSpacing:1.5,color:textColor,textTransform:"uppercase",marginBottom:8,display:"flex",alignItems:"center",gap:7}}>
                    <div style={{width:3,height:12,background:borderColor,borderRadius:1,flexShrink:0}}/>
                    {label}
                  </div>
                  <p style={{fontSize:13,color:"var(--text-muted)",lineHeight:1.7,margin:0}}>{text}</p>
                </div>
              ))}
            </div>

            {/* Keywords */}
            <div style={{animation:"fadeUp .4s .16s both",paddingBottom:28,borderBottom:"1px solid var(--border)"}}>
              <Section title="Term Analysis">
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:24}}>
                  {[
                    {label:M.a,words:result.topA,other:new Set(result.topB)},
                    {label:M.b,words:result.topB,other:new Set(result.topA)},
                  ].map(({label,words,other})=>(
                    <div key={label}>
                      <div style={{fontSize:12,color:"var(--text-faint)",marginBottom:10}}>{label}</div>
                      <div style={{display:"flex",flexWrap:"wrap",gap:5}}>
                        {words.map(w=>(
                          <Tag key={w} word={w} variant={other.has(w)?"shared":"neutral"}/>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{marginTop:10,display:"flex",alignItems:"center",gap:6}}>
                  <Tag word="shared term" variant="shared"/>
                  <span style={{fontSize:11,color:"var(--text-faint)"}}>= appears in both documents</span>
                </div>
              </Section>
            </div>

            {/* Overlap Breakdown */}
            <div style={{animation:"fadeUp .4s .22s both",paddingBottom:28,borderBottom:"1px solid var(--border)"}}>
              <Section title="Overlap Breakdown">
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:20}}>
                  {[
                    {label:`Shared (${result.overlap.length})`,words:result.overlap,variant:"shared"},
                    {label:`Only in ${M.a} (${result.onlyA.length})`,words:result.onlyA,variant:"unique"},
                    {label:`Only in ${M.b} (${result.onlyB.length})`,words:result.onlyB,variant:"unique"},
                  ].map(({label,words,variant})=>(
                    <div key={label}>
                      <div style={{fontSize:11,color:"var(--text-faint)",marginBottom:8}}>{label}</div>
                      <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
                        {words.length?words.map(w=><Tag key={w} word={w} variant={variant}/>):
                          <span style={{fontSize:11,color:"var(--text-faint)",fontStyle:"italic"}}>None detected</span>
                        }
                      </div>
                    </div>
                  ))}
                </div>
              </Section>
            </div>

            {/* Engine Stats */}
            <div style={{animation:"fadeUp .4s .28s both"}}>
              <Section title="Computation Details">
                <div style={{display:"flex",gap:32,flexWrap:"wrap"}}>
                  {[
                    ["Algorithm","TF-IDF + Cosine Similarity"],
                    ["Doc A — tokens",result.tokA.toLocaleString()],
                    ["Doc B — tokens",result.tokB.toLocaleString()],
                    ["Shared vocabulary",result.vocab.toLocaleString()+" terms"],
                    ["Raw cosine score",result.score.toFixed(6)],
                    ["Scale","0.000 (no match) → 1.000 (identical)"],
                  ].map(([k,v])=>(
                    <div key={k}>
                      <div style={{fontSize:10,fontWeight:600,letterSpacing:1.2,color:"var(--text-faint)",textTransform:"uppercase",marginBottom:4}}>{k}</div>
                      <div style={{fontSize:13,color:"var(--text-muted)"}}>{v}</div>
                    </div>
                  ))}
                </div>
              </Section>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}
