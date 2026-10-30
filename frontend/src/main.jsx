import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AudioLines, ChartNoAxesColumnIncreasing, Disc3, Headphones, LoaderCircle, Music2, Upload, X } from 'lucide-react';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function Waveform({ data }) {
  const path = useMemo(() => {
    if (!data?.length) return '';
    const mid = 48, scale = 39, dx = 900 / Math.max(1, data.length - 1);
    return data.map((v, i) => `${i ? 'L' : 'M'}${(i * dx).toFixed(1)},${(mid - v * scale).toFixed(1)}`).join(' ');
  }, [data]);
  return <svg className="wave" viewBox="0 0 900 96" preserveAspectRatio="none"><path d={path}/></svg>;
}

function Spectrogram({ rows }) {
  const pixels = useMemo(() => {
    if (!rows) return '';
    const colors = rows.map(row => row.map(v => {
      const t = Math.max(0, Math.min(1, v));
      const r = Math.round(18 + 230 * Math.pow(t, 1.5));
      const g = Math.round(22 + 105 * t);
      const b = Math.round(55 + 160 * (1 - t) * t);
      return `rgb(${r},${g},${b})`;
    }));
    const w = 720 / (rows[0]?.length || 1), h = 230 / rows.length;
    return colors.flatMap((row, y) => row.map((c, x) => <rect key={`${x}-${y}`} x={x*w} y={230-(y+1)*h} width={w+0.2} height={h+0.2} fill={c}/>));
  }, [rows]);
  return <svg className="spec" viewBox="0 0 720 230" preserveAspectRatio="none">{pixels}</svg>;
}

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const pick = f => { if (!f) return; setFile(f); setResult(null); setError(''); };
  const submit = async () => {
    if (!file) return;
    setBusy(true); setError('');
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch(`${API}/predict`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction request failed');
      setResult(data);
    } catch (e) { setError(e.message || 'Could not reach the API. Is it running?'); }
    finally { setBusy(false); }
  };
  const top = result?.probabilities?.[0];
  return <main className="shell">
    <header><a className="brand" href="#"><span className="brand-icon"><AudioLines size={19}/></span> Musify</a><span className="header-tag"><i/> LOCAL INFERENCE</span></header>
    <section className="hero"><div className="eyebrow"><span/> AUDIO INTELLIGENCE <span className="rule"/></div><h1>Find the style<br/>of <em>your music.</em></h1><p className="sub">Drop a track in. Explore its sonic fingerprint.</p></section>
    <section className={`drop ${drag?'drag':''} ${file?'has-file':''}`} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);pick(e.dataTransfer.files[0])}}>
      {!file ? <><div className="upload-icon"><Upload size={20}/></div><strong>Drop your audio here</strong><span className="muted">or <label htmlFor="file">browse files</label></span><small>WAV · MP3 · FLAC · OGG · M4A · up to 25 MB</small><input id="file" type="file" accept="audio/*,.au" onChange={e=>pick(e.target.files[0])}/></> : <><div className="file-icon"><Music2 size={20}/></div><div className="file-info"><strong>{file.name}</strong><span>{(file.size/1048576).toFixed(2)} MB · Ready to analyze</span></div><button className="clear" aria-label="Remove file" onClick={()=>{setFile(null);setResult(null)}}><X size={18}/></button></>}
    </section>
    {file && <button className="analyze" disabled={busy} onClick={submit}>{busy?<><LoaderCircle className="spin" size={16}/> ANALYZING TRACK</>:<><Headphones size={16}/> ANALYZE TRACK <span>↗</span></>}</button>}
    {error && <div className="error">{error}</div>}
    {result && <section className="results">
      <div className="result-head"><div><div className="eyebrow">ANALYSIS COMPLETE</div><h2>Track signature</h2><p className="result-model">{result.model_name}</p></div><span className="duration"><Disc3 size={15}/> {result.visualization.duration_seconds.toFixed(1)} SEC</span></div>
      <div className="prediction"><div className="vinyl"><Disc3 size={34}/></div><div><span className="label">PREDICTED GENRE</span><h3>{top?.genre}</h3>{result.parent_category && <span className="genre-parent">{result.parent_category} category</span>}</div><div className="confidence"><b>{(top?.probability*100).toFixed(1)}<small>%</small></b><span>CONFIDENCE</span></div></div>
      <audio className="player" controls src={URL.createObjectURL(file)}/>
      <div className="viz-grid"><div className="panel"><div className="panel-title"><span><AudioLines size={15}/> WAVEFORM</span><small>AMPLITUDE</small></div><Waveform data={result.visualization.waveform}/></div><div className="panel"><div className="panel-title"><span><ChartNoAxesColumnIncreasing size={15}/> MEL SPECTROGRAM</span><small>MODEL CLIP</small></div><Spectrogram rows={result.visualization.spectrogram}/></div></div>
      <div className="panel distribution"><div className="panel-title"><span><ChartNoAxesColumnIncreasing size={15}/> GENRE DISTRIBUTION</span><small>MODEL SCORE</small></div>{result.probabilities.map(({genre,probability})=><div className="bar-row" key={genre}><span>{genre}</span><div className="bar-bg"><i style={{width:`${Math.max(1,probability*100)}%`}}/></div><b>{(probability*100).toFixed(1)}%</b></div>)}</div>
      <p className="note">Predictions are model estimates for the supported training genres. Confidence is not a calibrated measure of certainty.</p>
    </section>}
    <footer><span>MUSIFY <b>·</b> MUSIC GENRE CLASSIFIER</span><span>4-SOURCE YAMNET + PANNS HYBRID</span></footer>
  </main>;
}

createRoot(document.getElementById('root')).render(<App/>);
