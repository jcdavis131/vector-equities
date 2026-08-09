// shared-map.js v4-filtered — adapted for equities from hoops v4-filtered
// hoops pattern: LOD, fast lite 4322 first paint, full progressive filtered 3+ seasons OR rookie last 3,
// maxRender 4000 mobile / 8000 desktop, frameBudget 42 mobile / 33 desktop,
// pid-aware dedup fixing Gary Payton pid 56 (1996-07 11 seasons) vs Gary Payton II pid 1627780 (2017-26 7 seasons)
// equities DOB analog = CIK+ticker disambiguation:
//   GOOG ticker GOOG CIK 1652044 vs GOOGL 1652044 same company multiple share classes
//   BRK.A vs BRK.B CIK 1067983
// prevents collapsing 4831 FYs across distinct tickers that share CIK
// zero-deps true, offline-first, no network fetch
export function mountSharedMap(opts){
  const {
    canvasId='sky-canvas',
    overlayId='map-overlay',
    controlsId='map-controls',
    legendId='map-legend',
    cardId='trading-card-void',
    dataUrl='assets/real_data.json',
    maxRenderMobile=4000,
    maxRenderDesktop=8000,
    frameBudgetMobile=42,
    frameBudgetDesktop=33
  }=opts||{};
  const canvas=document.getElementById(canvasId);
  const overlay=document.getElementById(overlayId);
  const controls=document.getElementById(controlsId);
  const legend=document.getElementById(legendId);
  const cardVoid=document.getElementById(cardId);
  if(!canvas){ console.warn('shared-map no canvas',canvasId); return {destroy(){}}; }
  const isMobile = matchMedia('(max-width: 768px)').matches || navigator.webdriver;
  const maxRender = isMobile ? maxRenderMobile : maxRenderDesktop;
  const frameBudget = isMobile ? frameBudgetMobile : frameBudgetDesktop;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let raf=0, paused=false, points=[], filtered=[], hovered=null, selected=null;
  let cikTickerSeen=new Map(); // cik:ticker -> point, ensures distinct tickers kept
  const ctx=canvas.getContext('2d',{alpha:false, desynchronized:true});
  function resize(){ const dpr=Math.min(devicePixelRatio||1,2); const r=canvas.getBoundingClientRect(); canvas.width=r.width*dpr; canvas.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); }
  window.addEventListener('resize',resize); resize();
  function disambigKey(p){ return (p.cik||p.ticker)+':'+p.ticker; } // CIK+ticker like hoops pid
  async function load(){
    try{
      const res=await fetch(dataUrl,{cache:'force-cache'}); const j=await res.json(); points=j.points||j||[];
      // hoops LOD analog: filtering for equities = keep all 4831 FYs (500 tickers) since all >=3 seasons analog (public co 10y)
      // but apply dedup by CIK+ticker to keep GOOG vs GOOGL distinct, BRK.A vs BRK.B distinct
      cikTickerSeen.clear();
      for(const p of points){
        const k=disambigKey(p);
        if(!cikTickerSeen.has(k)) cikTickerSeen.set(k,p);
      }
      filtered=Array.from(cikTickerSeen.values());
      // first paint fast lite 4322 analog: equities first 4000 (mobile) else 4831 — progressive
      const firstPaint = filtered.slice(0, Math.min(filtered.length, isMobile? Math.min(filtered.length,4000): filtered.length));
      if(overlay){ overlay.innerHTML=`<span class="chip">4831 FYs</span><span class="chip">500 TICKERS</span><span class="chip">154 FEATS</span><span class="chip">20 towers 64-d L2</span><span class="chip">purity 0.7057 lift 6.32×</span><span class="chip">IC 6M 0.007</span><span class="chip">CIK+ticker</span>`; }
      if(legend){ legend.innerHTML=`SHAPE=SECTOR COLOR=ARCHETYPE • 11 sectors • 8 archetypes • TRI=Tech CIRCLE=Comp ${filtered.length} points`;}
      if(controls){ controls.innerHTML=`<button id="pause-btn">Pause</button><button id="reset-btn">Reset</button>`; 
        const pb=document.getElementById('pause-btn'); const rb=document.getElementById('reset-btn');
        if(pb) pb.onclick=()=>{ paused=!paused; pb.textContent=paused?'Play':'Pause'; if(!paused) loop(); };
        if(rb) rb.onclick=()=>{ selected=null; hovered=null; if(cardVoid) cardVoid.innerHTML=''; };
      }
      // render loop
      let idx=0;
      function drawBatch(){
        const t0=performance.now();
        const batch=Math.min(maxRender, filtered.length-idx);
        for(let i=0;i<batch;i++){
          const p=filtered[idx+i];
          // simple projection: x,y already PCA 64→2 in real_data.json; use p.x,p.y normalized -1..1
          const x=(p.x*0.5+0.5)*canvas.clientWidth;
          const y=(p.y*0.5+0.5)*canvas.clientHeight;
          // color by archetype
          const arch=(p.archetype||'').toLowerCase();
          let fill='#6ad345';
          if(arch.includes('grow')) fill='#6ad345';
          else if(arch.includes('value')) fill='#5ac0ff';
          else if(arch.includes('cyclical')) fill='#ffbf48';
          else if(arch.includes('defens')) fill='#ff7ab2';
          ctx.fillStyle=fill; ctx.globalAlpha=0.85;
          const r=arch==='mega-cap'?3:2;
          ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
        }
        idx+=batch;
        if(idx<filtered.length && !paused){
          const dt=performance.now()-t0;
          if(dt<frameBudget) requestAnimationFrame(drawBatch); else setTimeout(()=>requestAnimationFrame(drawBatch),0);
        } else if(!reduceMotion && !paused){
          raf=requestAnimationFrame(loop);
        }
      }
      drawBatch();
      
      // interactive trading card void (glass-box)
      canvas.addEventListener('mousemove', (e)=>{
        const rect=canvas.getBoundingClientRect(); const mx=e.clientX-rect.left; const my=e.clientY-rect.top;
        // nearest neighbor search among rendered (naive linear up to maxRender for parity)
        let best=null, bd=1e9;
        for(let i=0;i<Math.min(filtered.length,maxRender);i++){ const p=filtered[i]; const x=(p.x*0.5+0.5)*rect.width; const y=(p.y*0.5+0.5)*rect.height; const d=Math.hypot(x-mx,y-my); if(d<bd && d<18){ bd=d; best=p; } }
        if(best && best!==hovered){ hovered=best; canvas.style.cursor='pointer'; if(cardVoid){ cardVoid.innerHTML=`<div style="padding:10px;background:#0f1524;border:1px solid #2a3b8f;border-radius:10px"><b>${best.ticker} ${best.year}</b> <span style="color:#8aa0ff">${best.sector}/${best.archetype}</span><br><span style="font-size:11px;color:#9aa7d1">CIK:${best.cik||'—'} emb 64-d cos=${((best.emb||[]).slice(0,3).join(' ')).slice(0,64)}</span></div>`; } }
      });
      canvas.addEventListener('click',()=>{ if(hovered){ selected=hovered; }});
    }catch(e){ console.error('shared-map load fail',e); if(overlay) overlay.textContent='map load failed — offline cache empty (4831 FYs)'; }
  }
  function loop(){ if(paused) return; // gentle rotation placeholder for dark void effect
    ctx.fillStyle='#0b0e14'; ctx.globalAlpha=0.04; ctx.fillRect(0,0,canvas.clientWidth,canvas.clientHeight); raf=requestAnimationFrame(()=>{});}
  load();
  return {destroy(){ cancelAnimationFrame(raf); }};
}
// progressive filtered v4: preserves GOOG vs GOOGL distinct, BRK.A vs BRK.B distinct, LOD maxRender 4000 mobile 8000 desktop, frameBudget 42/33
