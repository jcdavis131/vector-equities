/* shared-map.js v4-filtered — equities adaptation borrowing from hoops
 * 500 tickers 4831 FYs — same LOD / pause logic as hoops 3+ seasons OR rookie last 3
 * Sources: real_data.json then real_pca.json then real_pca_full.json
 * CIK+ticker disambiguation analog handling multiple share classes GOOG/GOOGL BRK.A/BRK.B
 */
export async function mountSharedMap(canvas, opts={}){
  if(!canvas) return null;
  const OKABE=['#0072B2','#D55E00','#009E73','#F0E442','#56B4E9','#CC79A7','#E69F00','#FFFEF7'];
  const ARCH=["Compounder","Cash_Cow","Turnaround","HyperGrowth_SaaS","Heavy_Industrial","Bank_Capital_Heavy","Moonshot_Bio","Serial_Acquirer"];
  const SECT=["Technology","Healthcare","Financials","Energy","Industrials","Consumer Staples","Consumer Discretionary","Utilities","Materials","Real Estate","Communication"];
  const highlightInit = opts.highlightId ?? null;
  const dark = !!opts.dark;
  const isMobile = (typeof window!=='undefined') && (window.innerWidth<700 || /Android|iPhone|iPad/i.test(navigator.userAgent||''));
  // hoops parity: mobile 4000 desktop 8000 to keep same tempo
  const maxRender = isMobile ? 4000 : 8000;
  const frameBudget = isMobile ? 42 : 33;
  const reduceMotion = (typeof window!=='undefined') && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let N=0, baseOx=null, baseOy=null, baseOz=null, baseC=null, baseI=null, baseN=[], baseS=[], baseP=[], baseSector=[];
  let projected=[], projById=null, maxId=0, tickerToIdx=new Map();
  let W=0,H=0, rotY=Math.PI*0.18, rotX=0.22, auto=!reduceMotion, lastT=0, isDragging=false, lastX=0,lastY=0, idleMs=0;
  let embedPaused=false, lastRender=0;
  let fullLoaded=false, fullLoading=false, pendingFocus=null;
  let totalRaw=4831, filteredCount=0;

  function fyEnd(y){ if(!y) return null; let n=parseInt(String(y).slice(-4),10); if(isNaN(n)) return null; if(n<100) n+= n>=50?1900:2000; return n; }

  function buildFilter(allPoints){
    // pid-aware for equities: CIK + ticker distinct, e.g. GOOG vs GOOGL same CIK diff class -> keep both, BRK.A/BRK.B similar
    // Use CIK if present else ticker lower
    let maxYear=0; for(const p of allPoints){ const y=fyEnd(p.year||p.fy); if(y && y>maxYear) maxYear=y; }
    if(!maxYear) maxYear=(new Date()).getFullYear();
    const recentMin = maxYear - 2;
    const byCo=new Map();
    for(const p of allPoints){
      const cik = p.cik? String(p.cik) : '';
      const tic = (p.ticker||'').trim();
      if(!tic) continue;
      // distinct by ticker primarily (share classes separate), group for count by CIK if multiple FYs same CIK merge but keep tickers
      const key = cik ? ('cik:'+cik+':'+tic.toUpperCase()) : ('tick:'+tic.toUpperCase());
      let rec=byCo.get(key); if(!rec){ rec={count:0,maxY:0,minY:9999,tick:tic,cik}; byCo.set(key,rec); }
      rec.count++; const y=fyEnd(p.year||p.fy)||0; if(y){ if(y>rec.maxY) rec.maxY=y; if(y<rec.minY) rec.minY=y; }
    }
    const keepKeys=new Set();
    for(const [k,rec] of byCo){ if(rec.count>=3) keepKeys.add(k); else if(rec.maxY && rec.maxY>=recentMin) keepKeys.add(k); }
    let kept=0; for(const p of allPoints){ const cik=p.cik?String(p.cik):''; const tick=(p.ticker||'').toUpperCase(); const k=cik?('cik:'+cik+':'+tick):('tick:'+tick); if(keepKeys.has(k)) kept++; }
    console.log('equities season filter CIK+ticker-aware: maxYear',maxYear,'recentMin',recentMin,'keptPersons',keepKeys.size,'keptPts',kept,'/',allPoints.length);
    return {keepKeys, maxYear, recentMin, kept, raw:allPoints.length, byCo};
  }

  function _injectPoint(p){
    try{
      if(!p||!p.ticker||!baseOx) return false;
      const idStr = (p.ticker.toUpperCase()+'::'+(p.year||p.fy||'')); // synthetic id for ticker+FY
      const existing = tickerToIdx.get(idStr);
      if(existing!=null) return true;
      const n=N+1;
      const nOx=new Float32Array(n), nOy=new Float32Array(n), nOz=new Float32Array(n);
      const nC=new Uint8Array(n), nI=new Int32Array(n);
      nOx.set(baseOx); nOy.set(baseOy); nOz.set(baseOz); nC.set(baseC); nI.set(baseI);
      nOx[N]=p.x!=null?((p.x-0.5)*2||0):(Math.random()-0.5); nOy[N]=p.y!=null?((p.y-0.5)*2||0):(Math.random()-0.5); nOz[N]=p.z!=null?((p.z-0.5)*2||0):0;
      const archIdx = ARCH.indexOf(p.archetype||p.arch||''); nC[N]=(archIdx>=0?archIdx:0)&7; nI[N]=N;
      baseOx=nOx; baseOy=nOy; baseOz=nOz; baseC=nC; baseI=nI;
      baseN[N]=p.ticker||''; baseS[N]=String(p.year||p.fy||''); baseP[N]=p.sector? SECT.indexOf(p.sector): -1; baseSector[N]=p.sector||'';
      projected.push({sx:0,sy:0,depth:0,alpha:0.6,c:nC[N]});
      tickerToIdx.set(idStr,N);
      N=n; projectFrame(); return true;
    }catch(e){ console.warn('_inject equities fail',e); return false; }
  }

  let targetId=highlightInit, guessIds=[];
  let hoverEl=null; try{hoverEl=document.getElementById('hover-tip');}catch{}
  let ctx=null; try{ ctx=canvas.getContext('2d',{alpha:false}); }catch{ ctx=canvas.getContext('2d'); }

  function getSize(){
    const rect=canvas.getBoundingClientRect();
    let w=rect.width, h=rect.height;
    if(w<10||h<10){ const pr=canvas.parentElement?.getBoundingClientRect(); w=Math.max(w, pr?.width||0, 320); h=Math.max(h, pr?.height||0, 380); if(w<10) w=window.innerWidth||390; if(h<10) h=Math.round((window.innerHeight||800)*0.5); }
    return {w:Math.max(10,Math.round(w)), h:Math.max(10,Math.round(h))};
  }
  function resize(){
    if(!canvas) return;
    const sz=getSize();
    if(W===sz.w && H===sz.h && canvas.width===sz.w && canvas.height===sz.h) return;
    W=sz.w; H=sz.h; canvas.width=W; canvas.height=H;
    if(canvas.style.width!==W+'px') canvas.style.width=W+'px';
    if(canvas.style.height!==H+'px') canvas.style.height=H+'px';
    if(ctx) ctx.setTransform(1,0,0,1,0,0);
    projectFrame(); draw();
  }
  function ensureArrays(len){
    if(!baseOx || baseOx.length!==len){
      baseOx=new Float32Array(len); baseOy=new Float32Array(len); baseOz=new Float32Array(len);
      baseC=new Uint8Array(len); baseI=new Int32Array(len);
      projected=new Array(len); for(let i=0;i<len;i++) projected[i]={sx:0,sy:0,depth:0,alpha:0.6};
      baseN=new Array(len); baseS=new Array(len); baseP=new Array(len); baseSector=new Array(len);
    }
  }

  async function fetchWithCache(url){
    if(window.__eqMapCache && window.__eqMapCache[url]) return window.__eqMapCache[url];
    try{
      if('caches' in window){
        const cache=await caches.open('vector-equities-maps-v4');
        const hit=await cache.match(url);
        if(hit){ const j=await hit.json(); window.__eqMapCache=window.__eqMapCache||{}; window.__eqMapCache[url]=j; return j; }
        const res=await fetch(url,{cache:'default'});
        if(res.ok){ cache.put(url, res.clone()); const j=await res.json(); window.__eqMapCache=window.__eqMapCache||{}; window.__eqMapCache[url]=j; return j; }
      }
    }catch{}
    const r=await fetch(url,{cache:'force-cache'});
    if(!r.ok) throw new Error('fetch failed '+url);
    const j=await r.json();
    window.__eqMapCache=window.__eqMapCache||{}; window.__eqMapCache[url]=j;
    return j;
  }

  async function loadLite(){
    const urls=['assets/real_data.json','assets/real_pca.json','assets/real_pca_full.json'];
    for(const u of urls){
      try{
        const j=await fetchWithCache(u);
        let arr = j.points || j.players || j;
        if(!Array.isArray(arr)||!arr.length) continue;
        // latest per ticker for lite first paint (500 points) = faster
        // But keep all 4831 for map if arr length 4831; lite is latest 500
        const latestMap=new Map();
        for(const p of arr){ const tk=(p.ticker||'').toUpperCase(); if(!tk) continue; const yr=parseInt(p.year||p.fy||0)||0; const ex=latestMap.get(tk); if(!ex||yr> (parseInt(ex.year||ex.fy)||0)) latestMap.set(tk,p); }
        const lite = Array.from(latestMap.values());
        const use = (lite.length>=200? lite: arr).slice(0,500);
        N=use.length; ensureArrays(N);
        let localMax=0;
        for(let i=0;i<N;i++){
          const p=use[i]||{}; baseOx[i]=(p.x!=null? (p.x-0.5)*2 : (Math.random()-0.5)); baseOy[i]=(p.y!=null?(p.y-0.5)*2:0); baseOz[i]=(p.z!=null?(p.z-0.5)*2:0);
          const archIdx=ARCH.indexOf(p.archetype||p.arch||''); baseC[i]=(archIdx>=0?archIdx:0)&7;
          baseI[i]=i; baseN[i]=p.ticker||''; baseS[i]=String(p.year||p.fy||''); const sec=p.sector||''; baseSector[i]=sec; baseP[i]=SECT.indexOf(sec); if(i>localMax) localMax=i;
          projected[i].c=baseC[i];
          tickerToIdx.set((p.ticker||'').toUpperCase()+'::'+String(p.year||p.fy||''), i);
        }
        maxId=localMax; projById=new Int32Array(maxId+1000); projById.fill(-1); for(let i=0;i<N;i++) projById[i]=i;
        console.log('equities shared-map lite loaded',N,u); return {fullArr:arr, lite:use};
      }catch(e){ console.warn('lite equities fail',u,e); }
    }
    return null;
  }

  async function loadFullProgressive(fullArr){
    if(fullLoaded||fullLoading) return; fullLoading=true;
    try{
      if(!fullArr){
        try{ const j=await fetchWithCache('assets/real_data.json'); fullArr=j.points||j; }catch{}
      }
      if(!fullArr||!fullArr.length){ fullLoading=false; return; }
      totalRaw=fullArr.length;
      const {keepKeys, maxYear, recentMin, kept} = buildFilter(fullArr);
      filteredCount=kept;
      // filter to kept if keeps >= 1500 else full
      const filtered = fullArr.filter(p=>{
        const cik=p.cik?String(p.cik):''; const tick=(p.ticker||'').toUpperCase(); const k=cik?('cik:'+cik+':'+tick):('tick:'+tick); return keepKeys.has(k);
      });
      const useArr = (filtered.length>=1500? filtered: fullArr);
      // keep all FYs for full map (4831) not just latest — user-scrolled
      const fullN=useArr.length;
      const newOx=new Float32Array(fullN), newOy=new Float32Array(fullN), newOz=new Float32Array(fullN);
      const newC=new Uint8Array(fullN), newI=new Int32Array(fullN);
      const newNArr=new Array(fullN), newSArr=new Array(fullN), newPArr=new Array(fullN), newSec=new Array(fullN);
      const newProj=new Array(fullN);
      for(let i=0;i<fullN;i++){
        const p=useArr[i]||{}; newOx[i]=p.x!=null?(p.x-0.5)*2:(Math.random()-0.5); newOy[i]=p.y!=null?(p.y-0.5)*2:0; newOz[i]=p.z!=null?(p.z-0.5)*2:0;
        const archIdx=ARCH.indexOf(p.archetype||p.arch||''); newC[i]=(archIdx>=0?archIdx:0)&7; newI[i]=i;
        newNArr[i]=p.ticker||''; newSArr[i]=String(p.year||p.fy||''); newPArr[i]=SECT.indexOf(p.sector||''); newSec[i]=p.sector||'';
        newProj[i]={sx:0,sy:0,depth:0,alpha:0.6,c:newC[i]};
      }
      baseOx=newOx; baseOy=newOy; baseOz=newOz; baseC=newC; baseI=newI; baseN=newNArr; baseS=newSArr; baseP=newPArr; baseSector=newSec; projected=newProj; N=fullN;
      maxId=fullN-1; projById=new Int32Array(maxId+1); projById.fill(-1); for(let i=0;i<N;i++) projById[i]=i;
      // rebuild tickerToIdx
      tickerToIdx=new Map(); for(let i=0;i<N;i++) tickerToIdx.set((baseN[i]||'').toUpperCase()+'::'+(baseS[i]||''), i);
      fullLoaded=true; console.log('equities full merged',N,'from raw',totalRaw,'maxYear',maxYear,'recentMin',recentMin);
      projectFrame(); draw();
      if(pendingFocus){ const {id,label}=pendingFocus; pendingFocus=null; const key=tickerToIdx.get(id); if(key!=null){ targetId=key; projectFrame(); draw(); focusOnTargetInternal(); } }
    }catch(e){ console.warn('full progressive equities fail',e); }
    fullLoading=false;
  }

  function projectFrame(){
    if(!baseOx||!N) return;
    if(!isFinite(rotY)||!isFinite(rotX)){ rotY=Math.PI*0.18; rotX=0.22; }
    const cy=Math.cos(rotY), sy=Math.sin(rotY), cx=Math.cos(rotX), sx=Math.sin(rotX);
    const persp=2.8, W2=W*0.5, H2=H*0.5, W40=W*0.40, H40=H*0.40;
    for(let i=0;i<N;i++){ const ox=baseOx[i], oy=baseOy[i], oz=baseOz[i]; const xr=ox*cy+oz*sy; const z1=-ox*sy+oz*cy; const yr=oy*cx - z1*sx; const zr=oy*sx + z1*cx; const sc=persp/(persp - zr*0.55); const pr=projected[i]; pr.sx=W2 + xr*sc*W40; pr.sy=H2 - yr*sc*H40; pr.depth=(zr+1)*0.5; pr.alpha=0.22+pr.depth*0.78; }
  }
  function draw(){
    if(!ctx||!W||!H) return;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle=dark?'#080A0F':'#FFFEF7'; ctx.fillRect(0,0,W,H);
    if(!N){ ctx.fillStyle=dark?'#FFFEF7':'#1A150F'; ctx.font='800 12px ui-monospace,monospace'; ctx.fillText(fullLoading? 'Loading 4831 FYs… '+N : 'Loading map…',14,22); return; }
    const step=Math.max(1, Math.ceil(N / maxRender));
    const dotSize = W<600?2:2.4;
    const sectorShape = {Technology:'●',Healthcare:'▲',Financials:'◆',Energy:'⬣',Industrials:'■',"Consumer Staples":'⬡',"Consumer Discretionary":'⬢',Utilities:'⭘',Materials:'◐',"Real Estate":'▣',Communication:'⬔'};
    for(let c=0;c<8;c++){
      ctx.fillStyle=OKABE[c];
      for(let i=0;i<N;i+=step){
        if(baseC[i]!==c) continue; const pr=projected[i]; if(!pr) continue; if(pr.sx<-20||pr.sx>W+20||pr.sy<-20||pr.sy>H+20) continue;
        ctx.fillRect(pr.sx|0, pr.sy|0, dotSize, dotSize);
      }
    }
    // selected emphasis
    if(targetId!=null && projById && targetId<=maxId){
      const tIdx = (typeof targetId==='number'? targetId : tickerToIdx.get(targetId)) ?? -1;
      const idx = Number.isInteger(tIdx)? tIdx : projById[targetId] ?? -1;
      if(idx>=0){ const pr=projected[idx]; if(pr && pr.sx>=-20&&pr.sx<=W+20&&pr.sy>=-20&&pr.sy<=H+20){
        const x=pr.sx|0,y=pr.sy|0; ctx.lineWidth=3; ctx.strokeStyle='#FFFFFF'; ctx.beginPath(); ctx.arc(x,y,11,0,Math.PI*2); ctx.stroke();
        ctx.lineWidth=2.4; ctx.strokeStyle='#1A150F'; ctx.beginPath(); ctx.arc(x,y,7.5,0,Math.PI*2); ctx.stroke();
        ctx.fillStyle='#F0E442'; ctx.beginPath(); ctx.arc(x,y,3.4,0,Math.PI*2); ctx.fill();
      }}
    }
    // progress note
    if(!fullLoaded && !fullLoading){ ctx.fillStyle=dark?'rgba(255,254,247,.65)':'rgba(26,21,15,.6)'; ctx.font='700 10px ui-monospace,monospace'; ctx.fillText((N||0)+'/'+(totalRaw||4831)+' • CIK+ticker filtered', 12, H-10); }
  }

  let rafPending=false;
  function scheduleLoop(){ if(!rafPending){ rafPending=true; requestAnimationFrame(loop); } }
  function loop(t){
    rafPending=false; if(embedPaused) return;
    const now=t||performance.now(); if(now-lastRender < frameBudget){ scheduleLoop(); return; } lastRender=now;
    if(!lastT) lastT=now; const dt=Math.min(50, now-lastT); lastT=now;
    if(!isDragging && auto){ rotY+=dt*0.00022; idleMs+=dt; if(idleMs>8000){ auto=false; embedPaused=true; console.log('equities map idle pause'); return; } }
    else if(!isDragging && !auto){ projectFrame(); try{ draw(); }catch(e){ console.warn('draw fail',e); } return; } else idleMs=0;
    projectFrame(); try{ draw(); }catch(e){ console.warn('draw fail',e); } scheduleLoop();
  }

  function onDown(ev){ const pt=ev.touches? ev.touches[0]:ev; isDragging=true; auto=false; idleMs=0; lastX=pt.clientX; lastY=pt.clientY; canvas.style.cursor='grabbing'; embedPaused=false; lastT=0; scheduleLoop(); const bp=document.getElementById('btn-pause'); if(bp) bp.textContent='Pause'; }
  function onMove(ev){
    const pt=ev.touches? ev.touches[0]:ev; const x=pt.clientX, y=pt.clientY;
    if(isDragging){ const dx=x-lastX, dy=y-lastY; rotY+=dx*0.0065; rotX+=dy*0.0045; rotX=Math.max(-0.92, Math.min(0.92, rotX)); lastX=x; lastY=y; return; }
    if(!hoverEl) return; const rect=canvas.getBoundingClientRect(); const mx=x-rect.left, my=y-rect.top; let best=null,bd=isMobile?28:22; const step=Math.max(1, Math.ceil(N/maxRender)); for(let i=0;i<N;i+=step){ const pr=projected[i]; if(!pr) continue; const d=Math.hypot(pr.sx-mx, pr.sy-my); if(d<bd){ bd=d; best=i; } }
    if(best!=null){ hoverEl.style.display='block'; hoverEl.style.left=projected[best].sx+'px'; hoverEl.style.top=(projected[best].sy-42)+'px'; const n=baseN[best]||''; const s=baseS[best]||''; const c=baseC[best]; const arch=ARCH[c%8]||''; const sec=baseSector[best]||''; const pIdx=baseP[best]; const shape = sec? ( {Technology:'●',Healthcare:'▲',Financials:'◆',Energy:'⬣',Industrials:'■','Consumer Staples':'⬡','Consumer Discretionary':'⬢',Utilities:'⭘',Materials:'◐','Real Estate':'▣',Communication:'⬔'}[sec]||'●') : ''; hoverEl.innerHTML=`<b>${(n||'').replace(/</g,'&lt;')}</b> FY${(s||'').replace(/</g,'&lt;')}<br><span style="font-family:ui-monospace,monospace;font-size:9px;opacity:.8">${shape} ${sec} • ${arch}</span>`; } else hoverEl.style.display='none';
  }
  function onUp(){ if(isDragging){ isDragging=false; canvas.style.cursor='grab'; lastT=0; } }

  try{ window.addEventListener('vh:pause-maps',()=>{ embedPaused=true; auto=false; }); window.addEventListener('vh:resume-maps',()=>{ embedPaused=false; auto=!reduceMotion; lastT=0; idleMs=0; scheduleLoop(); }); document.addEventListener('visibilitychange',()=>{ if(document.hidden){ embedPaused=true; } else { embedPaused=false; lastT=0; scheduleLoop(); } }); }catch{}

  canvas.addEventListener('mousedown', onDown); canvas.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
  canvas.addEventListener('touchstart', onDown, {passive:true}); canvas.addEventListener('touchmove', onMove, {passive:true}); canvas.addEventListener('touchend', onUp);
  canvas.addEventListener('mouseleave',()=>{ if(hoverEl) hoverEl.style.display='none'; });
  const pauseBtn=document.getElementById('btn-pause'); if(pauseBtn) pauseBtn.addEventListener('click',()=>{ auto=!auto; embedPaused=!auto; pauseBtn.textContent=auto?'Pause':'Resume'; lastT=0; idleMs=0; if(auto) scheduleLoop(); });
  const resetBtn=document.getElementById('btn-reset'); if(resetBtn) resetBtn.addEventListener('click',()=>{ rotY=Math.PI*0.18; rotX=0.22; auto=!reduceMotion; embedPaused=false; idleMs=0; lastT=0; if(pauseBtn) pauseBtn.textContent=auto?'Pause':'Resume'; resize(); scheduleLoop(); });

  resize();
  let ro=null, roPending=false;
  try{ const onResizeObserved=()=>{ if(roPending) return; roPending=true; requestAnimationFrame(()=>{ roPending=false; resize(); }); }; ro=new ResizeObserver(onResizeObserved); ro.observe(canvas); if(canvas.parentElement) ro.observe(canvas.parentElement); }catch{}
  const liteRes=await loadLite();
  if(liteRes){ projectFrame(); draw(); scheduleLoop(); setTimeout(()=>{ loadFullProgressive(liteRes.fullArr); }, 120); }
  else { ctx.fillStyle='#FFFEF7'; ctx.fillText('Map failed to load',14,22); }

  function ensureFullThenFocus(key,label){
    if(!fullLoaded && !fullLoading){ loadFullProgressive(liteRes?.fullArr); }
    const idx = tickerToIdx.get(key);
    if(idx!=null){ targetId=idx; projectFrame(); draw(); focusOnTargetInternal(); return true; }
    if(!fullLoaded){ pendingFocus={id:key,label:label||''}; if(document.getElementById('popular-current')) document.getElementById('popular-current').textContent='Loading for '+(label||key)+' … '+N+'/'+(totalRaw||4831); return false; }
    return false;
  }
  function focusOnTargetInternal(){
    if(targetId==null) return;
    let idx = typeof targetId==='number'? targetId : tickerToIdx.get(targetId);
    if(idx==null||idx<0||idx>=N) return;
    const ox=baseOx[idx], oy=baseOy[idx], oz=baseOz[idx]; const ry=-Math.atan2(ox,oz); const r=Math.sqrt(ox*ox+oz*oz)||1; const rx=-Math.atan2(oy,r)*0.85;
    if(isFinite(ry)&&isFinite(rx)){ rotY=ry; rotX=rx; } projectFrame(); draw();
  }

  return {
    setTarget(id){
      // id can be ticker or ticker::year or numeric index
      if(typeof id==='string' && id.includes('::')){ if(!ensureFullThenFocus(id,null)){ targetId=tickerToIdx.get(id)||0; draw(); return; } targetId=tickerToIdx.get(id)||0; draw(); return; }
      if(typeof id==='string'){ // ticker like META
        // find latest FY for that ticker
        let best=null,bestY=-1;
        for(let i=0;i<N;i++){ if((baseN[i]||'').toUpperCase()===id.toUpperCase()){ const yr=parseInt(baseS[i]||'0'); if(yr>bestY){bestY=yr; best=i; } } }
        if(best!=null){ targetId=best; projectFrame(); draw(); focusOnTargetInternal(); return; }
        // not in lite, queue
        ensureFullThenFocus(id+'::'+2024, id); return;
      }
      targetId=id==null?null:id|0; draw();
    },
    setGuesses(ids){ draw(); },
    focusOnTarget(){ focusOnTargetInternal(); },
    hasPoint(key){ if(typeof key==='string') return tickerToIdx.has(key); return key>=0&&key<N; },
    addPoint(p){ const ok=_injectPoint(p); if(ok) draw(); return ok; },
    ensureFull: ()=>loadFullProgressive(liteRes?.fullArr),
    getProgress(){ return {loaded:N, total:totalRaw, filtered:filteredCount, full:fullLoaded}; },
    resize, getCount(){return N;}, dispose(){ try{ro&&ro.disconnect();}catch{} }
  };
}
