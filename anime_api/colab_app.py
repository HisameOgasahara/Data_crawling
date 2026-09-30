from __future__ import annotations

import os
import sys
from urllib.parse import quote

KUHI_ROOT = os.environ.get("KUHI_ROOT", "/content/kuhi-anime-api")
if KUHI_ROOT not in sys.path:
    sys.path.insert(0, KUHI_ROOT)

from fastapi.responses import HTMLResponse
from src.main import app


@app.get("/viewer", response_class=HTMLResponse)
async def viewer() -> str:
    return r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kuhi Colab Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
main{max-width:1100px;margin:auto;padding:24px}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
input,select,button{padding:10px;border-radius:8px;border:1px solid #444;background:#1d1d1d;color:#eee}
input[type=text]{min-width:280px}
button{cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}
.card{background:#1a1a1a;border:1px solid #333;border-radius:10px;overflow:hidden;cursor:pointer}
.card img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block}
.card .txt{padding:9px;font-size:13px}
.card small{color:#aaa}
.player{margin-top:24px}
video{width:100%;max-height:70vh;background:#000}
#status{white-space:pre-wrap;color:#bbb;margin:12px 0}
</style>
</head>
<body>
<main>
<h1>Kuhi Anime Viewer</h1>
<div class="controls">
  <input id="query" type="text" placeholder="애니 제목 검색">
  <button onclick="searchAnime()">검색</button>
</div>
<div id="status"></div>
<div id="results" class="grid"></div>

<div class="player">
  <div class="controls">
    <input id="selected" type="text" placeholder="AniList ID" readonly>
    <input id="episode" type="number" value="1" min="1">
    <select id="audio"><option value="sub">sub</option><option value="dub">dub</option></select>
    <input id="provider" type="text" placeholder="provider (선택)">
    <button onclick="playSelected()">재생</button>
  </div>
  <video id="video" controls></video>
</div>
</main>
<script>
let hls = null;
let playToken = 0;

const PROVIDERS = [
  "anineko", "anizone", "anikoto", "reanime", "aniwaves",
  "kaa", "anibd", "animegg", "mkissa", "animeonsen"
];

function titleOf(item){
  const t=item.title||{};
  return t.english || t.romaji || t.native || String(item.id);
}

async function searchAnime(){
  const q=document.getElementById("query").value.trim();
  if(!q) return;
  const status=document.getElementById("status");
  status.textContent="검색 중...";
  const r=await fetch("/anime/search?query="+encodeURIComponent(q)+"&per_page=20");
  const data=await r.json();
  const root=document.getElementById("results");
  root.innerHTML="";
  for(const item of (data.results||[])){
    const card=document.createElement("div");
    card.className="card";
    const cover=((item.coverImage||{}).large)||"";
    card.innerHTML='<img src="'+cover+'"><div class="txt"><b>'+titleOf(item)+'</b><br><small>AniList '+item.id+'</small></div>';
    card.onclick=()=>{
      document.getElementById("selected").value=item.id;
      status.textContent="선택: "+titleOf(item)+" ("+item.id+")";
    };
    root.appendChild(card);
  }
  status.textContent=(data.results||[]).length+"개 검색됨";
}

async function tryStream(data, token){
  const streams=(data.streams||[]).filter(
    s => s.type==="hls" || s.type==="mp4" || s.type==="dash" || String(s.url||"").includes(".m3u8")
  );
  if(!streams.length) return false;

  const s=streams[0];
  const video=document.getElementById("video");
  const status=document.getElementById("status");

  if(hls){hls.destroy();hls=null;}
  video.removeAttribute("src");
  video.load();

  return await new Promise((resolve)=>{
    let settled=false;
    const finish=(ok)=>{
      if(settled) return;
      settled=true;
      clearTimeout(timer);
      resolve(ok);
    };

    const timer=setTimeout(()=>finish(false),12000);

    video.onloadedmetadata=()=>{
      if(token!==playToken) return finish(false);
      status.textContent="재생 중 | provider: "+(data.provider||"?")+" | stream: "+(s.server||s.type||"?");
      video.play().catch(()=>{});
      finish(true);
    };
    video.onerror=()=>finish(false);

    if(s.type==="hls" || String(s.url).includes(".m3u8")){
      const referer=s.referer || new URL(s.url).origin + "/";
      const playUrl="/proxy_m3u8?url="+encodeURIComponent(s.url)+"&referer="+encodeURIComponent(referer);

      if(Hls.isSupported()){
        hls=new Hls();
        hls.on(Hls.Events.ERROR,(_event,err)=>{
          if(err && err.fatal) finish(false);
        });
        hls.loadSource(playUrl);
        hls.attachMedia(video);
      }else{
        video.src=playUrl;
      }
    }else{
      video.src=s.url;
    }
  });
}

async function playSelected(){
  const id=document.getElementById("selected").value.trim();
  const ep=document.getElementById("episode").value;
  const audio=document.getElementById("audio").value;
  const requested=document.getElementById("provider").value.trim();
  const status=document.getElementById("status");
  if(!id){status.textContent="검색 결과에서 작품을 먼저 선택하세요.";return;}

  const token=++playToken;
  const candidates=requested ? [requested] : [null,...PROVIDERS];
  const tried=new Set();

  for(const provider of candidates){
    if(token!==playToken) return;

    let url="/anime/extract/"+encodeURIComponent(id)+"?e="+encodeURIComponent(ep)+"&type="+encodeURIComponent(audio);
    if(provider) url+="&provider="+encodeURIComponent(provider);

    status.textContent=(provider ? provider : "자동")+" 스트림 추출 중...";

    let r;
    let data;
    try{
      r=await fetch(url);
      data=await r.json();
    }catch(_e){
      continue;
    }

    if(!r.ok) continue;

    const actual=data.provider||provider||"unknown";
    if(tried.has(actual)) continue;
    tried.add(actual);

    status.textContent=actual+" 재생 확인 중...";
    const ok=await tryStream(data,token);
    if(ok) return;
  }

  status.textContent="재생 가능한 provider를 찾지 못했습니다.";
}
</script>
</body>
</html>"""
