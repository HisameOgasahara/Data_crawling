from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

KUHI_ROOT = os.environ.get("KUHI_ROOT", "/content/kuhi-anime-api")
if KUHI_ROOT not in sys.path:
    sys.path.insert(0, KUHI_ROOT)

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.main import app


DOWNLOAD_DIR = Path(
    os.environ.get("DOWNLOAD_DIR", "/content/downloads")
).resolve()
THUMB_DIR = DOWNLOAD_DIR / ".thumbnails"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/download-files",
    StaticFiles(directory=str(DOWNLOAD_DIR)),
    name="download-files",
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".webm",
    ".mkv",
    ".mov",
}


def _safe_download_path(relative_path: str) -> Path:
    candidate = (DOWNLOAD_DIR / relative_path).resolve()

    if DOWNLOAD_DIR not in candidate.parents and candidate != DOWNLOAD_DIR:
        raise HTTPException(status_code=400, detail="invalid path")

    return candidate


def _video_files() -> list[Path]:
    files = []

    for path in DOWNLOAD_DIR.rglob("*"):
        if not path.is_file():
            continue

        if THUMB_DIR in path.parents:
            continue

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        if path.name.endswith(".part"):
            continue

        if path.stat().st_size <= 0:
            continue

        files.append(path)

    files.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files


@app.get("/library")
async def library() -> dict:
    items = []

    for path in _video_files():
        relative = path.relative_to(DOWNLOAD_DIR).as_posix()
        stat = path.stat()

        items.append(
            {
                "name": path.name,
                "relativePath": relative,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "videoUrl": "/download-files/" + quote(relative),
                "thumbnailUrl": "/library/thumbnail?path=" + quote(relative),
            }
        )

    return {"items": items}


@app.get("/library/thumbnail")
async def library_thumbnail(
    path: str = Query(...),
):
    video_path = _safe_download_path(path)

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="video not found")

    thumb_name = (
        str(abs(hash(video_path.as_posix())))
        + "_"
        + str(video_path.stat().st_mtime_ns)
        + ".jpg"
    )
    thumb_path = THUMB_DIR / thumb_name

    if not thumb_path.exists():
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                "5",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-2",
                "-q:v",
                "3",
                str(thumb_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0 or not thumb_path.exists():
            raise HTTPException(
                status_code=500,
                detail="thumbnail generation failed",
            )

    return FileResponse(
        thumb_path,
        media_type="image/jpeg",
    )


@app.get("/viewer", response_class=HTMLResponse)
async def viewer() -> str:
    return r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Downloaded Anime Viewer</title>
<style>
body{
  font-family:system-ui,sans-serif;
  margin:0;
  background:#111;
  color:#eee;
}
main{
  max-width:1200px;
  margin:auto;
  padding:24px;
}
.controls{
  display:flex;
  gap:8px;
  flex-wrap:wrap;
  margin-bottom:18px;
}
input,button{
  padding:10px;
  border-radius:8px;
  border:1px solid #444;
  background:#1d1d1d;
  color:#eee;
}
input{
  min-width:320px;
  flex:1;
}
button{
  cursor:pointer;
}
#status{
  color:#aaa;
  margin:10px 0 18px;
}
.grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
  gap:14px;
}
.card{
  background:#1a1a1a;
  border:1px solid #333;
  border-radius:10px;
  overflow:hidden;
  cursor:pointer;
}
.card:hover{
  border-color:#666;
}
.card img{
  width:100%;
  aspect-ratio:16/9;
  object-fit:cover;
  display:block;
  background:#222;
}
.card .txt{
  padding:10px;
  font-size:13px;
  word-break:break-all;
}
.card small{
  color:#999;
}
.player{
  margin-top:26px;
}
.player h2{
  font-size:16px;
  font-weight:600;
  word-break:break-all;
}
video{
  width:100%;
  max-height:75vh;
  background:#000;
}
.empty{
  color:#999;
  border:1px dashed #444;
  border-radius:10px;
  padding:28px;
}
</style>
</head>
<body>
<main>
<h1>Downloaded Anime Viewer</h1>

<div class="controls">
  <input
    id="query"
    type="text"
    placeholder="다운로드된 파일명 검색"
    oninput="renderLibrary()"
  >
  <button onclick="loadLibrary()">새로고침</button>
</div>

<div id="status"></div>
<div id="results" class="grid"></div>

<div class="player">
  <h2 id="playingTitle">선택된 파일 없음</h2>
  <video id="video" controls preload="metadata"></video>
</div>
</main>

<script>
let libraryItems=[];

function humanSize(bytes){
  const units=["B","KB","MB","GB"];
  let value=Number(bytes||0);
  let unit=0;

  while(value>=1024 && unit<units.length-1){
    value/=1024;
    unit+=1;
  }

  return value.toFixed(unit===0 ? 0 : 1)+" "+units[unit];
}

function playItem(item){
  const video=document.getElementById("video");
  const title=document.getElementById("playingTitle");

  title.textContent=item.name;
  video.src=item.videoUrl;
  video.load();
  video.play().catch(()=>{});

  window.scrollTo({
    top:document.body.scrollHeight,
    behavior:"smooth",
  });
}

function renderLibrary(){
  const query=document
    .getElementById("query")
    .value
    .trim()
    .toLowerCase();

  const root=document.getElementById("results");
  const status=document.getElementById("status");

  const filtered=libraryItems.filter(
    item=>item.name.toLowerCase().includes(query)
  );

  root.innerHTML="";

  if(!filtered.length){
    root.innerHTML=
      '<div class="empty">조건에 맞는 다운로드 영상이 없습니다.</div>';
    status.textContent=
      libraryItems.length+
      "개 다운로드 파일 중 "+
      filtered.length+
      "개 표시";
    return;
  }

  for(const item of filtered){
    const card=document.createElement("div");
    card.className="card";

    const img=document.createElement("img");
    img.src=item.thumbnailUrl;
    img.loading="lazy";
    img.alt=item.name;

    const txt=document.createElement("div");
    txt.className="txt";

    const name=document.createElement("b");
    name.textContent=item.name;

    const meta=document.createElement("small");
    meta.textContent=humanSize(item.size);

    txt.appendChild(name);
    txt.appendChild(document.createElement("br"));
    txt.appendChild(meta);

    card.appendChild(img);
    card.appendChild(txt);
    card.onclick=()=>playItem(item);

    root.appendChild(card);
  }

  status.textContent=
    libraryItems.length+
    "개 다운로드 파일 중 "+
    filtered.length+
    "개 표시";
}

async function loadLibrary(){
  const status=document.getElementById("status");
  status.textContent="다운로드 폴더 확인 중...";

  try{
    const response=await fetch("/library");
    const data=await response.json();

    libraryItems=data.items||[];
    renderLibrary();
  }catch(error){
    status.textContent="다운로드 목록을 불러오지 못했습니다.";
  }
}

loadLibrary();
</script>
</body>
</html>"""
