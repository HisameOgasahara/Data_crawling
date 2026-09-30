from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from yt_dlp import YoutubeDL


class KuhiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, page: int = 1, per_page: int = 12) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/anime/search",
            params={"query": query, "page": page, "per_page": per_page},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def extract(
        self,
        query_or_id: str | int,
        episode: int = 1,
        audio: str = "sub",
        provider: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "e": episode,
            "type": audio,
        }
        if provider:
            params["provider"] = provider

        response = requests.get(
            f"{self.base_url}/anime/extract/{query_or_id}",
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def download(
        self,
        extract_result: dict[str, Any],
        output_path: str,
        stream_index: int = 0,
    ) -> Path:
        streams = extract_result.get("streams") or []
        if not streams:
            raise RuntimeError("추출된 stream이 없습니다.")
        if stream_index < 0 or stream_index >= len(streams):
            raise IndexError("stream_index가 streams 범위를 벗어났습니다.")

        stream = streams[stream_index]
        stream_url = stream.get("url")
        if not stream_url:
            raise RuntimeError("선택한 stream에 URL이 없습니다.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        headers: dict[str, str] = {}
        referer = stream.get("referer")
        if referer:
            headers["Referer"] = referer

        options = {
            "outtmpl": str(output),
            "http_headers": headers,
            "quiet": False,
            "noplaylist": True,
        }

        with YoutubeDL(options) as ydl:
            ydl.download([stream_url])

        return output
