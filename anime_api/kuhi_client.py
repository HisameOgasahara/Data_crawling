from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


QUALITY_FORMATS = {
    "best": "bestvideo+bestaudio/best",
    "1080": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720": "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480": "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "worst": "worstvideo+worstaudio/worst",
}


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

    def providers_status(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/anime/providers/status",
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

    def _subtitle_language(self, subtitle: dict[str, Any]) -> str | None:
        raw = " ".join(
            str(subtitle.get(key) or "")
            for key in ("srclang", "language", "label")
        ).lower()

        if any(token in raw for token in ("en-us", "en-gb", "english", " eng", "en")):
            return "en"

        if any(token in raw for token in ("ja-jp", "japanese", " jpn", " jp", "ja")):
            return "ja"

        return None

    def _download_subtitles(
        self,
        extract_result: dict[str, Any],
        output_path: str,
        languages: set[str],
    ) -> list[Path]:
        subtitles = extract_result.get("subtitles") or []
        if not subtitles or not languages:
            return []

        output = Path(output_path)
        base = output.with_suffix("")
        saved: list[Path] = []
        seen_languages: set[str] = set()

        for subtitle in subtitles:
            language = self._subtitle_language(subtitle)
            if language not in languages or language in seen_languages:
                continue

            url = subtitle.get("url")
            if not url:
                continue

            fmt = str(subtitle.get("format") or "").strip().lower()
            if not fmt:
                suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
                fmt = suffix or "vtt"

            subtitle_path = Path(f"{base}.{language}.{fmt}")

            headers: dict[str, str] = {}
            referer = subtitle.get("referer")
            if referer:
                headers["Referer"] = referer

            response = requests.get(
                url,
                headers=headers,
                timeout=30,
            )

            if not response.ok or not response.content:
                continue

            subtitle_path.write_bytes(response.content)
            saved.append(subtitle_path)
            seen_languages.add(language)

        return saved

    def _download_stream(
        self,
        extract_result: dict[str, Any],
        output_path: str,
        stream_index: int = 0,
        quality: str = "best",
        concurrent_fragments: int = 8,
    ) -> Path:
        streams = extract_result.get("streams") or []
        downloadable_streams = [
            stream
            for stream in streams
            if stream.get("type") in {"hls", "mp4", "dash"}
        ]

        if not downloadable_streams:
            raise RuntimeError("다운로드 가능한 hls/mp4/dash stream이 없습니다.")

        if stream_index < 0 or stream_index >= len(downloadable_streams):
            raise IndexError("stream_index가 다운로드 가능한 streams 범위를 벗어났습니다.")

        if quality not in QUALITY_FORMATS:
            raise ValueError(
                f"quality는 {', '.join(QUALITY_FORMATS)} 중 하나여야 합니다."
            )

        stream = downloadable_streams[stream_index]
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
            "format": QUALITY_FORMATS[quality],
            "concurrent_fragment_downloads": concurrent_fragments,
        }

        with YoutubeDL(options) as ydl:
            ydl.download([stream_url])

        return output

    def download(
        self,
        extract_result: dict[str, Any],
        output_path: str,
        stream_index: int = 0,
        retry_other_providers: bool = True,
        quality: str = "best",
        subtitle_english: bool = False,
        subtitle_japanese: bool = False,
        concurrent_fragments: int = 8,
    ) -> Path:
        current_provider = extract_result.get("provider")
        subtitle_languages: set[str] = set()

        if subtitle_english:
            subtitle_languages.add("en")

        if subtitle_japanese:
            subtitle_languages.add("ja")

        try:
            saved = self._download_stream(
                extract_result,
                output_path=output_path,
                stream_index=stream_index,
                quality=quality,
                concurrent_fragments=concurrent_fragments,
            )
            self._download_subtitles(
                extract_result,
                output_path=output_path,
                languages=subtitle_languages,
            )
            return saved
        except (DownloadError, RuntimeError) as first_error:
            if not retry_other_providers:
                raise

            anilist_id = extract_result.get("anilistId")
            episode = extract_result.get("episode", 1)
            audio = extract_result.get("type", "sub")

            if not anilist_id:
                raise first_error

            ranking = self.providers_status().get("ranking") or []
            attempted = {current_provider} if current_provider else set()
            last_error: Exception = first_error

            print(
                f"[fallback] {current_provider or 'unknown'} 다운로드 실패. "
                "다른 provider를 순서대로 시도합니다."
            )

            for provider in ranking:
                if provider in attempted:
                    continue

                attempted.add(provider)

                try:
                    candidate = self.extract(
                        anilist_id,
                        episode=episode,
                        audio=audio,
                        provider=provider,
                    )
                except requests.RequestException as error:
                    last_error = error
                    print(f"[fallback] {provider}: extract 실패")
                    continue

                actual_provider = candidate.get("provider")
                if actual_provider in attempted and actual_provider != provider:
                    continue

                if actual_provider:
                    attempted.add(actual_provider)

                try:
                    print(
                        f"[fallback] {provider}: "
                        f"실제 provider={actual_provider or provider} 다운로드 시도"
                    )
                    saved = self._download_stream(
                        candidate,
                        output_path=output_path,
                        stream_index=stream_index,
                        quality=quality,
                    )
                    self._download_subtitles(
                        candidate,
                        output_path=output_path,
                        languages=subtitle_languages,
                    )
                    return saved
                except (DownloadError, RuntimeError) as error:
                    last_error = error
                    print(f"[fallback] {actual_provider or provider}: 다운로드 실패")

            raise RuntimeError(
                "사용 가능한 provider에서 모두 다운로드에 실패했습니다."
            ) from last_error
