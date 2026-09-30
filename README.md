# Data_crawling

크롤링/수집 실험을 서비스별 독립 폴더로 관리합니다.

## anime_api

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/Data_crawling/blob/main/anime_api/anime_api_colab.ipynb?skip_cache=true)

[aryaniiil/anime-api (Kuhi)](https://github.com/aryaniiil/anime-api)를 Colab에서 테스트하기 위한 래퍼입니다.

- CPU Colab 런타임 사용 가능
- AniList 기반 애니 검색
- 검색 결과 썸네일 표시
- Kuhi `/anime/extract` 기반 영상 소스 추출
- `yt-dlp` 기반 영상 다운로드
- Cloudflare Tunnel을 통한 검색/동영상 뷰어
- Google Drive 미사용

노트북: `anime_api/anime_api_colab.ipynb`

## hbr_youtube_ocr

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/Data_crawling/blob/main/hbr_youtube_ocr/hbr_youtube_ocr_colab.ipynb?skip_cache=true)

- CPU Colab 런타임 기준
- YouTube 영상 다운로드
- 전체 WAV 오디오 추출
- OpenCV 고정 ROI + MangaOCR
- 화자명/대사 timestamp 매칭

노트북: `hbr_youtube_ocr/hbr_youtube_ocr_colab.ipynb`
