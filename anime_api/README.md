# anime_api

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/Data_crawling/blob/main/anime_api/anime_api_colab.ipynb?skip_cache=true)

[aryaniiil/anime-api (Kuhi)](https://github.com/aryaniiil/anime-api)를 Colab에서 테스트하기 위한 래퍼입니다.

- CPU Colab 런타임 사용 가능
- Google Drive 미사용
- AniList 검색 및 썸네일 표시
- Kuhi 스트림 추출
- yt-dlp 다운로드
- Cloudflare Tunnel 기반 검색/동영상 뷰어

## Files

- `anime_api_colab.ipynb`: Colab 실행 노트북
- `kuhi_client.py`: 검색/추출/다운로드 클라이언트
- `colab_app.py`: Cloudflare로 노출할 검색/동영상 뷰어
- `requirements.txt`: 래퍼 의존성
