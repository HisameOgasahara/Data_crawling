# anime_api

Kuhi `aryaniiil/anime-api`를 Colab CPU에서 테스트하기 위한 독립 폴더입니다.

## 파일

- `anime_api_colab.ipynb`: 설치, `@param` 입력, 검색 썸네일, 추출, 다운로드, Cloudflare Tunnel 실행
- `kuhi_client.py`: Kuhi REST API 호출과 yt-dlp 다운로드
- `colab_app.py`: 원본 Kuhi FastAPI 앱에 `/viewer` 검색/동영상 뷰어 추가
- `requirements.txt`: 래퍼 의존성

원본 Kuhi 코드는 이 저장소에 복사하지 않고 Colab 실행 시 최신 `main`을 clone합니다.

Google Drive는 사용하지 않습니다. 다운로드 결과 기본 경로는 `/content/downloads/`입니다.
