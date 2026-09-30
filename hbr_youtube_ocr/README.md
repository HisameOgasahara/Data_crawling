# HBR YouTube OCR

Heaven Burns Red YouTube 영상에서 화면의 화자명/대사를 시간축으로 추출하는 Colab 실험입니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/Data_crawling/blob/main/hbr_youtube_ocr/hbr_youtube_ocr_colab.ipynb?skip_cache=true)

현재 범위:

- `pytubefix`로 YouTube 영상 다운로드
- `ffmpeg`로 영상 전체 오디오를 WAV로 추출
- OpenCV 고정 ROI로 화자명/대사 영역 지정
- 기존 일본어 OCR 실험과 동일하게 `MangaOCR` 사용
- 일정 시간 간격으로 OCR 후 화자명 + 대사를 timestamp 기준으로 매칭
- `ocr_raw_samples.csv`, `dialogue_segments.csv` 저장
- 캐릭터별 음성 분리/VAD는 아직 하지 않음

기본 테스트 URL:

`https://www.youtube.com/watch?v=zHN25OVshIY`
