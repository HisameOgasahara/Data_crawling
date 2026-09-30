# HBR YouTube OCR

Heaven Burns Red 영상에서 화면의 화자명/대사를 시간축으로 추출하는 실험입니다.

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/Data_crawling/blob/main/hbr_youtube_ocr/hbr_youtube_ocr_colab.ipynb?skip_cache=true)

- YouTube / 로컬 업로드 / Google Drive 영상 입력
- 전체 WAV 오디오 추출
- OpenCV 고정 ROI
- MangaOCR
- 화자명/대사 timestamp 매칭

## Local ROI / OCR test

`local_roi_ocr.py`는 로컬 OpenCV 창에서 영상 전체를 보면서 ROI가 계속 같은 위치에 유지되는지 확인하기 위한 도구입니다.

실행:

```bash
pip install opencv-python manga-ocr pillow
python local_roi_ocr.py "D:\\video\\hbr.mp4"
```

영상이 큰 경우 기본적으로 가로 1280px에 맞춰 축소해서 표시합니다. 원본 영상은 축소하지 않으며 OCR도 원본 해상도의 ROI를 사용합니다. speaker/dialogue ROI는 수동 지정하지 않고 OpenCV로 매 프레임 자동 검출합니다.

```bash
python local_roi_ocr.py "D:\\video\\hbr.mp4" --width 960
```

키:

- `Space`: 재생 / 일시정지
- `a` / `d`: 이전 / 다음 1프레임
- `j` / `l`: 5초 뒤 / 앞으로 이동
- `1`: 현재 프레임에서 speaker ROI 지정
- `2`: 현재 프레임에서 dialogue ROI 지정
- `o`: 현재 프레임의 두 ROI만 MangaOCR 실행
- `q`: 종료

OCR 모델은 시작할 때 로드하지 않고 처음 `o`를 눌렀을 때만 CPU로 로드합니다.
