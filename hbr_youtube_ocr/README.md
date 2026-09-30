# HBR YouTube OCR

Heaven Burns Red 영상에서 화자명/대사를 시간축으로 추출하는 실험입니다.

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/Data_crawling/blob/main/hbr_youtube_ocr/hbr_youtube_ocr_colab.ipynb?skip_cache=true)

## Local fixed-ROI OCR

자동 ROI 검출은 사용하지 않습니다.

처음 한 번 영상에서 화자명 영역과 대사 영역을 수동으로 지정해 `roi.json`에 정규화 좌표로 저장하고, 이후 같은 UI 배치의 영상은 저장된 ROI를 그대로 사용합니다.

### 1. ROI 캘리브레이션

```bash
pip install opencv-python manga-ocr pillow
python local_roi_ocr.py "D:\\video\\hbr.mp4" --mode calibrate --time 60
```

키:

- `Space`: 재생 / 일시정지
- `a` / `d`: 이전 / 다음 1프레임
- `j` / `l`: 5초 뒤 / 앞으로
- `r`: 현재 프레임에서 speaker ROI → dialogue ROI 순서로 선택하고 `roi.json` 저장
- `q`: 종료

ROI는 원본 해상도의 픽셀 좌표가 아니라 0~1 정규화 좌표로 저장되므로 동일한 UI 비율이면 다른 해상도에도 그대로 적용됩니다.

### 2. 전체 영상 OCR

```bash
python local_roi_ocr.py "D:\\video\\hbr.mp4" --mode extract --interval 0.5
```

처리 순서:

1. `roi.json` 로드
2. 일정 시간 간격으로 프레임 샘플링
3. speaker/dialogue ROI crop
4. grayscale + threshold
5. MangaOCR
6. speaker OCR을 `character_names.txt`와 fuzzy match
7. 같은 캐릭터의 연속된 유사 대사를 하나의 구간으로 병합
8. 캐릭터별 TXT 저장

출력:

```text
hbr_ocr_output/
├── ocr_samples.csv
├── dialogue_segments.csv
└── characters/
    ├── 茅森月歌.txt
    ├── 和泉ユキ.txt
    └── ...
```

기본 ROI 파일: `hbr_youtube_ocr/roi.json`

기본 캐릭터 이름 파일: `hbr_youtube_ocr/character_names.txt`
