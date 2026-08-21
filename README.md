# Project 2: Multimodal Vision OCR Dashboard

> **영수증 및 수기 설문지 이미지 기반 멀티모달 OCR 추출, 결측치 보정, 품질 평가 및 대시보드 웹 애플리케이션**

[![GitHub Pages](https://img.shields.io/badge/Live_Demo-GitHub_Pages-00C9A7?style=for-the-badge&logo=github)](https://songjaemin94.github.io/project2-vision-ocr-dashboard/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Preprocessing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

---

## 🌐 라이브 대시보드 (Live Demo)

- **GitHub Pages 배포 URL**: 👉 **[https://songjaemin94.github.io/project2-vision-ocr-dashboard/](https://songjaemin94.github.io/project2-vision-ocr-dashboard/)**

---

## 📌 프로젝트 소개

본 프로젝트는 240장의 멀티모달 이미지(영수증 120장, 수기 설문지 120장)와 기준 정답 데이터(`ground_truth_multimodal_240.csv`)를 기반으로, **OCR 추출 → OpenCV 4단계 전처리 → 정답 데이터 대비 정확도 평가 → 결측치 자동 보간 → 인터랙티브 대시보드 구축 및 배포**에 이르는 전 과정을 자동화한 실습 프로젝트입니다.

---

## 📂 폴더 구조

```text
project2-vision-ocr-dashboard/
├── app/
│   └── vision_dashboard.py            # Streamlit 로컬 대시보드 실행 파일
├── data/
│   ├── source_structured/             # 원본 정답 기준 데이터셋 (240건)
│   ├── input_images/                  # 원본 영수증(120장) 및 설문지(120장) 이미지
│   ├── ocr/                           # OCR 추출 raw 데이터 및 OpenCV 전처리 이미지
│   └── processed/                     # 결측치 보정 완료된 최종 정제 데이터셋 (Excel/CSV/JSON)
├── reports/                           # 데이터 검증 및 OCR 품질 분석 보고서 (Excel/TXT)
├── scripts/
│   ├── validate_images.py             # 1단계: 원본 데이터와 이미지 1:1 매칭 검증
│   ├── ocr_extractor.py               # 2단계: Vision OCR 추출기
│   ├── preprocess_and_ocr.py          # 3단계: OpenCV 4단계 전처리 및 인식률 개선
│   ├── evaluate_ocr_quality.py        # 4단계: 정답 데이터 대비 품질 평가 리포트 생성
│   └── clean_and_impute.py            # 5단계: 4대 결측치 보정 정책 적용
├── index.html                         # GitHub Pages 배포용 정적 인터랙티브 웹 대시보드
├── README.md
└── requirements.txt
```

---

## ⚙️ 파이프라인 단계별 설명

1. **데이터 & 이미지 정합성 검증 (`scripts/validate_images.py`)**
   - 240건의 레코드와 이미지 파일 존재 여부, 확장자 유효성, 고유 ID 중복 검사 (매칭 성공률 100%)
2. **OpenCV 4단계 전처리 파이프라인 (`scripts/preprocess_and_ocr.py`)**
   - Grayscale 변환 → CLAHE 대비 향상 → FastNlMeans 노이즈 제거 → Adaptive Gaussian Thresholding 적응형 이진화
3. **정답 데이터 대비 품질 평가 (`scripts/evaluate_ocr_quality.py`)**
   - 문서 유형별, 필드별(금액, 점수, 메모, 일자, 상호/부서), 이미지 환경별(노이즈/저해상도) 정확도 분석
4. **결측치 보간 정책 적용 (`scripts/clean_and_impute.py`)**
   - 일자: 정답 DB 매칭 복원
   - 금액: 영수증 문서 원본 검증 복원 (`amount_imputed=True`)
   - 설문점수: 소속 부서별 평균 점수 보간 (`score_imputed=True`)
   - 수기메모: 누락 시 `'확인필요'` 지정 (`note_imputed=True`)

---

## 🚀 로컬 실행 방법 (Streamlit)

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Streamlit 대시보드 실행
streamlit run app/vision_dashboard.py
```
브라우저에서 `http://localhost:8501` 접속
