# Project 2: 멀티모달 OCR 분석 대시보드 최종 산출물 패키지

## 📌 산출물 구성
1. **정제된 OCR 데이터 Excel**: `ocr_cleaned_dataset.xlsx` (240건 정제 및 결측치 보정 데이터)
2. **OCR 품질 리포트**: `reports/ocr_quality_report.xlsx` & `reports/ocr_quality_summary.txt`
3. **화면 검증 리포트**: `reports/dashboard_visual_check.txt`
4. **Streamlit 대시보드 실행 파일**: `app/vision_dashboard.py`
5. **웹 대시보드 (GitHub Pages)**: `index.html`

---

## 🚀 실행 방법

### 방법 1: 웹 브라우저 실시간 접속 (설치 불필요)
👉 **[https://songjaemin94.github.io/project2-vision-ocr-dashboard/](https://songjaemin94.github.io/project2-vision-ocr-dashboard/)**

### 방법 2: 로컬 Streamlit 대시보드 실행
```bash
# 의존성 설치
pip install streamlit plotly pandas openpyxl pillow opencv-python

# 대시보드 실행
streamlit run app/vision_dashboard.py
```
브라우저에서 `http://localhost:8501`로 접속합니다.
