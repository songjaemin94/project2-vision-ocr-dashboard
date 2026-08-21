import os
import sys
import shutil
import zipfile

# Windows UTF-8 stdout encoding 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def package_project():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    final_dir = os.path.join(base_dir, 'final_output')
    zip_path = os.path.join(base_dir, 'final_output.zip')

    # 기존 final_output 초기화 및 재생성
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(os.path.join(final_dir, 'app'), exist_ok=True)
    os.makedirs(os.path.join(final_dir, 'reports'), exist_ok=True)

    print("[*] 1단계: 필수 산출물 복사 중...")

    # 1. 정제된 OCR 데이터 Excel
    src_xlsx = os.path.join(base_dir, 'data', 'processed', 'ocr_cleaned_dataset.xlsx')
    dst_xlsx = os.path.join(final_dir, 'ocr_cleaned_dataset.xlsx')
    shutil.copy2(src_xlsx, dst_xlsx)
    print(f"  ✓ 정제 데이터 Excel: {os.path.basename(dst_xlsx)}")

    # 2. OCR 품질 리포트 (Excel & TXT)
    src_report_xlsx = os.path.join(base_dir, 'reports', 'ocr_quality_report.xlsx')
    dst_report_xlsx = os.path.join(final_dir, 'reports', 'ocr_quality_report.xlsx')
    shutil.copy2(src_report_xlsx, dst_report_xlsx)

    src_report_txt = os.path.join(base_dir, 'reports', 'ocr_quality_summary.txt')
    dst_report_txt = os.path.join(final_dir, 'reports', 'ocr_quality_summary.txt')
    shutil.copy2(src_report_txt, dst_report_txt)
    print(f"  ✓ OCR 품질 리포트: ocr_quality_report.xlsx, ocr_quality_summary.txt")

    # 3. 대시보드 실행 파일 (Streamlit app & Web index.html)
    src_app = os.path.join(base_dir, 'app', 'vision_dashboard.py')
    dst_app = os.path.join(final_dir, 'app', 'vision_dashboard.py')
    shutil.copy2(src_app, dst_app)

    src_index = os.path.join(base_dir, 'index.html')
    dst_index = os.path.join(final_dir, 'index.html')
    shutil.copy2(src_index, dst_index)
    print(f"  ✓ 대시보드 실행 파일: app/vision_dashboard.py, index.html")

    # 4. 화면 검증 리포트
    src_check = os.path.join(base_dir, 'reports', 'dashboard_visual_check.txt')
    dst_check = os.path.join(final_dir, 'reports', 'dashboard_visual_check.txt')
    shutil.copy2(src_check, dst_check)
    print(f"  ✓ 화면 검증 리포트: reports/dashboard_visual_check.txt")

    # 5. 실행 방법 README
    readme_content = """# Project 2: 멀티모달 OCR 분석 대시보드 최종 산출물 패키지

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
"""
    with open(os.path.join(final_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ 실행 방법 README: README.md")

    # ZIP 파일 압축
    print("\n[*] 2단계: final_output 폴더 ZIP 압축 진행 중...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(final_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, final_dir)
                zipf.write(file_path, arcname=os.path.join('final_output', rel_path))

    zip_size_kb = os.path.getsize(zip_path) / 1024
    print(f"[+] ZIP 압축 완료! 파일 경로: {zip_path} (크기: {zip_size_kb:.1f} KB)")

if __name__ == '__main__':
    package_project()
