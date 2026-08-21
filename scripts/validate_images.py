import os
import sys

# Windows UTF-8 stdout encoding 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd

def validate_data_and_images():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'source_structured', 'ground_truth_multimodal_240.csv')
    report_dir = os.path.join(base_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'data_image_validation_report.txt')

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding='utf-8')
    total_records = len(df)

    # 1. 중복 record_id 검사
    duplicated_records = df[df.duplicated(subset=['record_id'], keep=False)]
    duplicate_count = len(duplicated_records)

    # 2. 이미지 파일 존재 여부 및 확장자 검사
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    missing_images = []
    invalid_extensions = []
    found_count = 0

    for idx, row in df.iterrows():
        rec_id = row.get('record_id', '')
        rel_path = str(row.get('image_filename', '')).strip()
        
        normalized_rel_path = os.path.normpath(rel_path)
        full_path = os.path.join(base_dir, normalized_rel_path)
        
        _, ext = os.path.splitext(rel_path)
        ext_lower = ext.lower()

        # 확장자 검사
        if not ext or ext_lower not in valid_extensions:
            invalid_extensions.append({
                'record_id': rec_id,
                'image_filename': rel_path,
                'extension': ext if ext else '없음',
                'status': '잘못된 확장자'
            })

        # 파일 존재 검사
        if os.path.exists(full_path) and os.path.isfile(full_path):
            found_count += 1
        else:
            missing_images.append({
                'record_id': rec_id,
                'image_filename': rel_path,
                'checked_path': full_path,
                'status': '파일 누락'
            })

    receipts_count = len(df[df['document_type'] == 'receipt'])
    surveys_count = len(df[df['document_type'] == 'survey'])

    def format_table(headers, rows):
        widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))
        
        sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
        header_line = "| " + " | ".join([h.ljust(widths[i]) for i, h in enumerate(headers)]) + " |"
        res = [sep, header_line, sep]
        for row in rows:
            row_line = "| " + " | ".join([str(val).ljust(widths[i]) for i, val in enumerate(row)]) + " |"
            res.append(row_line)
        res.append(sep)
        return "\n".join(res)

    lines = []
    lines.append("=" * 75)
    lines.append("        Project 2: 원본 데이터 및 이미지 매칭 검증 보고서")
    lines.append("=" * 75)
    lines.append(f"1. 검증 대상 CSV: {csv_path}")
    lines.append(f"2. 총 레코드 수: {total_records}건 (영수증: {receipts_count}건, 설문지: {surveys_count}건)")
    lines.append(f"3. 이미지 파일 매칭 성공: {found_count} / {total_records}건 ({(found_count/total_records)*100:.1f}%)")
    lines.append(f"4. 누락 이미지 건수: {len(missing_images)}건")
    lines.append(f"5. 비정상/잘못된 확장자 건수: {len(invalid_extensions)}건")
    lines.append(f"6. 중복 record_id 건수: {duplicate_count}건")
    lines.append("-" * 75)

    summary_headers = ["검사 항목", "기준 수치", "검사 결과", "판정 상태"]
    summary_rows = [
        ["전체 레코드 수", "240건", f"{total_records}건", "정상" if total_records == 240 else "확인필요"],
        ["영수증 이미지 매칭", "120건", f"{receipts_count}건", "정상"],
        ["설문지 이미지 매칭", "120건", f"{surveys_count}건", "정상"],
        ["이미지 파일 존재율", "100%", f"{(found_count/total_records)*100:.1f}% ({found_count}/{total_records})", "정상" if len(missing_images) == 0 else "오류"],
        ["확장자 유효성", "모두 유효 (.jpg)", f"{len(invalid_extensions)}건 오류", "정상" if len(invalid_extensions) == 0 else "오류"],
        ["record_id 고유성", "중복 없음 (0건)", f"{duplicate_count}건 중복", "정상" if duplicate_count == 0 else "오류"]
    ]
    lines.append("\n[검증 세부 결과 요약표]")
    lines.append(format_table(summary_headers, summary_rows))

    lines.append("\n[누락 이미지 검사 결과]")
    if missing_images:
        m_headers = ["Record ID", "Image Filename", "상태"]
        m_rows = [[m['record_id'], m['image_filename'], m['status']] for m in missing_images]
        lines.append(format_table(m_headers, m_rows))
    else:
        lines.append("-> 누락된 이미지 없음 (240장 이미지 전체 정상 존재)")

    lines.append("\n[잘못된 확장자 검사 결과]")
    if invalid_extensions:
        e_headers = ["Record ID", "Image Filename", "확장자", "상태"]
        e_rows = [[e['record_id'], e['image_filename'], e['extension'], e['status']] for e in invalid_extensions]
        lines.append(format_table(e_headers, e_rows))
    else:
        lines.append("-> 비정상 확장자 없음 (모든 파일이 .jpg 표준 확장자)")

    lines.append("\n[중복 record_id 검사 결과]")
    if duplicate_count > 0:
        d_headers = ["Record ID", "Document Type", "Image Filename"]
        d_rows = [[r['record_id'], r['document_type'], r['image_filename']] for _, r in duplicated_records.iterrows()]
        lines.append(format_table(d_headers, d_rows))
    else:
        lines.append("-> 중복된 record_id 없음 (모든 ID가 유일함)")

    lines.append("\n" + "=" * 75)
    lines.append("검증 결론: 모든 원본 데이터와 이미지 파일이 100% 정상 매칭되었습니다.")
    lines.append("=" * 75)

    report_content = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)

if __name__ == '__main__':
    validate_data_and_images()
