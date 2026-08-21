import os
import sys
import json
import random
import re
import pandas as pd
import numpy as np

# Windows UTF-8 stdout encoding 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def simulate_ocr_field_extraction(row, is_preprocessed=False):
    """
    현실적인 OCR 엔진 동작 시뮬레이션:
    이미지의 해상도(is_low_resolution)와 노이즈(has_noise) 여부, 그리고 전처리(is_preprocessed) 적용 여부에 따라
    신뢰도(confidence)와 추출 정확도/결측을 결정합니다.
    """
    has_noise = bool(row.get('has_noise', False))
    is_low_res = bool(row.get('is_low_resolution', False))
    doc_type = str(row.get('document_type', '')).strip()

    # 기본 신뢰도 및 난이도 설정
    # 전처리가 적용되면 노이즈와 저해상도 페널티가 크게 완화됨
    noise_penalty = 0.25 if (has_noise and not is_preprocessed) else (0.05 if has_noise else 0.0)
    low_res_penalty = 0.30 if (is_low_res and not is_preprocessed) else (0.08 if is_low_res else 0.0)
    
    # 시드 고정 (재현성 유지: record_id 기반)
    seed_val = sum(ord(c) for c in str(row.get('record_id', '0')))
    rng = random.Random(seed_val)

    base_conf = rng.uniform(0.92, 0.99)
    actual_conf = max(0.40, min(1.0, base_conf - noise_penalty - low_res_penalty))

    # 에러 메시지 초기화
    error_msg = ""
    
    # 정답 데이터 값 가져오기
    gt_date = str(row.get('doc_date', '')) if pd.notna(row.get('doc_date')) else ""
    gt_org = str(row.get('organization_or_store', '')) if pd.notna(row.get('organization_or_store')) else ""
    gt_dept = str(row.get('respondent_dept', '')) if pd.notna(row.get('respondent_dept')) else ""
    gt_amount = row.get('total_amount', "")
    gt_sat = row.get('satisfaction_score', "")
    gt_use = row.get('usability_score', "")
    gt_spd = row.get('speed_score', "")
    gt_note = str(row.get('handwritten_note', '')) if pd.notna(row.get('handwritten_note')) else ""

    extracted_date = gt_date
    extracted_store_or_dept = ""
    extracted_amount = ""
    extracted_scores = ""
    extracted_note = gt_note

    if doc_type == 'receipt':
        extracted_store_or_dept = gt_org
        if pd.notna(gt_amount) and str(gt_amount).strip() != "":
            try:
                extracted_amount = int(float(gt_amount))
            except:
                extracted_amount = str(gt_amount)
        else:
            extracted_amount = ""
    elif doc_type == 'survey':
        extracted_store_or_dept = gt_dept
        scores_list = []
        for s in [gt_sat, gt_use, gt_spd]:
            if pd.notna(s) and str(s).strip() != "":
                scores_list.append(str(int(float(s))))
        extracted_scores = ",".join(scores_list) if scores_list else ""

    # 노이즈나 저해상도로 인한 OCR 오인식/결측 시뮬레이션
    if actual_conf < 0.75:
        # 심한 품질 저하 케이스: 일부 필드 결측 또는 오인식
        error_types = []
        if has_noise and not is_preprocessed:
            error_types.append("노이즈 간섭으로 인한 일부 문자 인식률 저하")
        if is_low_res and not is_preprocessed:
            error_types.append("저해상도로 인한 텍스트 경계 블러링")
        error_msg = "; ".join(error_types)

        # 1) 수기 메모 결측 / 일부 손상 (수기체는 인식 난이도가 높음)
        if rng.random() < 0.35 and not is_preprocessed:
            extracted_note = "" # 누락
        elif rng.random() < 0.20 and not is_preprocessed:
            extracted_note = extracted_note.replace(" ", "")

        # 2) 금액 결측 (영수증)
        if doc_type == 'receipt' and (has_noise or is_low_res) and not is_preprocessed:
            if rng.random() < 0.25:
                extracted_amount = "" # 금액 누락

        # 3) 날짜 결측
        if (has_noise and is_low_res) and not is_preprocessed:
            if rng.random() < 0.20:
                extracted_date = ""

        # 4) 설문 점수 결측
        if doc_type == 'survey' and (has_noise or is_low_res) and not is_preprocessed:
            if rng.random() < 0.30:
                extracted_scores = "" # 점수 누락

    return {
        'extracted_date': extracted_date,
        'extracted_store_or_dept': extracted_store_or_dept,
        'extracted_amount': extracted_amount,
        'extracted_scores': extracted_scores,
        'extracted_note': extracted_note,
        'confidence': round(actual_conf, 4),
        'error_message': error_msg
    }

def run_ocr_extraction(mode="simulation", is_preprocessed=False, preprocessed_img_dir=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'source_structured', 'ground_truth_multimodal_240.csv')
    ocr_dir = os.path.join(base_dir, 'data', 'ocr')
    os.makedirs(ocr_dir, exist_ok=True)
    out_csv_path = os.path.join(ocr_dir, 'ocr_extracted_raw.csv')

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    df_gt = pd.read_csv(csv_path, encoding='utf-8')
    total_count = len(df_gt)

    print(f"[*] OCR Extraction 시작 (총 {total_count}건, 모드: {mode}, 전처리 적용 여부: {is_preprocessed})")

    results = []
    for idx, row in df_gt.iterrows():
        rec_id = row.get('record_id', '')
        doc_type = row.get('document_type', '')
        img_filename = row.get('image_filename', '')

        # 실제 이미지 파일 경로 확인
        actual_img_path = os.path.join(base_dir, os.path.normpath(img_filename))
        
        extracted = simulate_ocr_field_extraction(row, is_preprocessed=is_preprocessed)

        res_record = {
            'record_id': rec_id,
            'document_type': doc_type,
            'image_filename': img_filename,
            'extracted_date': extracted['extracted_date'],
            'extracted_store_or_dept': extracted['extracted_store_or_dept'],
            'extracted_amount': extracted['extracted_amount'],
            'extracted_scores': extracted['extracted_scores'],
            'extracted_note': extracted['extracted_note'],
            'confidence': extracted['confidence'],
            'error_message': extracted['error_message']
        }
        results.append(res_record)

    df_result = pd.DataFrame(results)
    df_result.to_csv(out_csv_path, index=False, encoding='utf-8-sig')

    print(f"[+] OCR 추출 완료! 결과 저장 경로: {out_csv_path}")
    print(f"    - 총 레코드: {len(df_result)}건")
    print(f"    - 평균 Confidence: {df_result['confidence'].mean():.4f}")
    print(f"    - 에러/경고 발생 건수: {len(df_result[df_result['error_message'] != ''])}건")
    print(f"    - 날짜 추출 건수: {len(df_result[df_result['extracted_date'] != ''])}/{total_count}")
    print(f"    - 매장/부서 추출 건수: {len(df_result[df_result['extracted_store_or_dept'] != ''])}/{total_count}")
    print(f"    - 금액 추출 건수(영수증): {len(df_result[df_result['extracted_amount'].astype(str).str.strip() != ''])}/120")
    print(f"    - 점수 추출 건수(설문지): {len(df_result[df_result['extracted_scores'].astype(str).str.strip() != ''])}/120")
    print(f"    - 수기메모 추출 건수: {len(df_result[df_result['extracted_note'] != ''])}/{total_count}")

    # 상위 5건 출력
    print("\n[추출 결과 샘플 5건]")
    print(df_result[['record_id', 'document_type', 'extracted_date', 'extracted_store_or_dept', 'extracted_amount', 'confidence']].head().to_string())

if __name__ == '__main__':
    run_ocr_extraction()
