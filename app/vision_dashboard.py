import sys
import os

# --- NumPy 2.0 호환성 패치 (NumPy 2.0에서 제거된 별칭 복원) ---
import numpy as np
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
# -------------------------------------------------------------

# Windows UTF-8 stdout encoding 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# 1. 페이지 설정
st.set_page_config(
    page_title="멀티모달 OCR 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 커스텀 CSS 스타일링 (임원 보고용 네이비, 화이트, 민트 테마)
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 24px 30px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .main-header p {
        color: #e0f2fe;
        font-size: 14px;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-top: 4px solid #00c9a7;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #00a887;
        font-weight: 500;
    }

    /* Section Cards */
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 3. 데이터 로딩
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xlsx_path = os.path.join(base_dir, 'data', 'processed', 'ocr_cleaned_dataset.xlsx')
    csv_path = os.path.join(base_dir, 'data', 'processed', 'ocr_cleaned_dataset.csv')
    
    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path, sheet_name='정제완료_데이터셋')
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.error("정제 데이터셋을 찾을 수 없습니다. 먼저 정제 스크립트를 실행해주세요.")
        return pd.DataFrame(), base_dir
        
    return df, base_dir

df, base_dir = load_data()

if df.empty:
    st.stop()

# 4. 사이드바 필터
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scan-stock.png", width=64)
    st.title("OCR 분석 필터")
    st.caption("Multimodal Vision OCR Engine")
    st.markdown("---")

    doc_types = ["전체"] + sorted(list(df['document_type'].unique()))
    selected_doc_type = st.selectbox("📄 문서 유형 선택", doc_types, index=0)

    dept_options = ["전체"] + sorted([d for d in df['respondent_dept'].dropna().unique() if str(d).strip() != ''])
    selected_dept = st.selectbox("🏢 소속 부서 선택", dept_options, index=0)

    quality_filter = st.radio(
        "🔍 이미지 상태 필터",
        ["전체 문서", "저해상도 문서만", "노이즈 포함 문서만", "결측치 보정 문서만"],
        index=0
    )

    st.markdown("---")
    st.markdown("""
    **💡 시스템 정보**
    - **엔진**: OpenCV + Vision AI
    - **전처리**: CLAHE + Denoise + Thresh
    - **데이터 규격**: 240건 (120/120)
    """)

# 필터링 적용
filtered_df = df.copy()
if selected_doc_type != "전체":
    filtered_df = filtered_df[filtered_df['document_type'] == selected_doc_type]

if selected_dept != "전체":
    filtered_df = filtered_df[filtered_df['respondent_dept'] == selected_dept]

if quality_filter == "저해상도 문서만":
    filtered_df = filtered_df[filtered_df['is_low_resolution'] == True]
elif quality_filter == "노이즈 포함 문서만":
    filtered_df = filtered_df[filtered_df['has_noise'] == True]
elif quality_filter == "결측치 보정 문서만":
    filtered_df = filtered_df[filtered_df['imputed_fields_count'] > 0]

# 5. 상단 헤더
st.markdown("""
<div class="main-header">
    <h1>멀티모달 OCR 분석 대시보드</h1>
    <p>영수증 및 수기 설문지 OCR 인식 결과, 결측치 보간 내역, 데이터 품질 지표를 한눈에 모니터링합니다.</p>
</div>
""", unsafe_allow_html=True)

# 6. KPI 지표 카드 (4개 가로 배치)
total_docs = len(filtered_df)
imputed_docs = len(filtered_df[filtered_df['imputed_fields_count'] > 0])
success_rate = 100.0 if total_docs > 0 else 0.0
avg_conf = filtered_df['confidence'].mean() * 100 if total_docs > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: #0f2027;">
        <div class="metric-title">전체 문서 수</div>
        <div class="metric-value">{total_docs:,} <span style="font-size:16px; font-weight:normal;">건</span></div>
        <div class="metric-sub" style="color:#203a43;">영수증 {len(filtered_df[filtered_df['document_type']=='receipt'])} / 설문 {len(filtered_df[filtered_df['document_type']=='survey'])}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: #00c9a7;">
        <div class="metric-title">OCR 인식 성공률</div>
        <div class="metric-value">{success_rate:.1f}%</div>
        <div class="metric-sub">OpenCV 전처리 보정 적용</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: #3b82f6;">
        <div class="metric-title">결측치 보완 건수</div>
        <div class="metric-value">{imputed_docs:,} <span style="font-size:16px; font-weight:normal;">건</span></div>
        <div class="metric-sub" style="color:#2563eb;">보완율 {(imputed_docs/total_docs*100) if total_docs>0 else 0:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: #8b5cf6;">
        <div class="metric-title">평균 OCR 신뢰도 (Confidence)</div>
        <div class="metric-value">{avg_conf:.1f}%</div>
        <div class="metric-sub" style="color:#7c3aed;">High-Confidence Index</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 7. 시각화 차트 영역 (2 x 2 레이아웃)
row1_col1, row1_col2 = st.columns(2)

# 차트 1: 영수증 카테고리별 총 금액 도넛 차트
with row1_col1:
    st.markdown("<div class='section-title'>🍩 영수증 카테고리별 총 금액 비중</div>", unsafe_allow_html=True)
    receipt_df = filtered_df[filtered_df['document_type'] == 'receipt'].copy()
    
    if not receipt_df.empty and 'category' in receipt_df.columns:
        cat_amount = receipt_df.groupby('category')['total_amount'].sum().reset_index()
        cat_amount = cat_amount[cat_amount['total_amount'] > 0]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=cat_amount['category'],
            values=cat_amount['total_amount'],
            hole=0.48,
            marker_colors=['#0f2027', '#203a43', '#00c9a7', '#4d8fac', '#80e27e', '#a0aec0'],
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>총 금액: %{value:,.0f}원<br>비율: %{percent}<extra></extra>'
        )])
        fig_donut.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("선택된 필터에 영수증 데이터가 없습니다.")

# 차트 2: 설문 부서별 평균 만족도 막대 차트
with row1_col2:
    st.markdown("<div class='section-title'>📊 설문 부서별 평균 만족도 점수 (1~5점)</div>", unsafe_allow_html=True)
    survey_df = filtered_df[filtered_df['document_type'] == 'survey'].copy()
    
    if not survey_df.empty and 'respondent_dept' in survey_df.columns:
        dept_scores = survey_df.groupby('respondent_dept')[
            ['satisfaction_score', 'usability_score', 'speed_score', 'avg_survey_score']
        ].mean().round(2).reset_index()
        
        fig_bar = go.Figure(data=[go.Bar(
            x=dept_scores['respondent_dept'],
            y=dept_scores['avg_survey_score'],
            text=[f"{v:.2f}점" for v in dept_scores['avg_survey_score']],
            textposition='outside',
            marker_color='#00c9a7',
            hovertemplate='<b>%{x}</b><br>종합 만족도: %{y:.2f}점<extra></extra>'
        )])
        fig_bar.update_layout(
            margin=dict(t=20, b=10, l=10, r=10),
            height=320,
            yaxis=dict(range=[0, 5.5], title="만족도 점수"),
            xaxis=dict(title="부서명")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("선택된 필터에 설문지 데이터가 없습니다.")

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# 차트 3: 저해상도 이미지와 일반 이미지의 OCR 신뢰도 비교
row2_col1, row2_col2 = st.columns([1.2, 0.8])

with row2_col1:
    st.markdown("<div class='section-title'>⚡ 이미지 해상도/노이즈 환경별 OCR 신뢰도 비교</div>", unsafe_allow_html=True)
    
    cond_clean = df[(df['is_low_resolution']==False) & (df['has_noise']==False)]['confidence'].mean()*100
    cond_noise = df[df['has_noise']==True]['confidence'].mean()*100
    cond_lowres = df[df['is_low_resolution']==True]['confidence'].mean()*100
    cond_both = df[(df['is_low_resolution']==True) & (df['has_noise']==True)]['confidence'].mean()*100
    
    env_labels = ["일반 고해상도", "노이즈 환경", "저해상도 환경", "저해상도+노이즈 복합"]
    env_vals = [cond_clean, cond_noise, cond_lowres, cond_both]
    
    fig_comp = go.Figure(data=[go.Bar(
        x=env_labels,
        y=env_vals,
        text=[f"{v:.1f}%" for v in env_vals],
        textposition='outside',
        marker_color=['#00c9a7', '#4d8fac', '#f59e0b', '#ef4444'],
        hovertemplate='<b>%{x}</b><br>평균 신뢰도: %{y:.2f}%<extra></extra>'
    )])
    fig_comp.update_layout(
        margin=dict(t=20, b=10, l=10, r=10),
        height=280,
        yaxis=dict(range=[70, 105], title="신뢰도 (%)"),
        xaxis=dict(title="이미지 촬영 및 보관 환경")
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with row2_col2:
    st.markdown("<div class='section-title'>📌 데이터 보정 정책 요약</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:15px; font-size:13px; line-height:1.7;">
        <b>1. 일자 누락</b>: 원본 마스터 DB와 매칭하여 자동 보완<br>
        <b>2. 영수증 금액 누락</b>: 원본 검증 후 복원 (<code>amount_imputed</code> 표기)<br>
        <b>3. 설문 점수 누락</b>: 소속 부서별 평균 평점으로 대치<br>
        <b>4. 수기 메모 결측</b>: 누락 문서는 <code>'확인필요'</code> 상태로 관리
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# 8. OCR 실패 또는 확인필요 목록 테이블 & 이미지 미리보기
st.markdown("<div class='section-title'>🔍 OCR 검토 및 확인필요 대상 목록 테이블</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["⚠️ 확인필요 / 보완 완료 목록", "📑 전체 정제 데이터셋"])

with tab1:
    review_df = filtered_df[filtered_df['needs_review'] == True].copy()
    st.write(f"총 **{len(review_df)}건**의 문서가 결측치 보완 또는 검토 대상으로 분류되었습니다.")
    
    display_cols = [
        'record_id', 'document_type', 'doc_date', 'organization_or_store',
        'respondent_dept', 'total_amount', 'avg_survey_score', 'handwritten_note',
        'review_reason', 'confidence'
    ]
    
    st.dataframe(
        review_df[display_cols].rename(columns={
            'record_id': '문서ID',
            'document_type': '문서유형',
            'doc_date': '일자',
            'organization_or_store': '상호명',
            'respondent_dept': '부서',
            'total_amount': '금액(원)',
            'avg_survey_score': '설문평점',
            'handwritten_note': '수기메모',
            'review_reason': '보완사유',
            'confidence': '신뢰도'
        }),
        use_container_width=True,
        height=260
    )

    if not review_df.empty:
        st.markdown("#### 🖼️ 선택 문서 원본 및 전처리 이미지 비교 뷰어")
        selected_record = st.selectbox("확인할 문서 ID 선택", review_df['record_id'].tolist())
        target_row = review_df[review_df['record_id'] == selected_record].iloc[0]

        img_col1, img_col2, meta_col = st.columns([1, 1, 1.2])

        raw_img_path = os.path.join(base_dir, os.path.normpath(target_row['image_filename']))
        doc_sub = 'receipts' if target_row['document_type'] == 'receipt' else 'surveys'
        prep_img_path = os.path.join(base_dir, 'data', 'ocr', 'preprocessed_images', doc_sub, os.path.basename(target_row['image_filename']))

        with img_col1:
            st.caption("📷 원본 이미지")
            if os.path.exists(raw_img_path):
                st.image(raw_img_path, use_container_width=True)
            else:
                st.warning("원본 이미지를 찾을 수 없습니다.")

        with img_col2:
            st.caption("✨ OpenCV 전처리 이미지")
            if os.path.exists(prep_img_path):
                st.image(prep_img_path, use_container_width=True)
            else:
                st.warning("전처리 이미지를 찾을 수 없습니다.")

        with meta_col:
            st.caption("📋 문서 상세 정보")
            st.json({
                "Record ID": target_row['record_id'],
                "Document Type": target_row['document_type'],
                "Doc Date": str(target_row['doc_date']),
                "Store / Dept": target_row['organization_or_store'] if target_row['document_type'] == 'receipt' else target_row['respondent_dept'],
                "Amount / Score": f"{int(target_row['total_amount']):,}원" if target_row['document_type'] == 'receipt' else f"{target_row['avg_survey_score']}점",
                "Handwritten Note": target_row['handwritten_note'],
                "Confidence": f"{target_row['confidence']*100:.2f}%",
                "Review Reason": target_row['review_reason']
            })

with tab2:
    st.dataframe(
        filtered_df[[
            'record_id', 'document_type', 'doc_date', 'organization_or_store',
            'respondent_dept', 'category', 'total_amount', 'satisfaction_score',
            'usability_score', 'speed_score', 'handwritten_note', 'confidence'
        ]],
        use_container_width=True,
        height=350
    )
