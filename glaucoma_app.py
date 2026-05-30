"""
GlaucomaGuard 👁 — 녹내장 조기진단 지원 시스템
환자용 자가 위험 평가 + 의사/연구자용 임상 의사결정 지원 대시보드

호환: Streamlit >= 1.45  (use_container_width 완전 제거)
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GlaucomaGuard — 녹내장 조기진단",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS 스타일
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
    div[data-testid="metric-container"] {
        background: #f0f6ff;
        border: 1px solid #d0e4ff;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .risk-low   { background:#d4edda; color:#155724; padding:6px 14px;
                  border-radius:20px; font-weight:700; display:inline-block; }
    .risk-mid   { background:#fff3cd; color:#856404; padding:6px 14px;
                  border-radius:20px; font-weight:700; display:inline-block; }
    .risk-high  { background:#f8d7da; color:#721c24; padding:6px 14px;
                  border-radius:20px; font-weight:700; display:inline-block; }
    .risk-vhigh { background:#c0392b; color:#fff; padding:6px 14px;
                  border-radius:20px; font-weight:700; display:inline-block; }
    .info-box    { background:#e8f4fd; border-left:4px solid #2196F3;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .warn-box    { background:#fff8e1; border-left:4px solid #FFC107;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
    .success-box { background:#e8f5e9; border-left:4px solid #4CAF50;
                   padding:12px 16px; border-radius:6px; margin:8px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 상수 — 임상 정상 범위
# ──────────────────────────────────────────────
NORMAL_RANGES = {
    "iop":  (10,    21,    "mmHg", "안압 (IOP)"),
    "cdr":  (0.0,   0.5,   "",     "컵-디스크 비율 (CDR)"),
    "vcdr": (0.0,   0.5,   "",     "수직 CDR (VCDR)"),
    "cct":  (500,   580,   "µm",   "중심각막두께 (CCT)"),
    "md":   (-2.0,  2.0,   "dB",   "시야 평균편차 (MD)"),
    "psd":  (0.0,   2.5,   "dB",   "시야 패턴표준편차 (PSD)"),
    "rnfl": (80,    120,   "µm",   "망막신경섬유층 (RNFL)"),
}

FEATURE_KO = {
    "iop":            "안압 (IOP, mmHg)",
    "cdr":            "컵-디스크 비율 (CDR)",
    "vcdr":           "수직 CDR (VCDR)",
    "cct":            "중심각막두께 (CCT, µm)",
    "md":             "시야 평균편차 (MD, dB)",
    "psd":            "시야 패턴표준편차 (PSD, dB)",
    "rnfl":           "망막신경섬유층 (RNFL, µm)",
    "age":            "나이",
    "sex":            "성별",
    "diabetes":       "당뇨",
    "hypertension":   "고혈압",
    "family_history": "녹내장 가족력",
}

FEATURE_COLS = [
    "age", "sex", "iop", "cdr", "vcdr",
    "cct", "md", "psd", "rnfl",
    "diabetes", "hypertension", "family_history",
]

DATA_PATH = Path("data/glaucoma/glaucoma_data.csv")


# ──────────────────────────────────────────────
# 합성 데이터 생성
# ──────────────────────────────────────────────
def _generate_synthetic(n: int = 800) -> pd.DataFrame:
    """임상 논문 기반 현실적 합성 데이터 (AUC ~0.90 난이도)"""
    rng = np.random.default_rng(42)
    half = n // 2

    def make_group(is_g: bool, size: int) -> dict:
        g = int(is_g)
        age  = rng.normal(53 + g * 8,   14,          size).clip(25, 85).round().astype(int)
        iop  = rng.normal(15.5 + g * 7.5, 3.5 + g * 2.0, size).clip(8, 45).round(1)
        cdr  = rng.normal(0.37 + g * 0.28, 0.09 + g * 0.06, size).clip(0.1, 0.95).round(3)
        vcdr = (cdr + rng.normal(0, 0.04, size)).clip(0.1, 0.95).round(3)
        cct  = rng.normal(545 - g * 18,  34,          size).clip(440, 640).round().astype(int)
        md   = rng.normal(-1.4 - g * 6.0, 2.0 + g * 4.5, size).clip(-30, 2).round(2)
        psd  = rng.normal(1.9 + g * 3.5,  1.0 + g * 2.0, size).clip(0.5, 15).round(2)
        rnfl = rng.normal(94 - g * 23,    12 + g * 6,  size).clip(40, 130).round().astype(int)
        sex  = rng.integers(0, 2, size)
        dm   = rng.choice([0, 1], size, p=[0.84 - g * 0.12, 0.16 + g * 0.12])
        htn  = rng.choice([0, 1], size, p=[0.74 - g * 0.16, 0.26 + g * 0.16])
        fam  = rng.choice([0, 1], size, p=[0.81 - g * 0.19, 0.19 + g * 0.19])
        return dict(
            age=age, sex=sex, iop=iop, cdr=cdr, vcdr=vcdr,
            cct=cct, md=md, psd=psd, rnfl=rnfl,
            diabetes=dm, hypertension=htn,
            family_history=fam,
            target=np.full(size, g, dtype=int),
        )

    df = pd.concat(
        [pd.DataFrame(make_group(False, half)), pd.DataFrame(make_group(True, n - half))],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        source = "저장된 데이터"
    else:
        df = _generate_synthetic()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        source = "임상 통계 기반 합성 데이터"
    return df, source


# ──────────────────────────────────────────────
# ML 모델 학습
# ──────────────────────────────────────────────
@st.cache_resource
def train_models(_df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in _df.columns]
    X = _df[available].fillna(_df[available].median())
    y = _df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_dict = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
    }

    results = {}
    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        cv = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "auc": roc_auc_score(y_test, y_prob),
            "cv_auc": cv,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "cm": confusion_matrix(y_test, y_pred),
        }

    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]

    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        raw = best_model.named_steps["clf"].coef_[0]
        fi = np.abs(raw) / (np.abs(raw).sum() + 1e-9)
    else:
        fi = np.ones(len(available)) / len(available)

    fi_df = pd.DataFrame({"feature": available, "importance": fi}).sort_values(
        "importance", ascending=False
    )
    return results, best_name, best_model, fi_df, available, X_test, y_test


# ──────────────────────────────────────────────
# 유틸 함수
# ──────────────────────────────────────────────
def risk_badge(prob: float) -> str:
    if prob < 0.25:
        return f'<span class="risk-low">낮음 ({prob:.0%})</span>'
    if prob < 0.50:
        return f'<span class="risk-mid">보통 ({prob:.0%})</span>'
    if prob < 0.75:
        return f'<span class="risk-high">높음 ({prob:.0%})</span>'
    return f'<span class="risk-vhigh">매우 높음 ({prob:.0%})</span>'


def gauge_chart(prob: float, title: str = "녹내장 위험도") -> go.Figure:
    color = (
        "#2ecc71" if prob < 0.25 else
        "#f39c12" if prob < 0.50 else
        "#e74c3c" if prob < 0.75 else
        "#c0392b"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": title, "font": {"size": 17}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 25],   "color": "#d5f5e3"},
                {"range": [25, 50],  "color": "#fef9e7"},
                {"range": [50, 75],  "color": "#fde8e8"},
                {"range": [75, 100], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20))
    return fig


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame, source: str) -> str:
    with st.sidebar:
        st.markdown("## 👁 GlaucomaGuard")
        st.caption("녹내장 조기진단 지원 시스템")
        st.divider()

        mode = st.radio(
            "사용자 모드 선택",
            ["👤 환자용  (자가 위험 평가)", "🩺 의사 / 연구자용"],
        )
        st.divider()

        n_g = int(df["target"].sum())
        n_t = len(df)
        st.metric("총 데이터", f"{n_t:,}건")
        c1, c2 = st.columns(2)
        c1.metric("녹내장", f"{n_g}")
        c2.metric("정상",   f"{n_t - n_g}")

        st.divider()
        st.caption(f"📂 출처: {source}")
        st.caption("⚠️ 본 앱은 참고용이며 의학 진단을 대체하지 않습니다.")
    return mode


# ══════════════════════════════════════════════
# 환자용 뷰
# ══════════════════════════════════════════════
def patient_view(df: pd.DataFrame, best_model, available: list):
    st.header("👤 나의 녹내장 위험도 자가 평가")
    st.markdown(
        '<div class="warn-box">⚠️ 이 평가는 <b>참고용</b>입니다. '
        "정확한 진단은 반드시 안과 전문의에게 받으세요.</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📋 위험도 평가", "📖 녹내장 정보"])

    # ── 위험도 평가 폼 ──
    with tab1:
        with st.form("patient_form"):
            st.subheader("기본 정보")
            c1, c2, _ = st.columns(3)
            age = c1.number_input("나이", 20, 90, 50, 1)
            sex = c2.selectbox("성별", ["여성", "남성"])
            sex_val = 1 if sex == "남성" else 0

            st.subheader("건강 이력")
            c1, c2, c3 = st.columns(3)
            family_history = c1.checkbox("녹내장 가족력")
            diabetes       = c2.checkbox("당뇨병")
            hypertension   = c3.checkbox("고혈압")

            st.subheader("눈 관련 증상 / 검진")
            c1, c2 = st.columns(2)
            high_iop    = c1.checkbox("이전 안과 검진에서 안압이 높다는 말을 들은 적 있음")
            vision_loss = c2.checkbox("최근 시야가 좁아지거나 흐릿해진 느낌이 있음")

            st.subheader("생활습관")
            c1, c2, c3 = st.columns(3)
            c1.checkbox("흡연 중 / 과거 흡연력")
            glasses = c2.checkbox("고도근시 (-6D 이상)")
            c3.checkbox("스테로이드 약물 장기 사용")

            submitted = st.form_submit_button("위험도 분석하기", width="stretch")

        if submitted:
            iop_est  = 22.0 if high_iop    else 15.5
            cdr_est  = 0.55 if (family_history or vision_loss) else 0.38
            vcdr_est = cdr_est - 0.02
            cct_est  = 520.0 if glasses     else 545.0
            md_est   = -4.0  if vision_loss else -1.0
            psd_est  = 3.5   if vision_loss else 1.8
            rnfl_est = 72.0  if vision_loss else 94.0

            row = dict(
                age=age, sex=sex_val,
                iop=iop_est, cdr=cdr_est, vcdr=vcdr_est,
                cct=cct_est, md=md_est, psd=psd_est, rnfl=rnfl_est,
                diabetes=int(diabetes), hypertension=int(hypertension),
                family_history=int(family_history),
            )
            inp = pd.DataFrame([{c: row.get(c, 0) for c in available}])
            prob = best_model.predict_proba(inp)[0, 1]

            st.divider()
            col_g, col_r = st.columns(2)
            with col_g:
                st.plotly_chart(gauge_chart(prob))
            with col_r:
                st.subheader("평가 결과")
                st.markdown(f"**종합 위험도:** {risk_badge(prob)}", unsafe_allow_html=True)

                factors = []
                if age >= 60:         factors.append("고령 (60세 이상)")
                if family_history:    factors.append("녹내장 가족력")
                if high_iop:          factors.append("안압 높음 이력")
                if vision_loss:       factors.append("시야 이상 증상")
                if diabetes:          factors.append("당뇨병")
                if hypertension:      factors.append("고혈압")
                if glasses:           factors.append("고도근시")

                if factors:
                    st.markdown("**해당 위험 요인:**")
                    for f in factors:
                        st.markdown(f"  • {f}")
                else:
                    st.markdown("**위험 요인:** 특별한 위험 요인 없음")

            st.divider()
            if prob < 0.25:
                st.markdown(
                    '<div class="success-box">✅ <b>낮은 위험군</b> — 2년에 1회 안과 정기 검진을 유지하세요.</div>',
                    unsafe_allow_html=True,
                )
            elif prob < 0.50:
                st.markdown(
                    '<div class="info-box">ℹ️ <b>보통 위험군</b> — 1년 내 안과 검진을 권장합니다. '
                    "안압·시신경 검사를 받아보세요.</div>",
                    unsafe_allow_html=True,
                )
            elif prob < 0.75:
                st.markdown(
                    '<div class="warn-box">⚠️ <b>높은 위험군</b> — 빠른 시일 내 안과 전문의 진료를 강력히 권합니다. '
                    "안압·시야·OCT 검사를 받으세요.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.error(
                    "🚨 **매우 높은 위험군** — 즉시 안과 전문의 진료를 받으세요. "
                    "조기 발견 시 실명을 예방할 수 있습니다."
                )

    # ── 녹내장 정보 ──
    with tab2:
        st.subheader("👁 녹내장이란?")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
**녹내장(Glaucoma)**은 시신경이 손상되어 시야가 점차 좁아지는 안질환입니다.
초기에는 자각 증상이 거의 없어 **조기 발견이 매우 중요**합니다.

한국 40세 이상 성인의 약 **3.5%**가 녹내장을 가지고 있으며,
실명 원인 2위를 차지합니다.
"""
            )
        with c2:
            st.markdown(
                """
**주요 증상**
- 주변 시야가 점차 좁아짐
- 안개 낀 것처럼 흐릿하게 보임
- 밤에 빛 주위 무지개 빛
- 눈의 통증·두통 (급성의 경우)

**⚠ 증상은 말기까지 없을 수 있습니다!**
"""
            )

        st.divider()
        st.subheader("📊 주요 위험 요인")
        risk_df = pd.DataFrame({
            "위험 요인": [
                "안압 상승 (>21mmHg)", "녹내장 가족력", "60세 이상 고령",
                "스테로이드 장기 사용", "얇은 각막 두께", "고도근시",
                "당뇨병", "고혈압",
            ],
            "상대 위험도": [6.0, 3.7, 2.5, 4.0, 2.1, 1.9, 1.8, 1.7],
        })
        fig_risk = px.bar(
            risk_df.sort_values("상대 위험도"),
            x="상대 위험도", y="위험 요인", orientation="h",
            color="상대 위험도", color_continuous_scale="Reds",
            text="상대 위험도",
            title="녹내장 주요 위험 요인별 상대 위험도",
        )
        fig_risk.update_traces(texttemplate="%{text:.1f}×", textposition="outside")
        fig_risk.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_risk)

        st.divider()
        st.subheader("🛡 예방 및 관리")
        c1, c2, c3 = st.columns(3)
        c1.info("**정기 검진**\n\n40세 이상 2년 1회, 위험군은 매년 안과 검진 권장")
        c2.info("**생활 습관**\n\n과도한 수분 자제, 적절한 운동, 금연, 스트레스 관리")
        c3.info("**조기 치료**\n\n발견 시 안약·레이저·수술로 진행 억제 가능")


# ══════════════════════════════════════════════
# 의사/연구자용 뷰
# ══════════════════════════════════════════════
def doctor_view(
    df: pd.DataFrame,
    results: dict,
    best_name: str,
    best_model,
    fi_df: pd.DataFrame,
    available: list,
    X_test,
    y_test,
):
    st.header("🩺 임상 의사결정 지원 시스템")

    tab1, tab2, tab3 = st.tabs(["🔬 환자 진단 지원", "📊 데이터 탐색", "🤖 모델 성능"])

    # ══ Tab 1: 환자 진단 지원 ══
    with tab1:
        st.subheader("임상 수치 입력 및 녹내장 위험 평가")
        st.markdown(
            '<div class="info-box">임상 측정값을 입력하면 ML 모델이 녹내장 위험도를 분석하고 '
            "각 인자의 기여도를 시각화합니다.</div>",
            unsafe_allow_html=True,
        )

        with st.form("doctor_form"):
            st.markdown("**기본 정보**")
            c1, c2 = st.columns(2)
            d_age = c1.number_input("나이 (세)", 20, 90, 60, 1)
            d_sex_str = c2.selectbox("성별", ["여성", "남성"])
            d_sex = 0 if d_sex_str == "여성" else 1

            st.markdown("**안압 및 구조 지표**")
            c1, c2, c3 = st.columns(3)
            d_iop  = c1.number_input("안압 IOP (mmHg)",  5.0,  60.0, 16.0, 0.5,
                                     help="정상: 10–21 mmHg")
            d_cdr  = c2.number_input("CDR (Cup:Disc)",   0.0,   1.0,  0.4, 0.01,
                                     help="정상: 0.0–0.5")
            d_vcdr = c3.number_input("수직 CDR (VCDR)",  0.0,   1.0,  0.4, 0.01,
                                     help="정상: 0.0–0.5")

            st.markdown("**각막 및 시야 지표**")
            c1, c2, c3, c4 = st.columns(4)
            d_cct  = c1.number_input("CCT (µm)",    440, 640, 545, 1,
                                     help="정상: 500–580 µm")
            d_md   = c2.number_input("MD (dB)",   -30.0, 2.0, -1.5, 0.1,
                                     help="정상: -2.0–2.0 dB")
            d_psd  = c3.number_input("PSD (dB)",    0.5, 15.0,  1.9, 0.1,
                                     help="정상: 0.5–2.5 dB")
            d_rnfl = c4.number_input("RNFL (µm)",  40, 130,  95, 1,
                                     help="정상: 80–120 µm")

            st.markdown("**전신 질환 / 가족력**")
            c1, c2, c3 = st.columns(3)
            d_dm  = c1.checkbox("당뇨병")
            d_htn = c2.checkbox("고혈압")
            d_fam = c3.checkbox("녹내장 가족력")

            submitted_doc = st.form_submit_button("분석 실행", width="stretch")

        if submitted_doc:
            row = dict(
                age=d_age, sex=d_sex,
                iop=d_iop, cdr=d_cdr, vcdr=d_vcdr,
                cct=d_cct, md=d_md, psd=d_psd, rnfl=d_rnfl,
                diabetes=int(d_dm), hypertension=int(d_htn),
                family_history=int(d_fam),
            )
            inp = pd.DataFrame([{c: row.get(c, 0) for c in available}])
            prob = best_model.predict_proba(inp)[0, 1]

            st.divider()
            col_gauge, col_detail = st.columns([1, 1.4])
            with col_gauge:
                st.plotly_chart(gauge_chart(prob, "녹내장 위험 확률"))
                st.markdown(
                    f"**판정:** {risk_badge(prob)} &nbsp;|&nbsp; 모델: `{best_name}`",
                    unsafe_allow_html=True,
                )

            with col_detail:
                st.subheader("임상 수치 정상 범위 체크")
                check_items = [
                    ("iop", d_iop), ("cdr", d_cdr), ("vcdr", d_vcdr),
                    ("cct", d_cct), ("md", d_md), ("psd", d_psd), ("rnfl", d_rnfl),
                ]
                rows = []
                for col_key, val in check_items:
                    lo, hi, unit, label = NORMAL_RANGES[col_key]
                    if lo <= val <= hi:
                        status = "✅ 정상"
                    elif val < lo:
                        status = "⬇ 낮음"
                    else:
                        status = "⬆ 높음"
                    rows.append({
                        "지표": label,
                        "측정값": f"{val} {unit}".strip(),
                        "정상 범위": f"{lo}–{hi} {unit}".strip(),
                        "상태": status,
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True)

            # 변수 기여도
            st.subheader("🔍 예측에 영향을 준 주요 요인")
            patient_vals = np.array([row.get(c, 0) for c in available], dtype=float)
            pop_median   = df[[c for c in available]].median().values

            if hasattr(best_model, "feature_importances_"):
                fi_vals = best_model.feature_importances_
            else:
                fi_vals = np.ones(len(available)) / len(available)

            contrib = (patient_vals - pop_median) * fi_vals
            pos_set = {"iop", "cdr", "vcdr", "psd", "diabetes", "hypertension", "family_history", "age"}
            neg_set = {"cct", "md", "rnfl"}
            dirs = np.array([
                1 if c in pos_set else (-1 if c in neg_set else 1)
                for c in available
            ])
            contrib_signed = contrib * dirs

            cdf = pd.DataFrame({
                "feature": [FEATURE_KO.get(c, c) for c in available],
                "contribution": contrib_signed,
            }).sort_values("contribution", ascending=True)

            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in cdf["contribution"]]
            fig_c = go.Figure(go.Bar(
                x=cdf["contribution"], y=cdf["feature"],
                orientation="h", marker_color=colors,
                text=[f"{v:+.3f}" for v in cdf["contribution"]],
                textposition="outside",
            ))
            fig_c.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_c.update_layout(
                title="🔴 빨간색 = 위험도 증가 요인  |  🟢 초록색 = 위험도 감소 요인",
                height=380,
                xaxis_title="기여도 (상대적)",
                margin=dict(l=10, r=70, t=50, b=10),
            )
            st.plotly_chart(fig_c)

            # 전체 분포 대비 환자 위치
            st.subheader("📈 환자 수치의 전체 분포 내 위치")
            key_cols = ["iop", "cdr", "md", "rnfl"]
            key_vals = [d_iop, d_cdr, d_md, d_rnfl]
            fig_dist = make_subplots(
                rows=1, cols=4,
                subplot_titles=[NORMAL_RANGES[c][3] for c in key_cols],
            )
            for i, (col_key, val) in enumerate(zip(key_cols, key_vals)):
                for label, color in [("정상", "#3498db"), ("녹내장", "#e74c3c")]:
                    subset = df[df["target"] == (0 if label == "정상" else 1)][col_key].dropna()
                    fig_dist.add_trace(
                        go.Histogram(x=subset, name=label, marker_color=color,
                                     opacity=0.6, showlegend=(i == 0)),
                        row=1, col=i + 1,
                    )
                fig_dist.add_vline(x=val, line_color="black", line_width=2,
                                   line_dash="dash", row=1, col=i + 1)
            fig_dist.update_layout(
                height=300, barmode="overlay",
                title="검은 점선 = 현재 환자",
                margin=dict(t=60, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_dist)

    # ══ Tab 2: 데이터 탐색 ══
    with tab2:
        st.subheader("데이터셋 개요")
        n_t = len(df)
        n_g = int(df["target"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전체 환자", f"{n_t}")
        m2.metric("녹내장", f"{n_g}", f"{n_g/n_t:.1%}")
        m3.metric("정상",   f"{n_t - n_g}", f"{(n_t - n_g)/n_t:.1%}")
        m4.metric("사용 변수", f"{len(available)}개")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                values=[n_g, n_t - n_g],
                names=["녹내장", "정상"],
                color_discrete_sequence=["#e74c3c", "#3498db"],
                title="진단 분포",
            )
            st.plotly_chart(fig_pie)
        with c2:
            if "age" in df.columns:
                fig_age = px.histogram(
                    df, x="age",
                    color=df["target"].map({0: "정상", 1: "녹내장"}),
                    color_discrete_map={"정상": "#3498db", "녹내장": "#e74c3c"},
                    barmode="overlay", opacity=0.7,
                    title="나이별 분포",
                    labels={"age": "나이", "color": "진단"},
                )
                st.plotly_chart(fig_age)

        st.subheader("주요 임상 지표 비교 (정상 vs 녹내장)")
        cmp_cols = [c for c in ["iop", "cdr", "cct", "md", "psd", "rnfl"] if c in df.columns]
        if cmp_cols:
            fig_box = make_subplots(
                rows=2, cols=3,
                subplot_titles=[NORMAL_RANGES[c][3] for c in cmp_cols],
            )
            for idx, col_key in enumerate(cmp_cols):
                r, ci = divmod(idx, 3)
                for label, color in [("정상", "#3498db"), ("녹내장", "#e74c3c")]:
                    subset = df[df["target"] == (0 if label == "정상" else 1)][col_key].dropna()
                    fig_box.add_trace(
                        go.Box(y=subset, name=label, marker_color=color,
                               showlegend=(idx == 0)),
                        row=r + 1, col=ci + 1,
                    )
            fig_box.update_layout(height=500, title="그룹별 임상 지표 분포", boxmode="group")
            st.plotly_chart(fig_box)

        st.subheader("상관관계 히트맵")
        corr = df[available + ["target"]].corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            title="변수 간 상관관계 (target = 녹내장 여부)",
            zmin=-1, zmax=1,
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat)

        st.subheader("원시 데이터 미리보기")
        show_tgt = st.toggle("진단 결과 표시", value=True)
        disp = df.copy()
        disp["진단"] = disp["target"].map({0: "정상", 1: "녹내장"})
        if not show_tgt:
            disp = disp.drop(columns=["target", "진단"], errors="ignore")
        st.dataframe(disp.head(50), hide_index=True)

    # ══ Tab 3: 모델 성능 ══
    with tab3:
        st.subheader("머신러닝 모델 성능 비교")

        rows_summary = []
        for name, res in results.items():
            rpt = res["report"]
            rows_summary.append({
                "모델": name,
                "AUC-ROC": f"{res['auc']:.4f}",
                "CV AUC (5-fold)": f"{res['cv_auc'].mean():.4f} ± {res['cv_auc'].std():.4f}",
                "정확도": f"{rpt['accuracy']:.4f}",
                "민감도 (Recall)": f"{rpt['1']['recall']:.4f}",
                "특이도": f"{rpt['0']['recall']:.4f}",
                "F1-Score": f"{rpt['1']['f1-score']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows_summary), hide_index=True)
        st.markdown(f"**최고 성능 모델: `{best_name}`**")

        colors_roc = ["#e74c3c", "#3498db", "#2ecc71"]

        c1, c2 = st.columns(2)
        with c1:
            fig_roc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dash", color="gray"))
            for (name, res), clr in zip(results.items(), colors_roc):
                fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{name} (AUC={res['auc']:.3f})",
                    line=dict(color=clr, width=2),
                ))
            fig_roc.update_layout(
                title="ROC 커브",
                xaxis_title="위양성률 (1 - Specificity)",
                yaxis_title="민감도 (Sensitivity)",
                height=400, legend=dict(x=0.4, y=0.1),
            )
            st.plotly_chart(fig_roc)

        with c2:
            cm = results[best_name]["cm"]
            fig_cm = px.imshow(
                cm,
                x=["정상 (예측)", "녹내장 (예측)"],
                y=["정상 (실제)", "녹내장 (실제)"],
                color_continuous_scale="Blues", text_auto=True,
                title=f"Confusion Matrix — {best_name}",
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm)

        c3, c4 = st.columns(2)
        with c3:
            fig_pr = go.Figure()
            for (name, res), clr in zip(results.items(), colors_roc):
                prec, rec, _ = precision_recall_curve(res["y_test"], res["y_prob"])
                ap = average_precision_score(res["y_test"], res["y_prob"])
                fig_pr.add_trace(go.Scatter(
                    x=rec, y=prec,
                    name=f"{name} (AP={ap:.3f})",
                    line=dict(color=clr, width=2),
                ))
            fig_pr.update_layout(
                title="Precision-Recall 커브",
                xaxis_title="Recall", yaxis_title="Precision",
                height=380,
            )
            st.plotly_chart(fig_pr)

        with c4:
            fi_disp = fi_df.copy()
            fi_disp["변수"] = fi_disp["feature"].map(lambda x: FEATURE_KO.get(x, x))
            fig_fi = px.bar(
                fi_disp.sort_values("importance"),
                x="importance", y="변수", orientation="h",
                color="importance", color_continuous_scale="Blues",
                title=f"변수 중요도 — {best_name}",
                labels={"importance": "중요도"},
            )
            fig_fi.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_fi)

        st.subheader("교차 검증 결과 (5-Fold)")
        cv_rows = [
            {"모델": name, "Fold": i + 1, "AUC": s}
            for name, res in results.items()
            for i, s in enumerate(res["cv_auc"])
        ]
        fig_cv = px.box(
            pd.DataFrame(cv_rows), x="모델", y="AUC",
            color="모델",
            color_discrete_sequence=colors_roc,
            title="5-Fold 교차 검증 AUC 분포",
            points="all",
        )
        fig_cv.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_cv)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    df, source = load_data()
    results, best_name, best_model, fi_df, available, X_test, y_test = train_models(df)
    mode = render_sidebar(df, source)

    if "환자" in mode:
        patient_view(df, best_model, available)
    else:
        doctor_view(df, results, best_name, best_model, fi_df, available, X_test, y_test)


if __name__ == "__main__":
    main()
