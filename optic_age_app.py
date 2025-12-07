import os
from pathlib import Path
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_BY_AGE_PATH = BASE_DIR / "optic_age_models_by_age.pkl"
FEATURES_BY_AGE_PATH = BASE_DIR / "optic_age_features_by_age.pkl"
RESULTS_PATH = BASE_DIR / "optic_age_results.csv"
SUPERNORMAL_PATH = BASE_DIR / "supernormal_fulloct_40_79.csv"


# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_features():
    if not MODELS_BY_AGE_PATH.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {MODELS_BY_AGE_PATH}")
        st.stop()
    if not FEATURES_BY_AGE_PATH.exists():
        st.error(f"feature 리스트 파일을 찾을 수 없습니다: {FEATURES_BY_AGE_PATH}")
        st.stop()

    models_by_age = joblib.load(MODELS_BY_AGE_PATH)
    features_by_age = joblib.load(FEATURES_BY_AGE_PATH)
    return models_by_age, features_by_age


@st.cache_data(show_spinner=False)
def load_results():
    if not RESULTS_PATH.exists():
        return None
    return pd.read_csv(RESULTS_PATH)


@st.cache_data(show_spinner=False)
def load_supernormal():
    if not SUPERNORMAL_PATH.exists():
        return None
    return pd.read_csv(SUPERNORMAL_PATH)


@st.cache_data(show_spinner=False)
def compute_age_adjust_params() -> Dict[str, Dict[str, float]]:
    """
    Super-normal 데이터에서 age–RNFL / age–GCIPL 선형식을 다시 추정.
    y = a*age + b 형태의 계수를 반환.
    """
    df = load_supernormal()
    if df is None:
        # app은 계속 동작하게 하되, residual 계산은 나중에 스킵
        return {}

    # 필수 컬럼 체크
    rnfl_cols = [f"CLOCKHOUR_{h}_OD" for h in range(1, 13)]
    gc_cols = [
        "GC_TEMPSUP_OD",
        "GC_SUP_OD",
        "GC_NASSUP_OD",
        "GC_NASINF_OD",
        "GC_INF_OD",
        "GC_TEMPINF_OD",
    ]

    missing_rnfl = [c for c in rnfl_cols if c not in df.columns]
    missing_gc = [c for c in gc_cols if c not in df.columns]

    if missing_rnfl or missing_gc or "age" not in df.columns:
        # 구조가 예상과 다른 경우에는 residual을 사용하지 않음
        return {}

    df["rnfl12_mean"] = df[rnfl_cols].mean(axis=1)
    df["gc6_mean"] = df[gc_cols].mean(axis=1)

    x = df["age"].to_numpy()
    X = np.vstack([x, np.ones_like(x)]).T

    a_rnfl, b_rnfl = np.linalg.lstsq(X, df["rnfl12_mean"].to_numpy(), rcond=None)[0]
    a_gc, b_gc = np.linalg.lstsq(X, df["gc6_mean"].to_numpy(), rcond=None)[0]

    return {
        "rnfl12": {"slope": float(a_rnfl), "intercept": float(b_rnfl)},
        "gc6": {"slope": float(a_gc), "intercept": float(b_gc)},
    }


@st.cache_data(show_spinner=False)
def compute_onas_sigma() -> float:
    """
    optic_age_results.csv 에서 ΔAge의 표준편차를 계산해 ONAS sigma로 사용.
    """
    df = load_results()
    if df is None or "delta_age" not in df.columns:
        # 기본값 (train 로그에서 확인된 값이 있으면 반영)
        return 2.2
    return float(df["delta_age"].std(ddof=0))


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def get_age_bin(age: float) -> str:
    if age < 40 or age >= 80:
        raise ValueError("모델은 40–79세에서만 유효합니다.")
    if age < 50:
        return "40s"
    if age < 60:
        return "50s"
    if age < 70:
        return "60s"
    return "70s"


def normal_cdf(z: float) -> float:
    """표준정규분포 CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def build_patient_dict(
    age: float,
    sex: str,
    al: float,
    rnfl_inputs: Dict[str, float],
    gc_inputs: Dict[str, float],
    age_adj: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    사용자 입력 + 파생변수(rnfl12_mean, gc6_mean, residual)를 모두 포함하는 dict 생성.
    feature 이름은 train_model_full_oct_tuned.py에서 사용한 이름과 일치해야 함.
    """
    patient = {}

    # 기본 인구학적 / 생체계측
    patient["age"] = float(age)
    # sex: 0 = female, 1 = male 로 인코딩
    patient["sex"] = 1.0 if sex == "Male" else 0.0
    patient["AL_R"] = float(al)

    # RNFL clock-hour 값
    for k, v in rnfl_inputs.items():
        patient[k] = float(v)

    # GCIPL 6-sector 값
    for k, v in gc_inputs.items():
        patient[k] = float(v)

    # rnfl12_mean / gc6_mean
    rnfl_vals = list(rnfl_inputs.values())
    gc_vals = list(gc_inputs.values())
    rnfl_mean = float(np.mean(rnfl_vals)) if rnfl_vals else np.nan
    gc_mean = float(np.mean(gc_vals)) if gc_vals else np.nan

    patient["rnfl12_mean"] = rnfl_mean
    patient["gc6_mean"] = gc_mean

    # residual (age-adjusted)
    if age_adj:
        # RNFL residual
        if "rnfl12" in age_adj and not math.isnan(rnfl_mean):
            a = age_adj["rnfl12"]["slope"]
            b = age_adj["rnfl12"]["intercept"]
            expected = a * age + b
            patient["rnfl12_resid"] = rnfl_mean - expected

        # GCIPL residual
        if "gc6" in age_adj and not math.isnan(gc_mean):
            a = age_adj["gc6"]["slope"]
            b = age_adj["gc6"]["intercept"]
            expected = a * age + b
            patient["gc6_resid"] = gc_mean - expected

    return patient


def run_model(
    patient: Dict[str, float],
) -> Tuple[float, float, float, str, np.ndarray, np.ndarray]:
    """
    모델 추론 + ONAS 계산 + SHAP 값을 반환.
    반환:
      - predicted_age
      - delta_age
      - onas_percentile
      - age_bin
      - shap_values (1D ndarray)
      - feature_list (np.ndarray of str)
    """
    models_by_age, features_by_age = load_models_and_features()
    age_adj = compute_age_adjust_params()
    sigma = compute_onas_sigma()

    age = patient["age"]
    age_bin = get_age_bin(age)

    if age_bin not in models_by_age:
        raise ValueError(f"{age_bin} 모델을 찾을 수 없습니다.")
    model = models_by_age[age_bin]
    feat_list = np.array(features_by_age[age_bin])

    # feature vector 구성 (없는 feature는 0.0으로 채움)
    x_vec = np.array([[float(patient.get(f, 0.0)) for f in feat_list]], dtype=float)

    # 예측
    y_pred = float(model.predict(x_vec)[0])
    delta_age = y_pred - age

    # ONAS (higher = more resilient / younger optic nerve)
    z = -delta_age / sigma
    onas = normal_cdf(z) * 100.0
    onas = max(0.0, min(100.0, onas))

    # SHAP 값 (tree 모델 가정)
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(x_vec)[0]
        shap_vals = np.asarray(shap_vals, dtype=float)
    except Exception:
        shap_vals = np.zeros_like(x_vec[0], dtype=float)

    return y_pred, delta_age, onas, age_bin, shap_vals, feat_list


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------
def make_regional_vulnerability_plots(
    shap_vals: np.ndarray, feat_list: np.ndarray
):
    """
    GCIPL 6-sector, RNFL 12H feature에 해당하는 SHAP 값을 polar map으로 시각화.
    6H와 Inferior가 아래쪽에 오도록 좌표계를 설정.
    """
    # GCIPL 6-sector
    gc_feat_order = [
        "GC_TEMPSUP_OD",
        "GC_SUP_OD",
        "GC_NASSUP_OD",
        "GC_NASINF_OD",
        "GC_INF_OD",
        "GC_TEMPINF_OD",
    ]
    # 각 sector의 polar angle (deg), 0deg = 위쪽, 시계방향 증가
    gc_angles_deg = {
        "GC_TEMPSUP_OD": 300.0,  # left–superior
        "GC_SUP_OD": 0.0,  # superior
        "GC_NASSUP_OD": 60.0,  # right–superior (nasal)
        "GC_NASINF_OD": 120.0,  # right–inferior
        "GC_INF_OD": 180.0,  # inferior (bottom)
        "GC_TEMPINF_OD": 240.0,  # left–inferior
    }

    # RNFL clock-hours 1–12
    rnfl_feat_order = [f"CLOCKHOUR_{h}_OD" for h in range(1, 13)]
    # 12H = 위, 시계방향으로 1H, 2H ..., 6H = 아래, 9H = 좌
    rnfl_angles = {f"CLOCKHOUR_{h}_OD": (h % 12) * 30.0 for h in range(1, 13)}

    # feature → shap 값 매핑
    shap_dict = {f: v for f, v in zip(feat_list, shap_vals)}

    gc_vals = [shap_dict.get(f, 0.0) for f in gc_feat_order]
    rnfl_vals = [shap_dict.get(f, 0.0) for f in rnfl_feat_order]

    # 전체 범위에서 공통 color scale
    all_vals = np.array(gc_vals + rnfl_vals, dtype=float)
    v_abs = float(np.nanmax(np.abs(all_vals))) if all_vals.size else 0.0
    if v_abs <= 0:
        v_abs = 1.0
    norm = Normalize(vmin=-v_abs, vmax=v_abs)
    cmap = plt.cm.coolwarm

    fig, axes = plt.subplots(
        1, 2, subplot_kw={"projection": "polar"}, figsize=(9, 4)
    )

    # 공통 polar 설정: 0deg 위, 시계방향 증가
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # ---------------- GCIPL map ----------------
    ax_gc = axes[0]
    thetas_gc = [math.radians(gc_angles_deg[f]) for f in gc_feat_order]
    width_gc = math.radians(60.0)

    for theta, val, label in zip(
        thetas_gc,
        gc_vals,
        ["TempSup", "Sup", "NasSup", "NasInf", "Inf", "TempInf"],
    ):
        ax_gc.bar(
            theta,
            1.0,
            width=width_gc,
            bottom=0.0,
            color=cmap(norm(val)),
            edgecolor="black",
            linewidth=1.0,
        )
        # label 은 바깥쪽에 배치
        r_label = 1.15
        ax_gc.text(theta, r_label, label, ha="center", va="center", fontsize=10)

    ax_gc.set_title("GCIPL sector\ncontribution (OD)", fontsize=12)

    # ---------------- RNFL map ----------------
    ax_rnfl = axes[1]
    width_rnfl = math.radians(30.0)
    for h, val in zip(range(1, 13), rnfl_vals):
        feat = f"CLOCKHOUR_{h}_OD"
        theta = math.radians(rnfl_angles[feat])
        ax_rnfl.bar(
            theta,
            1.0,
            width=width_rnfl,
            bottom=0.0,
            color=cmap(norm(val)),
            edgecolor="black",
            linewidth=0.8,
        )

    # 주요 clock-hour 라벨 (3H, 6H, 9H, 12H)
    label_hours = [12, 3, 6, 9]
    for h in label_hours:
        theta = math.radians((h % 12) * 30.0)
        r_label = 1.15
        ax_rnfl.text(theta, r_label, f"{h}H", ha="center", va="center", fontsize=10)

    ax_rnfl.set_title("RNFL clock-hour\ncontribution (OD)", fontsize=12)

    # 컬러바
    cax = fig.add_axes([0.46, 0.15, 0.02, 0.6])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("SHAP value\n(impact on optic nerve age)", fontsize=10)

    fig.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
    return fig


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Optic Age™ calculator",
        layout="wide",
    )

    st.title("Optic Age™ calculator")
    st.markdown(
        """
        이 도구는 KNHANES 기반 **풀 OCT + sex** 모델을 사용하여
        개별 눈의 optic nerve biological age를 추정하고, ΔAge 및
        Optic Nerve Aging Score (ONAS, percentile)를 제공합니다.

        - 대상 연령: **40–79세**
        - 축장: **AL_R ≤ 26 mm**에서 개발된 모델
        - 입력 수치 범위가 비정상적으로 벗어나는 경우 결과 해석에 주의가 필요합니다.
        """
    )

    # 미리 로딩(에러가 있으면 여기서 바로 stop)
    _models_by_age, _features_by_age = load_models_and_features()
    age_adj = compute_age_adjust_params()
    sigma = compute_onas_sigma()

    with st.form("input_form"):
        st.subheader("1. Demographics & biometry")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input(
                "Chronological age (years)",
                min_value=40.0,
                max_value=79.99,
                value=55.0,
                step=1.0,
            )
        with col2:
            sex = st.selectbox("Sex", ["Male", "Female"], index=0)
        with col3:
            al = st.number_input(
                "Axial length (AL_R, mm)",
                min_value=20.0,
                max_value=26.0,
                value=24.0,
                step=0.1,
            )

        st.markdown("---")
        st.subheader("2. OCT parameters (right eye, OD)")

        # RNFL clock-hour inputs
        st.markdown("**RNFL clock-hour thickness (µm)**")

        rnfl_inputs: Dict[str, float] = {}
        rnfl_labels_row1 = [1, 2, 3, 4, 5, 6]
        rnfl_labels_row2 = [7, 8, 9, 10, 11, 12]

        col_r1 = st.columns(6)
        for i, h in enumerate(rnfl_labels_row1):
            with col_r1[i]:
                key = f"CLOCKHOUR_{h}_OD"
                rnfl_inputs[key] = st.number_input(
                    f"{h}H",
                    min_value=40.0,
                    max_value=140.0,
                    value=100.0,
                    step=1.0,
                )

        col_r2 = st.columns(6)
        for i, h in enumerate(rnfl_labels_row2):
            with col_r2[i]:
                key = f"CLOCKHOUR_{h}_OD"
                rnfl_inputs[key] = st.number_input(
                    f"{h}H",
                    min_value=40.0,
                    max_value=140.0,
                    value=100.0,
                    step=1.0,
                )

        st.markdown("**GCIPL sectors (µm)**")

        gc_inputs: Dict[str, float] = {}
        col_gc1 = st.columns(3)
        with col_gc1[0]:
            gc_inputs["GC_TEMPSUP_OD"] = st.number_input(
                "TempSup",
                min_value=50.0,
                max_value=110.0,
                value=80.0,
                step=1.0,
            )
        with col_gc1[1]:
            gc_inputs["GC_SUP_OD"] = st.number_input(
                "Sup",
                min_value=50.0,
                max_value=110.0,
                value=80.0,
                step=1.0,
            )
        with col_gc1[2]:
            gc_inputs["GC_NASSUP_OD"] = st.number_input(
                "NasSup",
                min_value=50.0,
                max_value=110.0,
                value=80.0,
                step=1.0,
            )

        col_gc2 = st.columns(3)
        with col_gc2[0]:
            gc_inputs["GC_NASINF_OD"] = st.number_input(
                "NasInf",
                min_value=50.0,
                max_value=110.0,
                value=80.0,
                step=1.0,
            )
        with col_gc2[1]:
            gc_inputs["GC_INF_OD"] = st.number_input(
                "Inf",
                min_value=50.0,
                max_value=110.0,
                value=80.0,
                step=1.0,
            )
        with col_gc2[2]:
            gc_inputs["GC_TEMPINF_OD"] = st.number_input(
                "TempInf",
                min_value=50.0,
                max_value=110.0,
                value=80.0,
                step=1.0,
            )

        st.markdown(
            """
            기본값은 **건강한 super-normal 눈**에서 기대되는 평균 범위에 해당합니다.
            실제 환자 값을 그대로 입력하는 것이 가장 정확합니다.
            """
        )

        submitted = st.form_submit_button("Run Optic Age™ model")

    if not submitted:
        return

    # ---------------------------------------------------------------
    # Run model
    # ---------------------------------------------------------------
    try:
        patient = build_patient_dict(age, sex, al, rnfl_inputs, gc_inputs, age_adj)
        y_pred, delta_age, onas, age_bin, shap_vals, feat_list = run_model(patient)
    except Exception as e:
        st.error(f"모델 실행 중 오류가 발생했습니다: {e}")
        st.stop()

    # ---------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Optic Age™ summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Age bin (model)", age_bin)
    with c2:
        st.metric(
            "Predicted optic nerve age (years)", f"{y_pred:.2f}"
        )
    with c3:
        st.metric(
            "ΔAge (optic − chronological, years)", f"{delta_age:.2f}"
        )
    with c4:
        st.metric(
            "Optic Nerve Aging Score (ONAS, %)", f"{onas:.1f}"
        )

    st.caption(
        "ONAS는 ΔAge의 분포를 기준으로 한 percentile입니다. "
        "값이 **높을수록** optic nerve가 또래 평균에 비해 "
        "더 젊고(resilient) 건강하다는 것을 의미합니다."
    )

    # RNFL / GC mean 표시
    st.markdown("##### OCT summary (입력값 기반)")
    rnfl_mean = patient.get("rnfl12_mean", np.nan)
    gc_mean = patient.get("gc6_mean", np.nan)
    c5, c6 = st.columns(2)
    with c5:
        st.write(f"RNFL 12H mean: **{rnfl_mean:.1f} µm**")
    with c6:
        st.write(f"GCIPL 6-sector mean: **{gc_mean:.1f} µm**")

    # ---------------------------------------------------------------
    # Regional vulnerability map
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Regional Vulnerability Map (feature importance)")

    fig = make_regional_vulnerability_plots(shap_vals, feat_list)
    st.pyplot(fig)

    st.caption(
        "붉은 영역은 해당 부위가 optic nerve age를 **젊게(더 두껍고 건강하게)** 만드는 방향으로 기여함을, "
        "푸른 영역은 **더 늙게(더 얇고 취약하게)** 만드는 방향으로 기여함을 의미합니다. "
        "우안(OD)을 기준으로, temporal retina가 왼쪽, nasal retina가 오른쪽입니다. "
        "RNFL map에서 **6H와 Inferior sector는 항상 그림의 맨 아래**에 위치하도록 설정되어 있습니다."
    )


if __name__ == "__main__":
    main()

