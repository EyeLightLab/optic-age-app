# optic_age_app.py
# ----------------------------------------------------------
# Optic Age™ — Optic Nerve Biological Aging Calculator
#  - 40–79세에 대해서만 사용
#  - Age-bin specific models (40s, 50s, 60s, 70s)
#  - Full OCT + sex + age-adjusted RNFL/GC residuals
# ----------------------------------------------------------

import math
from math import pi

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# Paths
# -----------------------------
MODELS_BY_AGE_PATH = "optic_age_models_by_age.pkl"     # dict: age_bin -> model
FEATURES_BY_AGE_PATH = "optic_age_features_by_age.pkl" # dict: age_bin -> [features]
AGE_ADJUST_PATH = "age_adjust_params.pkl"              # dict: {"rnfl12":{a,b}, "gc6":{a,b}}
NORM_TABLE_PATH = "norm_table.csv"                     # RNFL mean/SD, ONAS sigma

# -----------------------------
# Defaults (fallback)
# -----------------------------
DEFAULT_RNFL_MEAN = 95.0
DEFAULT_RNFL_SD = 10.0
DEFAULT_ONAS_SIGMA = 6.0

# RNFL / GCIPL feature lists (for SHAP map)
RNFL_FEATURES = [f"CLOCKHOUR_{i}_OD" for i in range(1, 13)]
GCIPL_FEATURES = [
    "GC_TEMPSUP_OD",
    "GC_SUP_OD",
    "GC_NASSUP_OD",
    "GC_NASINF_OD",
    "GC_INF_OD",
    "GC_TEMPINF_OD",
]

# =========================================================
# 1. Cached loaders
# =========================================================
@st.cache_resource
def load_models_and_features():
    models_by_age = joblib.load(MODELS_BY_AGE_PATH)
    features_by_age = joblib.load(FEATURES_BY_AGE_PATH)
    age_adjust_params = joblib.load(AGE_ADJUST_PATH)
    return models_by_age, features_by_age, age_adjust_params


@st.cache_data
def load_normative_params():
    rnfl_mean = DEFAULT_RNFL_MEAN
    rnfl_sd = DEFAULT_RNFL_SD
    onas_sigma = DEFAULT_ONAS_SIGMA
    try:
        df = pd.read_csv(NORM_TABLE_PATH)
        if "rnfl_mean" in df.columns:
            rnfl_mean = float(df.loc[0, "rnfl_mean"])
        if "rnfl_sd" in df.columns:
            rnfl_sd = float(df.loc[0, "rnfl_sd"])
        if "onas_sigma" in df.columns:
            onas_sigma = float(df.loc[0, "onas_sigma"])
    except Exception:
        # fallback to defaults
        pass
    return rnfl_mean, rnfl_sd, onas_sigma


# =========================================================
# 2. Helper functions
# =========================================================
def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_onas(delta_age: float, onas_sigma: float):
    """
    ONAS percentile (낮을수록 취약):
      Δage ~ N(0, onas_sigma^2) 라고 가정하고,
      Δage가 클수록 (더 늙을수록) percentile이 낮게 계산되도록 설정.
    """
    if onas_sigma <= 0:
        onas_sigma = DEFAULT_ONAS_SIGMA
    z = delta_age / onas_sigma
    percentile = 100.0 * (1.0 - normal_cdf(z))  # Δage ↑ → percentile ↓
    return percentile, z


def compute_rnfl_z_and_percentile(rnfl_value: float, rnfl_mean: float, rnfl_sd: float):
    if rnfl_sd <= 0:
        rnfl_sd = DEFAULT_RNFL_SD
    z = (rnfl_value - rnfl_mean) / rnfl_sd
    percentile = 100.0 * normal_cdf(z)  # higher RNFL = higher percentile
    return z, percentile


def score_color_percentile(p: float) -> str:
    """낮은 percentile = 나쁨(빨강), 높은 percentile = 좋음(파랑)."""
    if p is None or math.isnan(p):
        return "#FFFFFF"
    if p < 20:
        return "#ff4b4b"
    if p < 40:
        return "#ffa94b"
    if p < 60:
        return "#ffffff"
    if p < 80:
        return "#74c0fc"
    return "#4dabf7"


def score_color_delta(delta: float) -> str:
    """ΔAge 색상: + (늙음) = 빨강, 0 근처 = 흰색, − (젊음) = 파랑."""
    if delta is None or math.isnan(delta):
        return "#FFFFFF"
    if delta > 10:
        return "#ff4b4b"
    if delta > 5:
        return "#ffa94b"
    if delta > -5:
        return "#ffffff"
    if delta > -10:
        return "#74c0fc"
    return "#4dabf7"


def render_big_metric(label: str, value_str: str, color: str, help_text: str = ""):
    """커다란 카드 형태 metric."""
    st.markdown(
        f"""
        <div style="
            background-color:#111827;
            border-radius:12px;
            padding:18px 20px;
            border:1px solid #374151;
            height:100%;
        ">
          <div style="font-size:0.9rem; color:#9ca3af; margin-bottom:6px;">
            {label}
          </div>
          <div style="font-size:2.2rem; font-weight:700; color:{color};">
            {value_str}
          </div>
          <div style="font-size:0.80rem; color:#9ca3af; margin-top:4px;">
            {help_text}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_regional_vulnerability_plot(rnfl_shap: np.ndarray, gcipl_shap: np.ndarray):
    """
    RNFL / GCIPL 취약도 polar map.

    - 왼쪽: GCIPL sector map (OD 기준, temporal = 왼쪽, nasal = 오른쪽)
    - 오른쪽: RNFL clock-hour map (12H 위, 6H 아래, 3H 오른쪽, 9H 왼쪽)
    - 컬러바 중앙 = 0, 위쪽 "More vulnerable", 아래쪽 "More resilient"
    """

    rnfl_vals = rnfl_shap
    gc_vals = gcipl_shap

    vmax = max(np.max(np.abs(rnfl_vals)), np.max(np.abs(gc_vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(10, 4))
    fig.patch.set_facecolor("#111827")

    # --------------------------------------------------
    # 1) GCIPL sector map (왼쪽)
    # --------------------------------------------------
    ax_gc = fig.add_subplot(1, 2, 1, polar=True)
    ax_gc.set_facecolor("#111827")
    ax_gc.set_theta_zero_location("N")   # 0 rad = 위쪽
    ax_gc.set_theta_direction(-1)        # 시계 방향

    # 6 sectors: TempSup, Sup, NasSup, NasInf, Inf, TempInf
    gc_order = ["Sup", "NasSup", "NasInf", "Inf", "TempInf", "TempSup"]
    gc_label_map = {
        "TempSup": "TempSup",
        "Sup": "Sup",
        "NasSup": "NasSup",
        "NasInf": "NasInf",
        "Inf": "Inf",
        "TempInf": "TempInf",
    }
    gc_value_map = {
        "TempSup": gc_vals[0],
        "Sup": gc_vals[1],
        "NasSup": gc_vals[2],
        "NasInf": gc_vals[3],
        "Inf": gc_vals[4],
        "TempInf": gc_vals[5],
    }

    num_gc = 6
    width_gc = 2 * pi / num_gc
    centers_gc = []

    for idx, key in enumerate(gc_order):
        center_angle = idx * width_gc
        start = center_angle - width_gc / 2
        centers_gc.append(center_angle)

        val = gc_value_map[key]
        ax_gc.bar(
            start,
            1.0,
            width=width_gc,
            bottom=0.0,
            color=plt.cm.coolwarm(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    gc_labels = [gc_label_map[k] for k in gc_order]
    ax_gc.set_xticks(centers_gc)
    ax_gc.set_xticklabels(gc_labels, color="white", fontsize=10)
    ax_gc.set_yticklabels([])
    ax_gc.set_title("GCIPL sector contribution (OD)", color="white", fontsize=13, pad=12)

    # --------------------------------------------------
    # 2) RNFL clock-hour map (오른쪽)
    # --------------------------------------------------
    ax_rnfl = fig.add_subplot(1, 2, 2, polar=True)
    ax_rnfl.set_facecolor("#111827")
    ax_rnfl.set_theta_zero_location("N")   # 12H 위쪽
    ax_rnfl.set_theta_direction(-1)        # 시계 방향 (clock)

    # rnfl_vals: index 0..11 = 1H..12H
    hour_to_val = {h: rnfl_vals[(h - 1) % 12] for h in range(1, 13)}

    # 12,1,2,...,11 순서로 배치 (12 위쪽, 3 오른쪽, 6 아래, 9 왼쪽)
    hour_order = [12] + list(range(1, 12))
    num_hours = 12
    width_h = 2 * pi / num_hours
    centers_h = []

    for idx, hour in enumerate(hour_order):
        center_angle = idx * width_h
        start = center_angle - width_h / 2
        centers_h.append(center_angle)

        val = hour_to_val[hour]
        ax_rnfl.bar(
            start,
            1.0,
            width=width_h,
            bottom=0.0,
            color=plt.cm.coolwarm(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    hour_labels = [f"{h}H" for h in hour_order]
    ax_rnfl.set_xticks(centers_h)
    ax_rnfl.set_xticklabels(hour_labels, color="white", fontsize=9)
    ax_rnfl.set_yticklabels([])
    ax_rnfl.set_title("RNFL clock-hour contribution (OD)", color="white", fontsize=13, pad=12)

    # --------------------------------------------------
    # 3) Colorbar (가운데; 위/아래 텍스트)
    # --------------------------------------------------
    # colorbar 축은 figure 좌표계 기준 위치 조정
    cax = fig.add_axes([0.48, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
        cax=cax,
    )
    cb.ax.set_ylabel("SHAP value\n(impact on optic nerve age)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white", fontsize=8)

    # 위/아래 텍스트
    fig.text(0.495, 0.90, "More vulnerable", ha="center", va="center", color="white", fontsize=9)
    fig.text(0.495, 0.08, "More resilient", ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    return fig


def get_age_bin(age: int):
    if 40 <= age < 50:
        return "40s"
    if 50 <= age < 60:
        return "50s"
    if 60 <= age < 70:
        return "60s"
    if 70 <= age < 80:
        return "70s"
    return None


# =========================================================
# 3. Main UI
# =========================================================
def main():
    st.set_page_config(
        page_title="Optic Age — Optic Nerve Biological Aging Calculator",
        page_icon="🧠",
        layout="wide",
    )
    plt.style.use("dark_background")

    # Header
    st.markdown(
        """
        <h1 style="margin-bottom:0.2rem;">Optic Age™ — Optic Nerve Biological Aging Calculator</h1>
        <p style="color:#9ca3af; margin-bottom:0.1rem;">
          Developed by <b>Professor Young Kook Kim</b>, Seoul National University Hospital (SNUH)
        </p>
        <p style="color:#6b7280; font-size:0.85rem; margin-bottom:0.4rem;">
          Age-stratified machine-learning models using full OCT metrics from KNHANES 2019–2021
          to estimate the biological aging of the optic nerve.
        </p>
        <p style="color:#f97316; font-size:0.85rem; margin-bottom:1.0rem;">
          <b>Note:</b> Models are trained on <b>40–79 year-old</b> super-normal eyes.
          Please apply the app only to patients between 40 and 79 years of age.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Load models & norms
    with st.spinner("Loading age-stratified models and normative parameters..."):
        models_by_age, features_by_age, age_adjust_params = load_models_and_features()
        rnfl_mean, rnfl_sd, onas_sigma = load_normative_params()
    st.success("Models and normative parameters loaded successfully.")

    # -------------------------
    # 1. Input section
    # -------------------------
    st.markdown("### 1. Input Patient Data (Right Eye Only, age 40–79)")

    with st.form("input_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Chronological age (years)", 40, 79, 60)
        with c2:
            sex_label = st.selectbox(
                "Sex",
                [("Male", 1), ("Female", 2)],
                format_func=lambda x: x[0],
            )
            sex = sex_label[1]
        with c3:
            al_r = st.number_input("Axial length AL_R (mm)", 20.0, 30.0, 24.0, step=0.1)
        with c4:
            disc_area = st.number_input("Disc area DISCAREA_OD (mm²)", 1.0, 4.0, 2.0, step=0.1)

        c5, c6, c7 = st.columns(3)
        with c5:
            vcd = st.number_input(
                "Vertical C/D ratio (VERTICAL_CD_RATIO_OD)",
                0.0,
                1.0,
                0.4,
                step=0.01,
            )
        with c6:
            rimarea = st.number_input("Rim area RIMAREA_OD (mm²)", 0.5, 3.0, 1.5, step=0.05)
        with c7:
            cup_vol = st.number_input("Cup volume CUPVOLUME_OD (mm³)", 0.0, 1.0, 0.2, step=0.01)

        # RNFL clock-hours
        st.markdown("#### RNFL 12 Clock-Hour (µm)")
        rnfl_vals = []
        row1 = st.columns(6)
        row2 = st.columns(6)
        for i in range(1, 7):
            with row1[i - 1]:
                rnfl_vals.append(
                    st.number_input(f"{i}H", 40.0, 180.0, 100.0, step=0.5, key=f"rnfl_{i}")
                )
        for i in range(7, 13):
            with row2[i - 7]:
                rnfl_vals.append(
                    st.number_input(f"{i}H", 40.0, 180.0, 100.0, step=0.5, key=f"rnfl_{i}")
                )

        # GCIPL sectors
        st.markdown("#### GCIPL sectors (µm)")
        g1, g2, g3 = st.columns(3)
        g4, g5, g6 = st.columns(3)
        gc_tempsup = g1.number_input("TempSup", 40.0, 120.0, 80.0, step=0.5)
        gc_sup = g2.number_input("Sup", 40.0, 120.0, 80.0, step=0.5)
        gc_nassup = g3.number_input("NasSup", 40.0, 120.0, 80.0, step=0.5)
        gc_nasinf = g4.number_input("NasInf", 40.0, 120.0, 80.0, step=0.5)
        gc_inf = g5.number_input("Inf", 40.0, 120.0, 80.0, step=0.5)
        gc_tempinf = g6.number_input("TempInf", 40.0, 120.0, 80.0, step=0.5)

        # Advanced OCT (옵션)
        with st.expander("Advanced OCT parameters (optional – default values are normative-like)"):
            cmt = st.number_input("CMT_OD (µm)", 150.0, 400.0, 250.0, step=1.0)
            avg_thk = st.number_input("AVERAGETHICKNESS_OD (µm)", 200.0, 350.0, 270.0, step=1.0)

            quad_t = st.number_input("QUADRANT_T_OD (µm)", 40.0, 180.0, 110.0, step=0.5)
            quad_s = st.number_input("QUADRANT_S_OD (µm)", 40.0, 180.0, 120.0, step=0.5)
            quad_n = st.number_input("QUADRANT_N_OD (µm)", 40.0, 180.0, 90.0, step=0.5)
            quad_i = st.number_input("QUADRANT_I_OD (µm)", 40.0, 180.0, 120.0, step=0.5)

            rnfl_sup = st.number_input("RNFL_SUP_OD (µm)", 40.0, 180.0, 120.0, step=0.5)
            rnfl_inf = st.number_input("RNFL_INF_OD (µm)", 40.0, 180.0, 120.0, step=0.5)
            rnfl_nsup = st.number_input("RNFL_NASSUP_OD (µm)", 40.0, 180.0, 90.0, step=0.5)
            rnfl_ninf = st.number_input("RNFL_NASINF_OD (µm)", 40.0, 180.0, 90.0, step=0.5)
            rnfl_tsup = st.number_input("RNFL_TEMPSUP_OD (µm)", 40.0, 180.0, 90.0, step=0.5)
            rnfl_tinf = st.number_input("RNFL_TEMPINF_OD (µm)", 40.0, 180.0, 90.0, step=0.5)
            rnfl_min = st.number_input("RNFL_MINIMUM_OD (µm)", 20.0, 120.0, 60.0, step=0.5)

            or_tsup = st.number_input("OR_TEMPSUP_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            or_sup = st.number_input("OR_SUP_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            or_nsup = st.number_input("OR_NASSUP_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            or_ninf = st.number_input("OR_NASINF_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            or_inf = st.number_input("OR_INF_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            or_tinf = st.number_input("OR_TEMPINF_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            or_avg = st.number_input("OR_AVERAGE_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            or_min = st.number_input("OR_MINIMUM_OD (µm)", 150.0, 350.0, 200.0, step=1.0)

            cube_thk = st.number_input("CUBEAVGTHICKNESS_ILMRPE_OD (µm)", 200.0, 350.0, 270.0, step=1.0)
            cube_thk_fit = st.number_input(
                "CUBEAVGTHICKNESS_ILMRPEFIT_OD (µm)", 200.0, 350.0, 270.0, step=1.0
            )
            cube_vol = st.number_input("CUBEVOLUME_ILMRPE_OD (mm³)", 6.0, 14.0, 10.0, step=0.1)
            cube_vol_fit = st.number_input(
                "CUBEVOLUME_ILMRPEFIT_OD (mm³)", 6.0, 14.0, 10.0, step=0.1
            )
            c_ilmrpe = st.number_input("C_ILMRPE_OD (µm)", 150.0, 350.0, 250.0, step=1.0)
            c_ilmrpe_fit = st.number_input("C_ILMRPEFIT_OD (µm)", 150.0, 350.0, 250.0, step=1.0)

            sfct = st.number_input("SFCT_OD (µm)", 100.0, 500.0, 250.0, step=1.0)

            z_center = st.number_input("Z_CENTER_OD (µm)", 150.0, 350.0, 260.0, step=1.0)
            z_ir = st.number_input("Z_INNERRIGHT_OD (µm)", 150.0, 350.0, 280.0, step=1.0)
            z_is = st.number_input("Z_INNERSUPERIOR_OD (µm)", 150.0, 350.0, 280.0, step=1.0)
            z_il = st.number_input("Z_INNERLEFT_OD (µm)", 150.0, 350.0, 280.0, step=1.0)
            z_ii = st.number_input("Z_INNERINFERIOR_OD (µm)", 150.0, 350.0, 280.0, step=1.0)
            z_or = st.number_input("Z_OUTERRIGHT_OD (µm)", 150.0, 350.0, 270.0, step=1.0)
            z_os = st.number_input("Z_OUTERSUPERIOR_OD (µm)", 150.0, 350.0, 270.0, step=1.0)
            z_ol = st.number_input("Z_OUTERLEFT_OD (µm)", 150.0, 350.0, 270.0, step=1.0)
            z_oi = st.number_input("Z_OUTERINFERIOR_OD (µm)", 150.0, 350.0, 270.0, step=1.0)

        submitted = st.form_submit_button("Run Optic Age™ model")

    if not submitted:
        st.stop()

    # -------------------------
    # Age bin 선택 (40–79만 허용)
    # -------------------------
    age_bin = get_age_bin(age)
    if age_bin is None or age_bin not in models_by_age:
        st.error("This app currently supports only ages between 40 and 79 years.")
        st.stop()

    model = models_by_age[age_bin]
    feat_list = features_by_age[age_bin]

    # -------------------------
    # 2. Age-adjusted derived features
    # -------------------------
    rnfl12_mean = float(np.mean(rnfl_vals))
    gc6_mean = float(np.mean([gc_tempsup, gc_sup, gc_nassup, gc_nasinf, gc_inf, gc_tempinf]))

    rnfl_params = age_adjust_params["rnfl12"]
    gc6_params = age_adjust_params["gc6"]

    rnfl_pred = rnfl_params["a"] + rnfl_params["b"] * age
    gc6_pred = gc6_params["a"] + gc6_params["b"] * age

    rnfl12_resid = rnfl12_mean - rnfl_pred
    gc6_resid = gc6_mean - gc6_pred

    # -------------------------
    # 3. Build patient feature dict
    # -------------------------
    patient = {}

    patient["sex"] = sex
    patient["AL_R"] = al_r
    patient["DISCAREA_OD"] = disc_area
    patient["VERTICAL_CD_RATIO_OD"] = vcd
    patient["RIMAREA_OD"] = rimarea
    patient["CUPVOLUME_OD"] = cup_vol

    # RNFL clock-hours
    for i, val in enumerate(rnfl_vals, start=1):
        patient[f"CLOCKHOUR_{i}_OD"] = val

    # GCIPL
    patient["GC_TEMPSUP_OD"] = gc_tempsup
    patient["GC_SUP_OD"] = gc_sup
    patient["GC_NASSUP_OD"] = gc_nassup
    patient["GC_NASINF_OD"] = gc_nasinf
    patient["GC_INF_OD"] = gc_inf
    patient["GC_TEMPINF_OD"] = gc_tempinf

    # Additional OCT
    patient["CMT_OD"] = cmt
    patient["AVERAGETHICKNESS_OD"] = avg_thk

    patient["QUADRANT_T_OD"] = quad_t
    patient["QUADRANT_S_OD"] = quad_s
    patient["QUADRANT_N_OD"] = quad_n
    patient["QUADRANT_I_OD"] = quad_i

    patient["RNFL_SUP_OD"] = rnfl_sup
    patient["RNFL_INF_OD"] = rnfl_inf
    patient["RNFL_NASSUP_OD"] = rnfl_nsup
    patient["RNFL_NASINF_OD"] = rnfl_ninf
    patient["RNFL_TEMPSUP_OD"] = rnfl_tsup
    patient["RNFL_TEMPINF_OD"] = rnfl_tinf
    patient["RNFL_MINIMUM_OD"] = rnfl_min

    patient["OR_TEMPSUP_OD"] = or_tsup
    patient["OR_SUP_OD"] = or_sup
    patient["OR_NASSUP_OD"] = or_nsup
    patient["OR_NASINF_OD"] = or_ninf
    patient["OR_INF_OD"] = or_inf
    patient["OR_TEMPINF_OD"] = or_tinf
    patient["OR_AVERAGE_OD"] = or_avg
    patient["OR_MINIMUM_OD"] = or_min

    patient["CUBEAVGTHICKNESS_ILMRPE_OD"] = cube_thk
    patient["CUBEAVGTHICKNESS_ILMRPEFIT_OD"] = cube_thk_fit
    patient["CUBEVOLUME_ILMRPE_OD"] = cube_vol
    patient["CUBEVOLUME_ILMRPEFIT_OD"] = cube_vol_fit
    patient["C_ILMRPE_OD"] = c_ilmrpe
    patient["C_ILMRPEFIT_OD"] = c_ilmrpe_fit

    patient["SFCT_OD"] = sfct

    patient["Z_CENTER_OD"] = z_center
    patient["Z_INNERRIGHT_OD"] = z_ir
    patient["Z_INNERSUPERIOR_OD"] = z_is
    patient["Z_INNERLEFT_OD"] = z_il
    patient["Z_INNERINFERIOR_OD"] = z_ii
    patient["Z_OUTERRIGHT_OD"] = z_or
    patient["Z_OUTERSUPERIOR_OD"] = z_os
    patient["Z_OUTERLEFT_OD"] = z_ol
    patient["Z_OUTERINFERIOR_OD"] = z_oi

    # age-adjusted features
    patient["RNFL12_MEAN"] = rnfl12_mean
    patient["RNFL12_RESID"] = rnfl12_resid
    patient["GC6_MEAN"] = gc6_mean
    patient["GC6_RESID"] = gc6_resid

    # feature 순서대로 벡터 생성
    x_vec = np.array([[patient[f] for f in feat_list]], dtype=float)

    # -------------------------
    # 4. Predict & scores
    # -------------------------
    predicted_age = float(model.predict(x_vec)[0])
    delta_age = predicted_age - age

    onas_percentile, _ = compute_onas(delta_age, onas_sigma)
    rnfl_z, rnfl_pct = compute_rnfl_z_and_percentile(rnfl12_mean, rnfl_mean, rnfl_sd)
    rnfl_pct_color = score_color_percentile(rnfl_pct)

    # -------------------------
    # 5. Optic Nerve Age & ONAS
    # -------------------------
    st.markdown("### 2. Optic Nerve Age & Optic Nerve Aging Score (ONAS)")

    cA, cB, cC = st.columns(3)

    colA_color = score_color_delta(delta_age)
    with cA:
        render_big_metric(
            "Predicted optic nerve age (years)",
            f"{predicted_age:.1f}",
            colA_color,
            help_text=f"Age-bin model used: {age_bin} (trained on KNHANES 40–79y super-normal eyes).",
        )

    colB_color = score_color_delta(delta_age)
    with cB:
        render_big_metric(
            "ΔAge (optic − chronological, years)",
            f"{delta_age:+.1f}",
            colB_color,
            help_text="Positive values: older-than-expected optic nerve. Negative: younger / more resilient.",
        )

    onas_color = score_color_percentile(onas_percentile)
    with cC:
        render_big_metric(
            "Optic Nerve Aging Score percentile (ONAS)",
            f"{onas_percentile:.1f} %",
            onas_color,
            help_text=(
                "ONAS percentile based on ΔAge distribution in KNHANES super-normal eyes. "
                "Lower percentile → more vulnerable; higher percentile → more resilient."
            ),
        )

    st.markdown(
        """
        **ONAS interpretation**  
        • Lower ONAS percentile → more accelerated aging / higher structural vulnerability.  
        • Higher ONAS percentile → slower aging / more resilient optic nerve.
        """
    )

    # -------------------------
    # 6. RNFL Normative Position
    # -------------------------
    st.markdown("### 3. RNFL Normative Position (KNHANES-based, 40–79y)")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_big_metric(
            "Average RNFL thickness (µm)",
            f"{rnfl12_mean:.1f}",
            "#e5e7eb",
            help_text="Arithmetic mean of CLOCKHOUR 1–12 (OD).",
        )

    z_color = score_color_delta(-rnfl_z)  # thinner (z < 0) = worse (red쪽)
    with c2:
        render_big_metric(
            "RNFL Z-score vs super-normal",
            f"{rnfl_z:+.2f}",
            z_color,
            help_text="Negative Z-score indicates thinner-than-average RNFL (for 40–79y super-normal eyes).",
        )

    with c3:
        render_big_metric(
            "RNFL thickness percentile",
            f"{rnfl_pct:.1f} %",
            rnfl_pct_color,
            help_text="Lower percentile means thinner RNFL than most of the 40–79y reference population.",
        )

    st.markdown(
        f"""
        Percentiles are computed using a normal approximation with mean ≈ {rnfl_mean:.1f} µm and
        SD ≈ {rnfl_sd:.1f} µm derived from KNHANES 40–79 year-old super-normal eyes
        (AL_R ≤ 26 mm, no major ocular disease).
        """
    )

    # -------------------------
    # 7. Regional Vulnerability Map
    # -------------------------
    st.markdown("### 4. Regional Vulnerability Map (Feature Importance)")

    try:
        import shap  # type: ignore

        @st.cache_resource
        def _build_shap_explainer(_model):
            return shap.TreeExplainer(_model)

        explainer = _build_shap_explainer(model)
        shap_values = explainer.shap_values(x_vec)[0]

        shap_dict = {feat: val for feat, val in zip(feat_list, shap_values)}
        rnfl_shap = np.array([shap_dict.get(f, 0.0) for f in RNFL_FEATURES])
        gcipl_shap = np.array([shap_dict.get(f, 0.0) for f in GCIPL_FEATURES])

        fig = build_regional_vulnerability_plot(rnfl_shap, gcipl_shap)
        st.pyplot(fig, use_container_width=True)

        st.caption(
            "Red regions contribute to an older (more vulnerable) optic nerve age, "
            "while blue regions contribute to a younger (more resilient) optic nerve. "
            "Orientation is for the right eye (OD): temporal retina is shown on the left, nasal on the right."
        )
    except Exception as e:
        st.warning(
            "SHAP-based regional vulnerability map could not be generated "
            f"(missing `shap` package or other error: {e}). "
            "The main Optic Age™ scores above remain valid."
        )


if __name__ == "__main__":
    main()

