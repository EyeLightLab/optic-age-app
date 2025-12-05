# optic_age_app.py
# Optic Age™ — Optic Nerve Biological Aging Calculator
# (no raw KNHANES DB required; uses only pre-trained model + summary stats)

import math
from math import pi

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# 0. Paths & basic settings
# -----------------------------
MODEL_PATH = "optic_age_model_tuned.pkl"      # tuned OD-only model
FEATURES_PATH = "optic_age_features.pkl"      # list of feature names used in the model
NORM_TABLE_PATH = "norm_table.csv"           # small summary table (optional override)

# 기본값 (norm_table.csv 가 없거나 형식이 안 맞을 때 사용)
DEFAULT_RNFL_MEAN = 100.0      # µm, KNHANES 분석값으로 나중에 교체 가능
DEFAULT_RNFL_SD = 10.0         # µm
DEFAULT_ONAS_SIGMA = 6.0       # years, Δage 분산의 대략적인 표준편차

RNFL_FEATURES = [
    f"CLOCKHOUR_{i}_OD" for i in range(1, 13)
]
GCIPL_FEATURES = [
    "GC_TEMPSUP_OD",
    "GC_SUP_OD",
    "GC_NASSUP_OD",
    "GC_NASINF_OD",
    "GC_INF_OD",
    "GC_TEMPINF_OD",
]

CORE_FEATURES = RNFL_FEATURES + GCIPL_FEATURES + [
    "GC_MINIMUM_OD",
    "RNFL_AVERAGE_OD",
    "GC_AVERAGE_OD",
    "DISCAREA_OD",
    "VERTICAL_CD_RATIO_OD",
    "AL_R",
    "sex",
]

# -----------------------------
# 1. Cached loaders
# -----------------------------
@st.cache_resource
def load_model_and_features():
    """Load pretrained model & feature list (no KNHANES raw data)."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols


@st.cache_data
def load_normative_params():
    """
    Load summary KNHANES-derived parameters for RNFL & ONAS.
    norm_table.csv (optional) should have columns:
        rnfl_mean, rnfl_sd, onas_sigma
    Only the first row is used.
    """
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
        # quietly fall back to defaults
        pass

    return rnfl_mean, rnfl_sd, onas_sigma


# -----------------------------
# 2. Helper functions
# -----------------------------
def normal_cdf(z: float) -> float:
    """Standard normal CDF using error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_onas(delta_age: float, onas_sigma: float):
    """
    Optic Nerve Aging Score (ONAS):
    - Input : delta_age = predicted_age - chronological_age
    - ONAS percentile : lower = weaker / more vulnerable.
      We assume Δage ~ N(0, onas_sigma^2) in super-normal population.
    """
    if onas_sigma <= 0:
        onas_sigma = DEFAULT_ONAS_SIGMA

    z = delta_age / onas_sigma
    # older (delta_age>0) -> lower percentile
    percentile = 100.0 * (1.0 - normal_cdf(z))
    return percentile, z


def compute_rnfl_z_and_percentile(rnfl_value: float, rnfl_mean: float, rnfl_sd: float):
    """
    RNFL Z-score & percentile vs KNHANES super-normal distribution.
    - Higher RNFL is better; lower percentile = thinner / more vulnerable.
    """
    if rnfl_sd <= 0:
        rnfl_sd = DEFAULT_RNFL_SD

    z = (rnfl_value - rnfl_mean) / rnfl_sd
    percentile = 100.0 * normal_cdf(z)
    return z, percentile


def score_color_percentile(p: float) -> str:
    """Blue-white-red color depending on percentile (low is bad)."""
    if p is None or math.isnan(p):
        return "#FFFFFF"
    if p < 20:
        return "#ff4b4b"      # red (worst)
    if p < 40:
        return "#ffa94b"      # orange
    if p < 60:
        return "#ffffff"      # white
    if p < 80:
        return "#74c0fc"      # light blue
    return "#4dabf7"          # strong blue (best)


def score_color_delta(delta: float) -> str:
    """Color for Δage (older vs younger)."""
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
    """Single large metric card with colored text."""
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


def build_regional_vulnerability_plot(
    rnfl_shap: np.ndarray,
    gcipl_shap: np.ndarray,
):
    """
    Create polar maps for RNFL clock-hour and GCIPL sector contribution.

    오른눈(OD) 기준:
      - 플롯 왼쪽: 시신경유두 (ONH) RNFL 12H
      - 플롯 오른쪽: 황반 GCIPL 6 sectors
      - 컬러바 중앙이 0, warm = optic nerve older (worse), cool = younger (better)
    """
    # RNFL 12H
    rnfl_vals = rnfl_shap
    # GCIPL 6 sectors : [TempSup, Sup, NasSup, NasInf, Inf, TempInf]
    gc_vals = gcipl_shap

    vmax = max(np.max(np.abs(rnfl_vals)), np.max(np.abs(gc_vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(9, 4))
    fig.patch.set_facecolor("#111827")

    # RNFL clock-hour (left)
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax1.set_facecolor("#111827")

    # 우안 기준: temporal 쪽이 왼쪽(9H), nasal이 오른쪽(3H) 에 오도록 설정
    # theta=0을 nasal (3H) 로 두고, 시계방향으로 진행
    num = 12
    theta = np.linspace(0, 2 * pi, num + 1)
    width = 2 * pi / num
    # RNFL shap 값을 1H..12H 순서로 받았다고 가정
    for i in range(num):
        val = rnfl_vals[i]
        start = theta[i]
        ax1.bar(
            start,
            1.0,
            width=width,
            bottom=0.0,
            color=plt.cm.coolwarm(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    # tick labels (우안 기준)
    hour_labels = ["3H", "2H", "1H", "12H", "11H", "10H", "9H", "8H", "7H", "6H", "5H", "4H"]
    ax1.set_xticks(theta[:-1] + width / 2)
    ax1.set_xticklabels(hour_labels, color="white", fontsize=9)
    ax1.set_yticklabels([])
    ax1.set_title("RNFL clock-hour\ncontribution (OD)", color="white", fontsize=11, pad=10)

    # GCIPL sectors (right)
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    ax2.set_facecolor("#111827")

    # 6 sectors: TempSup, Sup, NasSup, NasInf, Inf, TempInf
    num_gc = 6
    theta_gc = np.linspace(0, 2 * pi, num_gc + 1)
    width_gc = 2 * pi / num_gc
    for i in range(num_gc):
        val = gc_vals[i]
        start = theta_gc[i]
        ax2.bar(
            start,
            1.0,
            width=width_gc,
            bottom=0.0,
            color=plt.cm.coolwarm(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    # 라벨: 우안 기준으로 temporal이 왼쪽, nasal이 오른쪽
    gc_labels = ["TempSup", "Sup", "NasSup", "NasInf", "Inf", "TempInf"]
    ax2.set_xticks(theta_gc[:-1] + width_gc / 2)
    ax2.set_xticklabels(gc_labels, color="white", fontsize=9)
    ax2.set_yticklabels([])
    ax2.set_title("GCIPL sector\ncontribution (OD)", color="white", fontsize=11, pad=10)

    # 가운데 세로 컬러바
    cax = fig.add_axes([0.47, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
        cax=cax,
    )
    cb.ax.set_ylabel("SHAP value\n(impact on optic nerve age)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white", fontsize=8)

    plt.tight_layout()
    return fig


# -----------------------------
# 3. Main UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Optic Age — Optic Nerve Biological Aging Calculator",
        page_icon="🧠",
        layout="wide",
    )

    # Dark theme background adjustment (for embedded matplotlib)
    plt.style.use("dark_background")

    # ----- Header -----
    st.markdown(
        """
        <h1 style="margin-bottom:0.2rem;">Optic Age™ — Optic Nerve Biological Aging Calculator</h1>
        <p style="color:#9ca3af; margin-bottom:0.1rem;">
          Developed by <b>Professor Young Kook Kim</b>, Seoul National University Hospital (SNUH)
        </p>
        <p style="color:#6b7280; font-size:0.85rem; margin-bottom:1.0rem;">
          A machine-learning based clinical decision support tool estimating the biological aging
          of the optic nerve using OCT metrics from KNHANES 2019–2021.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Load model & summary parameters
    with st.spinner("Loading model and KNHANES-derived summary parameters..."):
        model, feature_cols = load_model_and_features()
        rnfl_mean, rnfl_sd, onas_sigma = load_normative_params()

    st.success("Model and summary parameters loaded successfully.")

    # =============================
    # 1. Patient input (OD only)
    # =============================
    st.markdown("### 1. Input Patient Data (Right Eye Only)")

    with st.form("input_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Chronological age (years)", 20, 90, 50)
        with c2:
            sex_label = st.selectbox("Sex", [("Male", 1), ("Female", 2)], format_func=lambda x: x[0])
            sex = sex_label[1]
        with c3:
            al_r = st.number_input("Axial length AL_R (mm)", 20.0, 30.0, 24.0, step=0.1)
        with c4:
            disc_area = st.number_input("Disc area DISCAREA_OD (mm²)", 1.0, 4.0, 2.0, step=0.1)

        c5, c6 = st.columns(2)
        with c5:
            vcd = st.number_input("Vertical C/D ratio (VERTICAL_CD_RATIO_OD)", 0.0, 1.0, 0.4, step=0.01)
        with c6:
            rnfl_avg = st.number_input("Average RNFL thickness (RNFL_AVERAGE_OD, µm)", 50.0, 140.0, 95.0, step=0.5)

        st.markdown("#### RNFL 12 Clock-Hour (µm)")
        rnfl_cols = []
        rnfl_row1 = st.columns(6)
        rnfl_row2 = st.columns(6)
        for i in range(1, 7):
            with rnfl_row1[i - 1]:
                rnfl_cols.append(
                    st.number_input(f"{i}H", 40.0, 180.0, 100.0, step=0.5, key=f"rnfl_{i}")
                )
        for i in range(7, 13):
            with rnfl_row2[i - 7]:
                rnfl_cols.append(
                    st.number_input(f"{i}H", 40.0, 180.0, 100.0, step=0.5, key=f"rnfl_{i}")
                )

        st.markdown("#### GCIPL sectors (µm)")
        g1, g2, g3 = st.columns(3)
        g4, g5, g6 = st.columns(3)
        gc_tempsup = g1.number_input("TempSup", 40.0, 120.0, 80.0, step=0.5)
        gc_sup = g2.number_input("Sup", 40.0, 120.0, 80.0, step=0.5)
        gc_nassup = g3.number_input("NasSup", 40.0, 120.0, 80.0, step=0.5)
        gc_nasinf = g4.number_input("NasInf", 40.0, 120.0, 80.0, step=0.5)
        gc_inf = g5.number_input("Inf", 40.0, 120.0, 80.0, step=0.5)
        gc_tempinf = g6.number_input("TempInf", 40.0, 120.0, 80.0, step=0.5)

        gc_min = st.number_input("GCIPL minimum (GC_MINIMUM_OD, µm)", 40.0, 120.0, 70.0, step=0.5)
        gc_avg = st.number_input("GCIPL average (GC_AVERAGE_OD, µm)", 40.0, 120.0, 80.0, step=0.5)

        submitted = st.form_submit_button("Run Optic Age™ model")

    if not submitted:
        st.stop()

    # =============================
    # 2. Build feature vector
    # =============================
    patient = {
        "age": age,
        "sex": sex,
        "AL_R": al_r,
        "DISCAREA_OD": disc_area,
        "VERTICAL_CD_RATIO_OD": vcd,
        "RNFL_AVERAGE_OD": rnfl_avg,
        "GC_MINIMUM_OD": gc_min,
        "GC_AVERAGE_OD": gc_avg,
        "GC_TEMPSUP_OD": gc_tempsup,
        "GC_SUP_OD": gc_sup,
        "GC_NASSUP_OD": gc_nassup,
        "GC_NASINF_OD": gc_nasinf,
        "GC_INF_OD": gc_inf,
        "GC_TEMPINF_OD": gc_tempinf,
    }
    for i, val in enumerate(rnfl_cols, start=1):
        patient[f"CLOCKHOUR_{i}_OD"] = val

    # align with training feature order
    x_vec = np.array([[patient[c] for c in feature_cols]], dtype=float)

    # =============================
    # 3. Predict & compute scores
    # =============================
    predicted_age = float(model.predict(x_vec)[0])
    delta_age = predicted_age - age

    onas_percentile, onas_z = compute_onas(delta_age, onas_sigma)
    rnfl_z, rnfl_pct = compute_rnfl_z_and_percentile(rnfl_avg, rnfl_mean, rnfl_sd)

    # =============================
    # 4. Display Optic Age & ONAS
    # =============================
    st.markdown("### 2. Optic Nerve Age & Optic Nerve Aging Score (ONAS)")

    colA, colB, colC = st.columns(3)

    # Predicted optic nerve age
    colA_color = score_color_delta(delta_age)
    render_big_metric(
        "Predicted optic nerve age (years)",
        f"{predicted_age:.1f}",
        colA_color,
        help_text="Estimated biological age of the optic nerve based on OCT features.",
    )

    # ΔAge
    colB_color = score_color_delta(delta_age)
    render_big_metric(
        "ΔAge (optic − chronological, years)",
        f"{delta_age:+.1f}",
        colB_color,
        help_text="Positive values indicate an older-than-expected optic nerve; "
        "negative values indicate a younger, more resilient nerve.",
    )

    # ONAS percentile
    onas_color = score_color_percentile(onas_percentile)
    render_big_metric(
        "Optic Nerve Aging Score percentile (ONAS, higher = stronger / more resilient)",
        f"{onas_percentile:.1f} %",
        onas_color,
        help_text=(
            "ONAS percentile is calculated from the estimated distribution of ΔAge in KNHANES-based "
            "super-normal eyes. Lower ONAS percentile indicates a weaker or more vulnerable optic nerve; "
            "higher percentile indicates a stronger, more resilient nerve."
        ),
    )

    st.markdown(
        """
        **ONAS category explanation**  
        * Lower ONAS percentile → more accelerated aging / greater vulnerability.  
        * Higher ONAS percentile → slower aging / more resilient optic nerve.
        """,
    )

    # =============================
    # 5. RNFL normative position
    # =============================
    st.markdown("### 3. RNFL Normative Position (KNHANES-based)")

    c1, c2, c3 = st.columns(3)

    render_big_metric(
        "RNFL_AVERAGE_OD (µm)",
        f"{rnfl_avg:.1f}",
        "#e5e7eb",
        help_text="Average peripapillary RNFL thickness of the right eye.",
    )

    z_color = score_color_delta(-rnfl_z)  # more negative z = worse
    render_big_metric(
        "RNFL Z-score vs super-normal",
        f"{rnfl_z:+.2f}",
        z_color,
        help_text=(
            "Z-score relative to KNHANES super-normal RNFL distribution. "
            "Negative values indicate thinner-than-average RNFL."
        ),
    )

    pct_color = score_color_percentile(rnfl_pct)
    render_big_metric(
        "RNFL thickness percentile (lower = thinner / more vulnerable)",
        f"{rnfl_pct:.1f} %",
        pct_color,
        help_text=(
            "Percentile based on KNHANES super-normal RNFL thickness. "
            "A lower percentile means the RNFL is thinner than most of the reference population, "
            "suggesting more advanced structural loss."
        ),
    )

    st.markdown(
        f"""
        RNFL percentiles are computed using a normal approximation with mean ≈ {rnfl_mean:.1f} µm and
        standard deviation ≈ {rnfl_sd:.1f} µm derived from KNHANES super-normal eyes (age ≥20 years,
        without glaucoma or major retinal disease).
        """
    )

    # =============================
    # 6. Regional vulnerability maps (feature importance)
    # =============================
    st.markdown("### 4. Regional Vulnerability Map (Feature Importance)")

    # ---- SHAP computation ----
    try:
        import shap  # type: ignore

        @st.cache_resource
        def _build_shap_explainer(_model):
            return shap.TreeExplainer(_model)

        explainer = _build_shap_explainer(model)
        shap_values = explainer.shap_values(x_vec)[0]  # 1 x n_features

        # map to RNFL & GCIPL
        shap_dict = {feat: val for feat, val in zip(feature_cols, shap_values)}
        rnfl_shap = np.array([shap_dict.get(f, 0.0) for f in RNFL_FEATURES])
        gcipl_shap = np.array([shap_dict.get(f, 0.0) for f in GCIPL_FEATURES])

        fig = build_regional_vulnerability_plot(rnfl_shap, gcipl_shap)
        st.pyplot(fig, use_container_width=True)

        st.caption(
            "Red regions contribute to an older (more vulnerable) optic nerve age, "
            "while blue regions contribute to a younger (more resilient) optic nerve. "
            "Plots are oriented for the right eye (OD): temporal retina is displayed on the left, "
            "nasal on the right."
        )
    except Exception as e:
        st.warning(
            "SHAP-based regional vulnerability map could not be generated "
            f"(missing `shap` package or other error: {e}). "
            "The main Optic Age™ scores above remain valid."
        )


if __name__ == "__main__":
    main()
