# optic_age_app.py
# ----------------------------------------------------------
# Optic Age™ — Optic Nerve Biological Aging Calculator
#  - Valid for 40–79-year-old patients only
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
MODELS_BY_AGE_PATH = "optic_age_models_by_age.pkl"
FEATURES_BY_AGE_PATH = "optic_age_features_by_age.pkl"
AGE_ADJUST_PATH = "age_adjust_params.pkl"
NORM_TABLE_PATH = "norm_table.csv"

# -----------------------------
# Defaults (fallback)
# -----------------------------
DEFAULT_RNFL_MEAN = 95.0
DEFAULT_RNFL_SD = 10.0
DEFAULT_ONAS_SIGMA = 6.0

# For regional vulnerability map
RNFL_FEATURES = [f"CLOCKHOUR_{i}_OD" for i in range(1, 13)]
GCIPL_FEATURES = [
    "GC_TEMPSUP_OD",
    "GC_SUP_OD",
    "GC_NASSUP_OD",
    "GC_NASINF_OD",
    "GC_INF_OD",
    "GC_TEMPINF_OD",
]


# -----------------------------
# 1. Cached loaders
# -----------------------------
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
        pass
    return rnfl_mean, rnfl_sd, onas_sigma


# -----------------------------
# 2. Helper functions
# -----------------------------
def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_onas(delta_age: float, onas_sigma: float):
    """ONAS percentile: lower = more vulnerable, higher = more resilient."""
    if onas_sigma <= 0:
        onas_sigma = DEFAULT_ONAS_SIGMA
    z = delta_age / onas_sigma
    # older (delta>0) → lower percentile
    percentile = 100.0 * (1.0 - normal_cdf(z))
    return percentile, z


def compute_rnfl_z_and_percentile(rnfl_value: float, rnfl_mean: float, rnfl_sd: float):
    """Higher RNFL is better (thicker)."""
    if rnfl_sd <= 0:
        rnfl_sd = DEFAULT_RNFL_SD
    z = (rnfl_value - rnfl_mean) / rnfl_sd
    percentile = 100.0 * normal_cdf(z)
    return z, percentile


def score_color_percentile(p: float) -> str:
    """
    Color for percentile (low = bad, high = good).
    bad → blue / good → red
    """
    if p is None or math.isnan(p):
        return "#FFFFFF"
    if p < 20:
        return "#1e40af"  # deep blue (worst)
    if p < 40:
        return "#3b82f6"  # blue
    if p < 60:
        return "#ffffff"  # neutral
    if p < 80:
        return "#f97316"  # orange-red
    return "#ef4444"      # strong red (best)


def score_color_delta(delta: float) -> str:
    """
    Color for ΔAge (optic − chrono).
    Older-than-expected (delta>0) → blue (bad),
    Younger-than-expected (delta<0) → red (good).
    """
    if delta is None or math.isnan(delta):
        return "#FFFFFF"
    if delta > 10:
        return "#1e40af"  # very old → deep blue
    if delta > 5:
        return "#3b82f6"  # old → blue
    if delta > -5:
        return "#ffffff"  # neutral
    if delta > -10:
        return "#f97316"  # somewhat younger → orange-red
    return "#ef4444"      # much younger → strong red


def render_big_metric(label: str, value_str: str, color: str, help_text: str = ""):
    """Metric card with colored value text."""
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
    Polar maps for GCIPL sectors (left) and RNFL clock-hours (right).

    - Orientation for OD:
        * Temporal retina is on the LEFT, nasal on the RIGHT.
        * 12H is superior, 6H is inferior (bottom).
    - Color:
        * Red: more resilient / protective (contributes to younger optic nerve age)
        * Blue: more vulnerable / harmful (contributes to older optic nerve age)
    """
    # Flip sign so that protective (age-decreasing) regions appear red
    rnfl_vals = -rnfl_shap
    gc_vals = -gcipl_shap

    vmax = max(np.max(np.abs(rnfl_vals)), np.max(np.abs(gc_vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.coolwarm_r  # high=red, low=blue

    fig = plt.figure(figsize=(10, 4))
    fig.patch.set_facecolor("#111827")

    # ---------- Left: GCIPL sectors ----------
    ax1 = fig.add_subplot(1, 3, 1, polar=True)
    ax1.set_facecolor("#111827")

    num_gc = 6
    theta_gc = np.linspace(0, 2 * pi, num_gc + 1)
    width_gc = 2 * pi / num_gc

    for i in range(num_gc):
        val = gc_vals[i]
        start = theta_gc[i]
        ax1.bar(
            start,
            1.0,
            width=width_gc,
            bottom=0.0,
            color=cmap(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    gc_labels = ["NasInf", "NasSup", "Sup", "TempSup", "TempInf", "Inf"]
    ax1.set_xticks(theta_gc[:-1] + width_gc / 2)
    ax1.set_xticklabels(gc_labels, color="white", fontsize=9)
    ax1.set_yticklabels([])
    ax1.set_title("GCIPL sector\ncontribution (OD)", color="white", fontsize=12, pad=10)

    # ---------- Middle: color bar ----------
    cax = fig.add_subplot(1, 3, 2)
    cax.set_visible(False)
    cb_ax = fig.add_axes([0.44, 0.18, 0.03, 0.64])  # [left, bottom, width, height]
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cb_ax,
    )
    cb_ax.set_ylabel("SHAP value\n(impact on optic nerve age)", color="white", fontsize=9)
    cb_ax.yaxis.set_tick_params(color="white")
    plt.setp(cb_ax.get_yticklabels(), color="white", fontsize=8)
    # Top / bottom labels
    cb_ax.text(
        0.5,
        1.05,
        "More resilient (healthier / thicker)",
        color="white",
        fontsize=8,
        ha="center",
        transform=cb_ax.transAxes,
    )
    cb_ax.text(
        0.5,
        -0.12,
        "More vulnerable (older / thinner)",
        color="white",
        fontsize=8,
        ha="center",
        transform=cb_ax.transAxes,
    )

    # ---------- Right: RNFL clock-hour ----------
    ax2 = fig.add_subplot(1, 3, 3, polar=True)
    ax2.set_facecolor("#111827")

    num = 12
    theta = np.linspace(0, 2 * pi, num + 1)
    width = 2 * pi / num

    for i in range(num):
        val = rnfl_vals[i]
        start = theta[i]
        ax2.bar(
            start,
            1.0,
            width=width,
            bottom=0.0,
            color=cmap(norm(val)),
            edgecolor="#111827",
            linewidth=1.0,
            align="edge",
        )

    # Labels arranged so that 12H is superior, 6H inferior
    hour_labels = ["12H", "1H", "2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "10H", "11H"]
    ax2.set_xticks(theta[:-1] + width / 2)
    ax2.set_xticklabels(hour_labels, color="white", fontsize=9)
    ax2.set_yticklabels([])
    ax2.set_title("RNFL clock-hour\ncontribution (OD)", color="white", fontsize=12, pad=10)

    plt.tight_layout()
    return fig


def get_age_bin(age: int) -> str | None:
    if 40 <= age < 50:
        return "40s"
    if 50 <= age < 60:
        return "50s"
    if 60 <= age < 70:
        return "60s"
    if 70 <= age < 80:
        return "70s"
    return None


# -----------------------------
# 3. Main UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Optic Age — Optic Nerve Biological Aging Calculator",
        page_icon="🧠",
        layout="wide",
    )
    plt.style.use("dark_background")

    # ---- Global CSS (button styling) ----
    st.markdown(
        """
        <style>
        /* Yellow primary button */
        div.stButton > button {
            background-color: #facc15 !important;
            color: #000000 !important;
            border-radius: 999px !important;
            border: 2px solid #e5e7eb !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.8rem !important;
            font-size: 1.05rem !important;
        }
        div.stButton > button:hover {
            background-color: #eab308 !important;
            border-color: #facc15 !important;
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----- Header -----
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

    # Load models + norms
    with st.spinner("Loading age-stratified models and normative parameters..."):
        models_by_age, features_by_age, age_adjust_params = load_models_and_features()
        rnfl_mean, rnfl_sd, onas_sigma = load_normative_params()
    st.success("Models and normative parameters loaded successfully.")

    # =============================
    # 1. Patient input
    # =============================
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
            # label without code
            al_r = st.number_input("Axial length (mm)", 20.0, 30.0, 24.0, step=0.1)
        with c4:
            disc_area = st.number_input("Disc area (mm²)", 1.0, 4.0, 2.0, step=0.1)

        c5, c6, c7 = st.columns(3)
        with c5:
            vcd = st.number_input("Vertical C/D ratio", 0.0, 1.0, 0.4, step=0.01)
        with c6:
            rimarea = st.number_input("Rim area (mm²)", 0.5, 3.0, 1.5, step=0.05)
        with c7:
            cup_vol = st.number_input("Cup volume (mm³)", 0.0, 1.0, 0.2, step=0.01)

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

        st.markdown("#### GCIPL sectors (µm)")
        g1, g2, g3 = st.columns(3)
        g4, g5, g6 = st.columns(3)
        gc_tempsup = g1.number_input("TempSup", 40.0, 120.0, 80.0, step=0.5)
        gc_sup = g2.number_input("Sup", 40.0, 120.0, 80.0, step=0.5)
        gc_nassup = g3.number_input("NasSup", 40.0, 120.0, 80.0, step=0.5)
        gc_nasinf = g4.number_input("NasInf", 40.0, 120.0, 80.0, step=0.5)
        gc_inf = g5.number_input("Inf", 40.0, 120.0, 80.0, step=0.5)
        gc_tempinf = g6.number_input("TempInf", 40.0, 120.0, 80.0, step=0.5)

        st.markdown("#### Advanced OCT parameters (optional)")
        st.caption(
            "Default values approximate a healthy, super-normal eye. "
            "If detailed OCT metrics are available, entering them may improve precision."
        )

        with st.expander("Show / hide advanced parameters"):
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

        run_clicked = st.form_submit_button("Run Optic Age™ model")

    if not run_clicked:
        st.stop()

    # Age check
    age_bin = get_age_bin(age)
    if age_bin is None or age_bin not in models_by_age:
        st.error("This app currently supports only patients between 40 and 79 years of age.")
        st.stop()

    model = models_by_age[age_bin]
    feat_list = features_by_age[age_bin]

    # =============================
    # 2. Derived features (age-adjusted)
    # =============================
    rnfl12_mean = float(np.mean(rnfl_vals))
    gc6_mean = float(np.mean([gc_tempsup, gc_sup, gc_nassup, gc_nasinf, gc_inf, gc_tempinf]))

    rnfl_params = age_adjust_params["rnfl12"]
    gc6_params = age_adjust_params["gc6"]

    rnfl_pred = rnfl_params["a"] + rnfl_params["b"] * age
    gc6_pred = gc6_params["a"] + gc6_params["b"] * age

    rnfl12_resid = rnfl12_mean - rnfl_pred
    gc6_resid = gc6_mean - gc6_pred

    # =============================
    # 3. Build patient feature dict
    # =============================
    patient = {}

    # core variables
    patient["sex"] = sex
    patient["AL_R"] = al_r
    patient["DISCAREA_OD"] = disc_area
    patient["VERTICAL_CD_RATIO_OD"] = vcd
    patient["RIMAREA_OD"] = rimarea
    patient["CUPVOLUME_OD"] = cup_vol

    for i, val in enumerate(rnfl_vals, start=1):
        patient[f"CLOCKHOUR_{i}_OD"] = val

    patient["GC_TEMPSUP_OD"] = gc_tempsup
    patient["GC_SUP_OD"] = gc_sup
    patient["GC_NASSUP_OD"] = gc_nassup
    patient["GC_NASINF_OD"] = gc_nasinf
    patient["GC_INF_OD"] = gc_inf
    patient["GC_TEMPINF_OD"] = gc_tempinf

    # advanced
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

    # age-adjusted residuals
    patient["RNFL12_MEAN"] = rnfl12_mean
    patient["RNFL12_RESID"] = rnfl12_resid
    patient["GC6_MEAN"] = gc6_mean
    patient["GC6_RESID"] = gc6_resid

    # Build vector in the correct feature order
    x_vec = np.array([[patient[f] for f in feat_list]], dtype=float)

    # =============================
    # 4. Predict & scores
    # =============================
    predicted_age = float(model.predict(x_vec)[0])
    delta_age = predicted_age - age

    onas_percentile, onas_z = compute_onas(delta_age, onas_sigma)
    rnfl_z, rnfl_pct = compute_rnfl_z_and_percentile(rnfl12_mean, rnfl_mean, rnfl_sd)

    # =============================
    # 5. Optic nerve age & ONAS
    # =============================
    st.markdown("### 2. Optic Nerve Age & Optic Nerve Aging Score (ONAS)")

    cA, cB, cC = st.columns(3)

    colA_color = score_color_delta(delta_age)
    with cA:
        render_big_metric(
            "Predicted optic nerve age (years)",
            f"{predicted_age:.1f}",
            colA_color,
            help_text=f"Age-bin model used: {age_bin} (KNHANES 40–79y super-normal eyes).",
        )

    colB_color = score_color_delta(delta_age)
    with cB:
        render_big_metric(
            "ΔAge (optic − chronological, years)",
            f"{delta_age:+.1f}",
            colB_color,
            help_text="Positive: optic nerve appears older than age. "
            "Negative: younger / more resilient optic nerve.",
        )

    onas_color = score_color_percentile(onas_percentile)
    with cC:
        render_big_metric(
            "Optic Nerve Aging Score percentile (ONAS)",
            f"{onas_percentile:.1f} %",
            onas_color,
            help_text="Lower percentile → more vulnerable optic nerve; "
            "higher percentile → stronger / more resilient.",
        )

    st.markdown(
        """
        **ONAS interpretation**  
        • Lower ONAS percentile → more accelerated aging / higher structural vulnerability.  
        • Higher ONAS percentile → slower aging / more resilient optic nerve.
        """
    )

    # =============================
    # 6. RNFL normative position
    # =============================
    st.markdown("### 3. RNFL Normative Position (KNHANES-based, 40–79y)")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_big_metric(
            "Average RNFL thickness (µm)",
            f"{rnfl12_mean:.1f}",
            "#e5e7eb",
            help_text="Arithmetic mean of CLOCKHOUR 1–12 (OD).",
        )

    z_color = score_color_delta(-rnfl_z)  # thinner (negative z) → blue
    with c2:
        render_big_metric(
            "RNFL Z-score vs super-normal",
            f"{rnfl_z:+.2f}",
            z_color,
            help_text="Negative Z-score indicates thinner-than-average RNFL "
            "among 40–79y KNHANES super-normal eyes.",
        )

    rnfl_pct_color = score_color_percentile(rnfl_pct)
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

    # =============================
    # 7. Regional vulnerability map
    # =============================
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
            "Red regions contribute to a healthier / more resilient optic nerve (younger predicted age), "
            "while blue regions contribute to a more vulnerable / older optic nerve. "
            "Orientation is for the right eye (OD): temporal retina is on the left, nasal on the right."
        )
    except Exception as e:
        st.warning(
            "SHAP-based regional vulnerability map could not be generated "
            f"(missing `shap` package or other error: {e}). "
            "The main Optic Age™ scores above remain valid."
        )


if __name__ == "__main__":
    main()

