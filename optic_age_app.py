"""
Optic Age™ — Optic Nerve Biological Aging Calculator
Streamlit App

Requirements (your venv):
    pip install streamlit pandas numpy matplotlib shap joblib pillow xgboost

Expected local files in the same folder:
    - optic_age_model_tuned.pkl   (tuned official OD-only model)
    - optic_age_features.pkl      (list of feature column names)
    - master_2019_2021.csv        (KNHANES 2019–2021 master file)
    - optic_age_icon.png          (optional, App Store style icon)
"""

import math

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image

# ============================================================
# 0. PAGE CONFIG & GLOBAL STYLE (DARK THEME)
# ============================================================

# Load app icon if exists
try:
    APP_ICON = Image.open("optic_age_icon.png")
except Exception:
    APP_ICON = None

if APP_ICON is not None:
    st.set_page_config(
        page_title="Optic Age™ — Optic Nerve Biological Aging Calculator",
        page_icon=APP_ICON,
        layout="wide",
    )
else:
    st.set_page_config(
        page_title="Optic Age™ — Optic Nerve Biological Aging Calculator",
        layout="wide",
    )

# --- Dark background + font 색 조정 ---
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
    }
    .stApp {
        background-color: #000000;
        color: #f5f5f5;
    }
    div.block-container {
        padding-top: 1rem;
    }
    /* Remove top padding of forms etc. */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #111111;
        color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header: icon + title + author
col_icon, col_title = st.columns([1, 6])

with col_icon:
    if APP_ICON is not None:
        st.image(APP_ICON, width=90)

with col_title:
    st.markdown(
        """
        # **Optic Age™ — Optic Nerve Biological Aging Calculator**
        **Developed by Professor Young Kook Kim, Seoul National University Hospital (SNUH)**
        """
    )

st.caption(
    "A machine-learning–based clinical decision support tool estimating the biological aging "
    "of the optic nerve using OCT metrics from KNHANES 2019–2021."
)

# ============================================================
# 1. CONFIG PATHS
# ============================================================

MODEL_PATH = "optic_age_model_tuned.pkl"   # tuned official model (OD only)
FEATURES_PATH = "optic_age_features.pkl"   # feature list used in training
MASTER_PATH = "master_2019_2021.csv"       # KNHANES 2019–2021 master CSV


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=True)
def load_model_and_features():
    """Load tuned optic nerve age model and list of features."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols


@st.cache_resource(show_spinner=True)
def load_supernormal_and_normative(feature_cols):
    """
    1) Apply 'super-normal' criteria:
       - age >= 20
       - E_GR != 1 (no glaucoma by KNHANES grading)
       - E_DM, E_AMD, E_EM, E_MH, E_RVO != 1 (no major retinal disease)
       - E_Pr_1 <= 21 mmHg (or missing)
       - AL_R <= 26.0 mm
       - RNFL_AVERAGE_OD >= 70 µm
    2) Keep complete cases for feature_cols + age + wt_itvex
    3) Compute 3-year combined weights (wt_itvex / 3)
    4) Compute weighted mean/SD of RNFL_AVERAGE_OD (norm database)
    """

    df = pd.read_csv(MASTER_PATH, low_memory=False)

    extra_cols = [
        "E_Pr_1",
        "age",
        "wt_itvex",
        "E_GR",
        "E_DM",
        "E_AMD",
        "E_EM",
        "E_MH",
        "E_RVO",
        "RNFL_AVERAGE_OD",
        "AL_R",
    ]

    needed_cols = list(set(feature_cols + extra_cols))
    needed_cols = [c for c in needed_cols if c in df.columns]
    df_sub = df[needed_cols].copy()

    # handle missing codes for IOP, RNFL, AL
    if "E_Pr_1" in df_sub.columns:
        df_sub["E_Pr_1"] = df_sub["E_Pr_1"].replace(
            {
                888: np.nan, 888.8: np.nan, 888.88: np.nan,
                999: np.nan, 999.8: np.nan, 999.99: np.nan,
                8888: np.nan, 9999: np.nan,
            }
        )

    if "AL_R" in df_sub.columns:
        df_sub.loc[df_sub["AL_R"] > 40, "AL_R"] = np.nan

    if "RNFL_AVERAGE_OD" in df_sub.columns:
        df_sub["RNFL_AVERAGE_OD"] = df_sub["RNFL_AVERAGE_OD"].replace(
            {
                888: np.nan, 999: np.nan,
                888.8: np.nan, 888.88: np.nan,
                999.8: np.nan, 999.99: np.nan,
                8888: np.nan, 9999: np.nan,
            }
        )
        df_sub.loc[df_sub["RNFL_AVERAGE_OD"] >= 900, "RNFL_AVERAGE_OD"] = np.nan

    for col in ["E_GR", "E_DM", "E_AMD", "E_EM", "E_MH", "E_RVO"]:
        if col in df_sub.columns:
            df_sub.loc[~df_sub[col].isin([0, 1]), col] = np.nan

    crit = (
        (df_sub["age"] >= 20)
        & (df_sub["E_GR"] != 1)
        & (df_sub["E_DM"] != 1)
        & (df_sub["E_AMD"] != 1)
        & (df_sub["E_EM"] != 1)
        & (df_sub["E_MH"] != 1)
        & (df_sub["E_RVO"] != 1)
        & ((df_sub["E_Pr_1"].isna()) | (df_sub["E_Pr_1"] <= 21))
        & (df_sub["AL_R"] <= 26.0)
    )

    if "RNFL_AVERAGE_OD" in df_sub.columns:
        crit = crit & (df_sub["RNFL_AVERAGE_OD"] >= 70)

    super_normal = df_sub[crit].copy()
    required_cols = list(feature_cols) + ["age", "wt_itvex"]
    super_normal = super_normal.dropna(subset=required_cols)

    # 3년 통합 weight
    super_normal["wt3"] = super_normal["wt_itvex"] / 3.0

    # RNFL normative mean/SD
    if "RNFL_AVERAGE_OD" in super_normal.columns:
        w = super_normal["wt3"].values
        x = super_normal["RNFL_AVERAGE_OD"].values
        w_sum = np.sum(w)
        rnfl_mean = float(np.sum(w * x) / w_sum)
        rnfl_var = float(np.sum(w * (x - rnfl_mean) ** 2) / w_sum)
        rnfl_sd = float(np.sqrt(rnfl_var))
    else:
        rnfl_mean, rnfl_sd = None, None

    return super_normal, rnfl_mean, rnfl_sd


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_rnfl_z_and_percentile(rnfl_value, rnfl_mean, rnfl_sd):
    if rnfl_mean is None or rnfl_sd is None or rnfl_sd <= 0 or rnfl_value is None:
        return None, None
    z = (rnfl_value - rnfl_mean) / rnfl_sd
    pct = normal_cdf(z) * 100.0
    return float(z), float(pct)


def compute_onas(pred_age, chrono_age, resid_mean, resid_sd):
    """
    ONAS:
      1) ΔAge = Predicted optic nerve age − chronological age
      2) raw_z = (ΔAge − resid_mean) / resid_sd
      3) ONAS_percentile = (1 − Φ(raw_z)) × 100
         → Lower percentile = more accelerated aging (worse / more vulnerable)
         → Higher percentile = younger / more resilient optic nerve
    """
    delta = pred_age - chrono_age
    if resid_sd is None or resid_sd <= 0:
        return delta, None, None, "Not available"

    raw_z = (delta - resid_mean) / resid_sd
    onas_pct = (1.0 - normal_cdf(raw_z)) * 100.0

    if onas_pct < 20:
        cat = "Markedly accelerated aging"
    elif onas_pct < 40:
        cat = "Mildly accelerated aging"
    elif onas_pct < 70:
        cat = "Age-appropriate optic nerve"
    else:
        cat = "Younger-than-average / robust optic nerve"

    return float(delta), float(onas_pct), float(raw_z), cat


@st.cache_resource(show_spinner=True)
def build_shap_explainer(_model):
    return shap.TreeExplainer(_model)


def clamp(value, vmin=-1.0, vmax=1.0):
    return max(vmin, min(vmax, value))


def rwb_color_from_score(score: float) -> str:
    """
    score ∈ [-1,1]
      -1 → red (worst)
       0 → white
      +1 → blue (best)
    """
    s = clamp(score, -1.0, 1.0)
    white = (255, 255, 255)
    red = (220, 50, 50)
    blue = (80, 140, 255)

    if s >= 0:
        r = int(white[0] + (blue[0] - white[0]) * s)
        g = int(white[1] + (blue[1] - white[1]) * s)
        b = int(white[2] + (blue[2] - white[2]) * s)
    else:
        t = -s
        r = int(red[0] + (white[0] - red[0]) * t)
        g = int(red[1] + (white[1] - red[1]) * t)
        b = int(red[2] + (white[2] - red[2]) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


def colored_metric(label: str, value_str: str, color: str):
    """
    Dark card + 컬러 숫자
    """
    st.markdown(
        f"""
        <div style="
            padding:0.7rem 0.9rem;
            border-radius:0.9rem;
            border:1px solid #444444;
            background-color:#111111;
            ">
          <div style="font-size:0.8rem;color:#dddddd;margin-bottom:0.2rem;">
            {label}
          </div>
          <div style="font-size:1.6rem;font-weight:700;color:{color};">
            {value_str}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gradient_bar(label_left: str, label_mid: str, label_right: str):
    """
    Red–white–blue horizontal bar 설명용.
    """
    st.markdown(
        f"""
        <div style="
            margin-top:0.5rem;
            margin-bottom:0.5rem;
            ">
          <div style="
              height:18px;
              border-radius:10px;
              border:1px solid #555555;
              background: linear-gradient(90deg,
                  #dc3232 0%, #ffffff 50%, #508cff 100%);
              ">
          </div>
          <div style="display:flex;
              justify-content:space-between;
              font-size:0.75rem;
              color:#cccccc;
              margin-top:0.25rem;">
            <span>{label_left}</span>
            <span>{label_mid}</span>
            <span>{label_right}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_rnfl_gc_figure(rnfl_contrib, gc_contrib):
    """
    Circular maps:
      - Left  : macular GCIPL (right eye, temporal = 화면 좌측)
      - Right : optic disc RNFL clock-hour (right eye)
    """
    rnfl_labels = list(rnfl_contrib.keys())
    rnfl_vals = np.array(list(rnfl_contrib.values()))

    gc_labels = list(gc_contrib.keys())
    gc_vals = np.array(list(gc_contrib.values()))

    max_abs = max(np.max(np.abs(rnfl_vals)), np.max(np.abs(gc_vals)))
    if max_abs == 0:
        max_abs = 1.0

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-max_abs, vmax=max_abs)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.08, 1])

    # ----- GCIPL (macula, left; temporal = left, nasal = right) -----
    ax_gc = fig.add_subplot(gs[0, 0], polar=True)
    m = len(gc_labels)
    theta_g = np.linspace(0, 2 * np.pi, m, endpoint=False)
    width_g = 2 * np.pi / m

    # 0 rad at West(좌측) so temporal sectors appear on the left side
    ax_gc.set_theta_zero_location("W")
    ax_gc.set_theta_direction(-1)

    colors_gc = cmap(norm(gc_vals))
    ax_gc.bar(
        theta_g,
        np.ones(m),
        width=width_g,
        color=colors_gc,
        edgecolor="k",
        linewidth=0.5,
    )

    ax_gc.set_xticks(theta_g)
    ax_gc.set_xticklabels(gc_labels, fontsize=9)
    ax_gc.set_yticklabels([])
    ax_gc.set_title("GCIPL sector contribution (macula, OD)", fontsize=11, color="w")

    # ----- RNFL 12 clock-hour (disc, right) -----
    ax_rnfl = fig.add_subplot(gs[0, 2], polar=True)
    n = len(rnfl_labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n

    ax_rnfl.set_theta_zero_location("N")  # 12H at top
    ax_rnfl.set_theta_direction(-1)       # clockwise

    colors_rnfl = cmap(norm(rnfl_vals))
    ax_rnfl.bar(
        theta,
        np.ones(n),
        width=width,
        color=colors_rnfl,
        edgecolor="k",
        linewidth=0.5,
    )

    ax_rnfl.set_xticks(theta)
    ax_rnfl.set_xticklabels(rnfl_labels, fontsize=9)
    ax_rnfl.set_yticklabels([])
    ax_rnfl.set_title("RNFL clock-hour contribution (optic disc, OD)", fontsize=11, color="w")

    # ----- Color bar between the two -----
    ax_cbar = fig.add_subplot(gs[0, 1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_ticks([-max_abs, 0.0, max_abs])
    cbar.set_ticklabels(
        ["Younger / thicker\n(protective)",
         "Neutral",
         "Older / thinner\n(vulnerable)"]
    )
    cbar.ax.tick_params(labelsize=7, color="w")
    cbar.ax.yaxis.set_tick_params(color="w")
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("w")
    cbar.set_label("SHAP value\n(impact on optic nerve age)", fontsize=8, color="w")

    fig.patch.set_facecolor("#000000")
    for ax in [ax_gc, ax_rnfl]:
        ax.set_facecolor("#000000")
        for spine in ax.spines.values():
            spine.set_edgecolor("w")
        ax.tick_params(colors="w")

    fig.tight_layout()
    return fig


# ============================================================
# 3. LOAD MODEL & NORMATIVE DB
# ============================================================

with st.spinner("Loading model and KNHANES-derived normative database..."):
    model, feature_cols = load_model_and_features()
    super_normal_df, rnfl_mean, rnfl_sd = load_supernormal_and_normative(feature_cols)

    X_super = super_normal_df[feature_cols].values
    y_super = super_normal_df["age"].values
    w_super = super_normal_df["wt3"].values

    y_hat_super = model.predict(X_super)
    resid = y_hat_super - y_super

    w_sum = np.sum(w_super)
    resid_mean = float(np.sum(w_super * resid) / w_sum)
    resid_var = float(np.sum(w_super * (resid - resid_mean) ** 2) / w_sum)
    resid_sd = float(np.sqrt(resid_var))

    shap_explainer = build_shap_explainer(model)

st.success("Model and KNHANES-derived norms loaded successfully.")

# ============================================================
# 4. INPUT FORM
# ============================================================

st.subheader("1. Input Patient Data (Right Eye Only)")

with st.form("input_form"):
    col_left, col_right = st.columns(2)

    # -------- Basic info & global OCT --------
    with col_left:
        chrono_age = st.number_input(
            "Chronological age (years)",
            min_value=20,
            max_value=100,
            value=50,
        )

        sex_label = st.radio(
            "Sex", options=["Male", "Female"], index=0, horizontal=True
        )
        sex_code = 1 if sex_label == "Male" else 2

        al_r = st.number_input(
            "Axial length AL_R (mm)",
            min_value=20.0,
            max_value=30.0,
            value=24.0,
            step=0.1,
        )

        disc_area = st.number_input(
            "Disc area (DISCAREA_OD, mm²)",
            min_value=0.5,
            max_value=4.0,
            value=2.0,
            step=0.1,
        )

        v_cdr = st.number_input(
            "Vertical C/D ratio (VERTICAL_CD_RATIO_OD)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )

        rnfl_avg = st.number_input(
            "Global RNFL thickness (RNFL_AVERAGE_OD, µm)",
            min_value=40.0,
            max_value=150.0,
            value=95.0,
            step=0.5,
        )

        gc_avg = st.number_input(
            "Global GCIPL thickness (GC_AVERAGE_OD, µm)",
            min_value=40.0,
            max_value=120.0,
            value=80.0,
            step=0.5,
        )

        gc_min = st.number_input(
            "GCIPL minimum thickness (GC_MINIMUM_OD, µm)",
            min_value=30.0,
            max_value=110.0,
            value=70.0,
            step=0.5,
        )

    # -------- RNFL clock-hour & GCIPL sectors --------
    with col_right:
        st.markdown("**RNFL clock-hour thickness (OD)**")
        cols_clock = st.columns(4)
        clock_inputs = {}
        default_clock = [95, 100, 100, 105, 110, 110, 95, 85, 80, 75, 80, 90]

        for i in range(12):
            with cols_clock[i % 4]:
                label = f"{i+1}H (CLOCKHOUR_{i+1}_OD)"
                clock_inputs[i + 1] = st.number_input(
                    label,
                    min_value=30.0,
                    max_value=200.0,
                    value=float(default_clock[i]),
                    step=0.5,
                )

        st.markdown("**GCIPL sector thickness (OD)**")
        gc_temp_sup = st.number_input(
            "GC_TEMPSUP_OD (µm)",
            min_value=40.0,
            max_value=120.0,
            value=80.0,
            step=0.5,
        )
        gc_sup = st.number_input(
            "GC_SUP_OD (µm)",
            min_value=40.0,
            max_value=120.0,
            value=82.0,
            step=0.5,
        )
        gc_nas_sup = st.number_input(
            "GC_NASSUP_OD (µm)",
            min_value=40.0,
            max_value=120.0,
            value=81.0,
            step=0.5,
        )
        gc_nas_inf = st.number_input(
            "GC_NASINF_OD (µm)",
            min_value=40.0,
            max_value=120.0,
            value=79.0,
            step=0.5,
        )
        gc_inf = st.number_input(
            "GC_INF_OD (µm)",
            min_value=40.0,
            max_value=120.0,
            value=78.0,
            step=0.5,
        )
        gc_temp_inf = st.number_input(
            "GC_TEMPINF_OD (µm)",
            min_value=40.0,
            max_value=120.0,
            value=77.0,
            step=0.5,
        )

    submitted = st.form_submit_button("Run Optic Age™ Analysis")

# ============================================================
# 5. PREDICTION & VISUALIZATION
# ============================================================

if submitted:
    # ----- 5.1 assemble patient row -----
    patient_dict = {}

    for i in range(1, 13):
        colname = f"CLOCKHOUR_{i}_OD"
        if colname in feature_cols:
            patient_dict[colname] = float(clock_inputs[i])

    mapping_gc = {
        "GC_TEMPSUP_OD": gc_temp_sup,
        "GC_SUP_OD": gc_sup,
        "GC_NASSUP_OD": gc_nas_sup,
        "GC_NASINF_OD": gc_nas_inf,
        "GC_INF_OD": gc_inf,
        "GC_TEMPINF_OD": gc_temp_inf,
        "GC_MINIMUM_OD": gc_min,
        "RNFL_AVERAGE_OD": rnfl_avg,
        "GC_AVERAGE_OD": gc_avg,
        "DISCAREA_OD": disc_area,
        "VERTICAL_CD_RATIO_OD": v_cdr,
        "AL_R": al_r,
        "sex": sex_code,
    }
    for col, val in mapping_gc.items():
        if col in feature_cols:
            patient_dict[col] = float(val)

    for col in feature_cols:
        if col not in patient_dict:
            patient_dict[col] = np.nan

    patient_df = pd.DataFrame([[patient_dict[c] for c in feature_cols]],
                              columns=feature_cols)

    # ----- 5.2 predictions & scores -----
    pred_age = float(model.predict(patient_df.values)[0])

    rnfl_z, rnfl_pct = compute_rnfl_z_and_percentile(rnfl_avg, rnfl_mean, rnfl_sd)
    delta_age, onas_pct, onas_z, onas_cat = compute_onas(
        pred_age, chrono_age, resid_mean, resid_sd
    )

    shap_values = shap_explainer(patient_df)
    shap_array = shap_values.values[0]
    feature_to_shap = {f: v for f, v in zip(feature_cols, shap_array)}

    rnfl_contrib = {
        f"{i}H": feature_to_shap.get(f"CLOCKHOUR_{i}_OD", 0.0)
        for i in range(1, 13)
    }

    gc_cols = [
        "GC_TEMPSUP_OD",
        "GC_SUP_OD",
        "GC_NASSUP_OD",
        "GC_NASINF_OD",
        "GC_INF_OD",
        "GC_TEMPINF_OD",
    ]
    gc_label_map = {
        "GC_TEMPSUP_OD": "TempSup",
        "GC_SUP_OD": "Sup",
        "GC_NASSUP_OD": "NasSup",
        "GC_NASINF_OD": "NasInf",
        "GC_INF_OD": "Inf",
        "GC_TEMPINF_OD": "TempInf",
    }
    gc_contrib = {gc_label_map[c]: feature_to_shap.get(c, 0.0) for c in gc_cols}

    # ----- 5.3 color scores -----
    goodness_delta = -delta_age / 10.0          # ΔAge ↑ → worse
    goodness_onas = (onas_pct - 50.0) / 50.0 if onas_pct is not None else 0.0
    goodness_pred = goodness_delta
    goodness_rnfl_z = rnfl_z / 2.0 if rnfl_z is not None else 0.0
    goodness_rnfl_pct = (rnfl_pct - 50.0) / 50.0 if rnfl_pct is not None else 0.0

    # ========================================================
    # 2. Optic Nerve Age & ONAS
    # ========================================================
    st.subheader("2. Optic Nerve Age & Optic Nerve Aging Score (ONAS)")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        color_pred = rwb_color_from_score(goodness_pred)
        colored_metric(
            "Predicted optic nerve age (years)",
            f"{pred_age:.1f}",
            color_pred,
        )

        color_delta = rwb_color_from_score(goodness_delta)
        colored_metric(
            "ΔAge (optic − chronological, years)",
            f"{delta_age:+.1f}",
            color_delta,
        )

    with col_b:
        if onas_pct is not None:
            color_onas = rwb_color_from_score(goodness_onas)
            colored_metric(
                "Optic Nerve Aging Score (ONAS) percentile\n(higher = stronger / more resilient)",
                f"{onas_pct:.1f} %",
                color_onas,
            )
        else:
            colored_metric(
                "Optic Nerve Aging Score (ONAS) percentile",
                "N/A",
                "#cccccc",
            )

    with col_c:
        if rnfl_z is not None:
            color_rnfl_z = rwb_color_from_score(goodness_rnfl_z)
            colored_metric(
                "RNFL Z-score vs super-normal",
                f"{rnfl_z:.2f}",
                color_rnfl_z,
            )

            color_rnfl_pct = rwb_color_from_score(goodness_rnfl_pct)
            colored_metric(
                "RNFL thickness percentile (lower = thinner / more vulnerable)",
                f"{rnfl_pct:.1f} %",
                color_rnfl_pct,
            )
        else:
            colored_metric("RNFL Z-score vs super-normal", "N/A", "#cccccc")
            colored_metric("RNFL thickness percentile", "N/A", "#cccccc")

    # horizontal legend for section 2
    gradient_bar(
        "Weaker / older optic nerve",
        "Average",
        "Stronger / younger optic nerve",
    )

    if onas_pct is not None:
        worse_than = 100.0 - onas_pct
        st.write(f"**ONAS category:** {onas_cat}")
        st.caption(
            f"ONAS percentile is calculated from KNHANES super-normal eyes. "
            f"A value of **{onas_pct:.1f}%** means this optic nerve appears more aged / weaker "
            f"than about **{worse_than:.1f}%** of the reference population. "
            f"**Lower ONAS percentile indicates a weaker / more vulnerable optic nerve**, "
            f"while higher percentile indicates a stronger and more resilient optic nerve."
        )
    else:
        st.write("**ONAS category:** Not available (residual SD invalid).")

    st.markdown("---")

    # ========================================================
    # 3. RNFL Normative Position
    # ========================================================
    st.subheader("3. RNFL Normative Position (KNHANES-based)")

    if rnfl_z is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            colored_metric("RNFL_AVERAGE_OD (µm)", f"{rnfl_avg:.1f}",
                           rwb_color_from_score(goodness_rnfl_pct))
        with col2:
            colored_metric("RNFL Z-score vs super-normal",
                           f"{rnfl_z:.2f}",
                           rwb_color_from_score(goodness_rnfl_z))
        with col3:
            colored_metric(
                "RNFL thickness percentile (lower = thinner / more vulnerable)",
                f"{rnfl_pct:.1f} %",
                rwb_color_from_score(goodness_rnfl_pct),
            )

        gradient_bar(
            "Thinner / weaker RNFL",
            "Average thickness",
            "Thicker / more robust RNFL",
        )

        thinner_than = 100.0 - rnfl_pct
        st.caption(
            f"Percentile is based on KNHANES super-normal eyes "
            f"(OD, age ≥20, no glaucoma or major retinal disease, IOP ≤21 mmHg, "
            f"AL_R ≤26.0 mm, RNFL_AVERAGE_OD ≥70 µm). "
            f"A value of **{rnfl_pct:.1f}%** means this RNFL is thinner than about "
            f"**{thinner_than:.1f}%** of the reference population. "
            f"**Lower percentile indicates thinner / weaker RNFL and more advanced structural loss.**"
        )
    else:
        st.write("RNFL normative statistics are not available for this input.")

    st.markdown("---")

    # ========================================================
    # 4. Regional Vulnerability Map
    # ========================================================
    st.subheader("4. Regional Vulnerability Map (Feature Importance)")

    fig = make_rnfl_gc_figure(rnfl_contrib, gc_contrib)
    st.pyplot(fig, use_container_width=True)

    worst_rnfl = max(rnfl_contrib, key=lambda k: rnfl_contrib[k])
    worst_gc = max(gc_contrib, key=lambda k: gc_contrib[k])

    st.markdown(
        f"- **Macula (left circle, GCIPL):** The reddest sector (highest contribution to aging) is **{worst_gc}**.\n"
        f"- **Optic disc (right circle, RNFL):** The darkest red clock-hour is **{worst_rnfl}H**, "
        f"indicating the region that contributes most to increasing the optic nerve age."
    )
    st.caption(
        "For this **right eye**, the **left circular map** represents macular GCIPL sectors "
        "(temporal on the left side, nasal on the right), and the **right circular map** represents "
        "optic disc RNFL clock-hour sectors. Red sectors make the optic nerve appear **older / thinner / more vulnerable**, "
        "while blue sectors make it appear **younger / thicker / structurally preserved**."
    )
