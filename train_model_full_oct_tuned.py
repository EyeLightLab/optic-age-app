
# train_model_full_oct_tuned.py
# ---------------------------------------------------------
# Optic Age™ 시신경 나이 모델 (풀 OCT + sex, age-bin 별 aggressive tuning)
#
# - 데이터 파일: master_2019_2021.csv  (같은 폴더에 있어야 합니다)
# - 연령: 40–79세, AL_R ≤ 26mm, OCT 결측(888/999/NaN) 제거
# - 추가 필터:
#     * RNFL 12H mean:   60–150 µm
#     * GCIPL 6-sector:  50–110 µm
# - 제외 파라미터:
#     * ACD, CCT, RNFL_AVERAGE, GCIPL_AVG, GCIPL_MIN (있어도 사용하지 않음)
# - 출력:
#     * optic_age_models_by_age.pkl   (40s/50s/60s/70s 모델)
#     * optic_age_features_by_age.pkl (각 age-bin 별 feature 리스트)
#     * age_adjust_params.pkl         (RNFL, GC age-adjust 선형식)
#     * norm_table.csv                (RNFL mean/SD, ONAS sigma)
#     * supernormal_fulloct_40_79.csv (학습에 사용된 슈퍼노멀 데이터)
#     * optic_age_results.csv         (독립 test set 40–79세 예측 결과; figure용)
# ---------------------------------------------------------

import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import xgboost as xgb

# -----------------------------
# 0. 경로 및 상수
# -----------------------------
DATA_PATH = "master_2019_2021.csv"
SUPERNORMAL_SAVE_PATH = "supernormal_fulloct_40_79.csv"

MODELS_BY_AGE_PATH = "optic_age_models_by_age.pkl"
FEATURES_BY_AGE_PATH = "optic_age_features_by_age.pkl"
AGE_ADJUST_PATH = "age_adjust_params.pkl"
NORM_TABLE_PATH = "norm_table.csv"
RESULTS_PATH = "optic_age_results.csv"

RANDOM_STATE = 42

# RNFL / GCIPL 컬럼 정의
RNFL_CLOCK_COLS = [f"CLOCKHOUR_{i}_OD" for i in range(1, 13)]
GCIPL_SECTOR_COLS = [
    "GC_TEMPSUP_OD",
    "GC_SUP_OD",
    "GC_NASSUP_OD",
    "GC_NASINF_OD",
    "GC_INF_OD",
    "GC_TEMPINF_OD",
]

# 사용 OCT feature (ACD, CCT, RNFL_AVERAGE, GCIPL_AVG, GCIPL_MIN 고의로 제외)
OCT_FEATURES_BASE = [
    # AL & 디스크 / rim / cup
    "AL_R",
    "DISCAREA_OD",
    "DISK_DIAMETER_OD",
    "VERTICAL_CD_RATIO_OD",
    "RIMAREA_OD",
    "CUPVOLUME_OD",

    # RNFL clock hours
    *RNFL_CLOCK_COLS,

    # RNFL quadrants / sectors
    "QUADRANT_T_OD",
    "QUADRANT_S_OD",
    "QUADRANT_N_OD",
    "QUADRANT_I_OD",
    "RNFL_SUP_OD",
    "RNFL_INF_OD",
    "RNFL_NASSUP_OD",
    "RNFL_NASINF_OD",
    "RNFL_TEMPSUP_OD",
    "RNFL_TEMPINF_OD",
    "RNFL_MINIMUM_OD",

    # GCIPL sectors
    *GCIPL_SECTOR_COLS,

    # Outer retina
    "OR_TEMPSUP_OD",
    "OR_SUP_OD",
    "OR_NASSUP_OD",
    "OR_NASINF_OD",
    "OR_INF_OD",
    "OR_TEMPINF_OD",
    "OR_AVERAGE_OD",
    "OR_MINIMUM_OD",

    # Thickness / volume / SFCT
    "CMT_OD",
    "AVERAGETHICKNESS_OD",
    "CUBEAVGTHICKNESS_ILMRPE_OD",
    "CUBEAVGTHICKNESS_ILMRPEFIT_OD",
    "CUBEVOLUME_ILMRPE_OD",
    "CUBEVOLUME_ILMRPEFIT_OD",
    "C_ILMRPE_OD",
    "C_ILMRPEFIT_OD",
    "SFCT_OD",

    # Macular ETDRS rings (Z_*)
    "Z_CENTER_OD",
    "Z_INNERRIGHT_OD",
    "Z_INNERSUPERIOR_OD",
    "Z_INNERLEFT_OD",
    "Z_INNERINFERIOR_OD",
    "Z_OUTERRIGHT_OD",
    "Z_OUTERSUPERIOR_OD",
    "Z_OUTERLEFT_OD",
    "Z_OUTERINFERIOR_OD",
]

# 불포함 변수 이름 (있어도 사용하지 않음)
EXCLUDED_COLUMNS = {
    "ACD_OD",
    "CCT_OD",
    "RNFL_AVERAGE_OD",
    "GC_AVERAGE_OD",
    "GC_MINIMUM_OD",
    "ACD",
    "CCT",
    "RNFL_AVERAGE",
    "GCIPL_AVG",
    "GCIPL_MIN",
}


# -----------------------------
# 1. 유틸 함수
# -----------------------------
def normal_cdf(z):
    """표준정규 CDF (vectorized)."""
    z_arr = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z_arr / math.sqrt(2.0)))


def print_count(msg, before, after):
    print(f"[INFO] {msg}: {before:,} → {after:,}명")


def safe_numeric(df, cols):
    """지정 컬럼을 숫자로 변환 (없으면 무시)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -----------------------------
# 2. XGBoost 튜닝 (aggressive, train set 내부에서만)
# -----------------------------
def tune_xgb_for_bin(X, y, n_iter=20, random_state=RANDOM_STATE):
    """
    한 age-bin에 대해 aggressive random search로 XGBRegressor 튜닝.
    여기서는 train set만 받아 내부에서 train/valid로 나누어 사용합니다.
    """
    from random import Random

    rnd = Random(random_state)

    param_space = {
        "n_estimators": [800, 1000, 1200, 1500, 1800, 2000],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0.0, 0.5, 1.0, 2.0, 5.0],
    }

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    best_model = None
    best_params = None
    best_mae = float("inf")
    best_r2 = None

    print(f"[INFO]   Random search 시작 (n_iter={n_iter})...")

    for i in range(1, n_iter + 1):
        params = {k: rnd.choice(v_list) for k, v_list in param_space.items()}

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",
            eval_metric="mae",
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=100,
        )

        pred_valid = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, pred_valid)
        r2 = r2_score(y_valid, pred_valid)

        print(
            f"  [튜닝 {i:2d}/{n_iter:2d}] "
            f"MAE(valid) = {mae:5.3f}  R² = {r2:5.3f}  params={params}"
        )

        if mae < best_mae:
            best_mae = mae
            best_r2 = r2
            best_model = model
            best_params = params

    print("[INFO]   최적 파라미터:", best_params)
    print(f"[INFO]   최적 MAE(valid) = {best_mae:.3f}, R² = {best_r2:.3f}")
    return best_model, best_params, best_mae, best_r2


# -----------------------------
# 3. 메인 파이프라인
# -----------------------------
def main():
    # -------------------------
    # 3-1. 데이터 로드
    # -------------------------
    print("[INFO] 데이터 로드 중...]")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} 파일을 찾을 수 없습니다. "
            f"train_model_full_oct_tuned.py 와 같은 폴더에 있는지 확인해 주세요."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    n0 = len(df)
    print(f"[INFO] 원본 대상자 수: {n0:,}명")

    # age / sex / AL_R 숫자 변환
    base_numeric_cols = ["age", "AL_R", "sex"]
    safe_numeric(df, base_numeric_cols)

    # -------------------------
    # 3-2. 기본 필터
    # -------------------------
    # (1) 나이 >= 20 (전체 pool)
    df = df[df["age"] >= 20]
    print_count("age >= 20세 필터 적용", n0, len(df))

    # (2) AL_R ≤ 26mm
    n_before_al = len(df)
    if "AL_R" in df.columns:
        df = df[df["AL_R"] <= 26]
    print_count("AL_R ≤ 26mm 필터", n_before_al, len(df))

    # (3) OCT 숫자 변환 및 결측/888/999 제거
    all_numeric_cols = set(OCT_FEATURES_BASE + RNFL_CLOCK_COLS + GCIPL_SECTOR_COLS)
    safe_numeric(df, all_numeric_cols)

    n_before_oct = len(df)
    for c in all_numeric_cols:
        if c in df.columns:
            df = df[(df[c] != 888) & (df[c] != 999) & (~df[c].isna())]
    print_count("OCT 결측(888/999/NaN) 제거", n_before_oct, len(df))

    # -------------------------
    # 3-3. RNFL / GCIPL mean 및 품질 필터
    # -------------------------
    # RNFL 12H mean
    rnfl_cols_in_df = [c for c in RNFL_CLOCK_COLS if c in df.columns]
    if len(rnfl_cols_in_df) < 6:
        raise ValueError(
            f"RNFL clock-hour 컬럼이 너무 적습니다: {rnfl_cols_in_df}. "
            "CLOCKHOUR_1_OD ~ CLOCKHOUR_12_OD 이 존재하는지 확인해 주세요."
        )
    df["rnfl12_mean"] = df[rnfl_cols_in_df].mean(axis=1)

    # GCIPL 6-sector mean
    gc_cols_in_df = [c for c in GCIPL_SECTOR_COLS if c in df.columns]
    if len(gc_cols_in_df) < 4:
        raise ValueError(
            f"GCIPL 6-sector 컬럼이 너무 적습니다: {gc_cols_in_df}. "
            "GC_TEMPSUP_OD ~ GC_TEMPINF_OD 이 존재하는지 확인해 주세요."
        )
    df["gc6_mean"] = df[gc_cols_in_df].mean(axis=1)

    # 품질 필터
    n_before_quality = len(df)
    df = df[
        (df["rnfl12_mean"] >= 60)
        & (df["rnfl12_mean"] <= 150)
        & (df["gc6_mean"] >= 50)
        & (df["gc6_mean"] <= 110)
    ]
    print_count(
        "RNFL 12H mean 60–150 µm & GCIPL 6-sector 50–110 µm 필터",
        n_before_quality,
        len(df),
    )

    # -------------------------
    # 3-4. 40–79세 슈퍼노멀 학습 데이터
    # -------------------------
    n_before_40_79 = len(df)
    df_train = df[(df["age"] >= 40) & (df["age"] <= 79)].copy()
    print_count("최종 학습 데이터 수 (40–79세)", n_before_40_79, len(df_train))

    # supernormal 데이터 저장
    df_train.to_csv(SUPERNORMAL_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(
        f"[INFO] super-normal 40–79세 학습 데이터 저장 완료: "
        f"{os.path.abspath(SUPERNORMAL_SAVE_PATH)}"
    )

    # -------------------------
    # 3-5. RNFL / GC age-adjust 선형식
    # -------------------------
    # RNFL vs age
    x_r = df_train["age"].values
    y_r = df_train["rnfl12_mean"].values
    mask_r = ~np.isnan(x_r) & ~np.isnan(y_r)
    coef_r = np.polyfit(x_r[mask_r], y_r[mask_r], 1)  # slope, intercept

    # GCIPL vs age
    x_g = df_train["age"].values
    y_g = df_train["gc6_mean"].values
    mask_g = ~np.isnan(x_g) & ~np.isnan(y_g)
    coef_g = np.polyfit(x_g[mask_g], y_g[mask_g], 1)

    age_adjust_params = {
        "rnfl12": {"b": float(coef_r[0]), "a": float(coef_r[1])},
        "gc6": {"b": float(coef_g[0]), "a": float(coef_g[1])},
    }
    joblib.dump(age_adjust_params, AGE_ADJUST_PATH)
    print(
        f"[INFO] age-adjustment 파라미터 저장 완료: {os.path.abspath(AGE_ADJUST_PATH)}"
    )

    # age-adjusted residuals 추가
    df_train["rnfl12_pred"] = (
        age_adjust_params["rnfl12"]["a"]
        + age_adjust_params["rnfl12"]["b"] * df_train["age"]
    )
    df_train["gc6_pred"] = (
        age_adjust_params["gc6"]["a"]
        + age_adjust_params["gc6"]["b"] * df_train["age"]
    )
    df_train["rnfl12_resid"] = df_train["rnfl12_mean"] - df_train["rnfl12_pred"]
    df_train["gc6_resid"] = df_train["gc6_mean"] - df_train["gc6_pred"]

    # -------------------------
    # 3-6. feature 리스트 구성 (sex + full OCT + age-adjusted)
    # -------------------------
    feature_list = []

    # 성별(있으면 사용)
    if "sex" in df_train.columns:
        feature_list.append("sex")

    # 기본 OCT features (존재하는 것만 사용; 제외 목록은 자동 무시)
    for c in OCT_FEATURES_BASE:
        if c in df_train.columns and c not in EXCLUDED_COLUMNS:
            feature_list.append(c)

    # age-adjusted features
    for c in ["rnfl12_mean", "rnfl12_resid", "gc6_mean", "gc6_resid"]:
        if c in df_train.columns:
            feature_list.append(c)

    # 중복 제거
    feature_list = list(dict.fromkeys(feature_list))

    print(f"[INFO] 최종 feature 수: {len(feature_list)}")
    for f in feature_list:
        print(f"   - {f}")

    # -------------------------
    # 3-7. Age-bin 별 모델 학습 & aggressive tuning (독립 test set 포함)
    # -------------------------
    models_by_age = {}
    features_by_age = {}

    age_bins = {
        "40s": (40, 49),
        "50s": (50, 59),
        "60s": (60, 69),
        "70s": (70, 79),
    }

    results_list = []

    for bin_name, (a_min, a_max) in age_bins.items():
        df_bin = df_train[(df_train["age"] >= a_min) & (df_train["age"] <= a_max)].copy()
        n_bin = len(df_bin)
        if n_bin < 100:
            print(
                f"[WARN] {bin_name} (age {a_min}-{a_max}) 데이터가 {n_bin}명으로 너무 적어 "
                f"이 age-bin 모델은 건너뜁니다."
            )
            continue

        print(
            f"\n[INFO] ===== {bin_name} 모델 학습 (n = {n_bin}) 시작 "
            f"(age {a_min}-{a_max}) ====="
        )

        X_bin = df_bin[feature_list].values
        y_bin = df_bin["age"].values

        # (1) 외부 train/test split – test set은 tuning에 사용하지 않음
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X_bin,
            y_bin,
            df_bin.index,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        # (2) train set 안에서만 hyperparameter tuning
        model, best_params, best_mae_valid, best_r2_valid = tune_xgb_for_bin(
            X_tr, y_tr, n_iter=20, random_state=RANDOM_STATE
        )

        # (3) 최종 모델을 train set 전체에 대해 평가
        y_hat_tr = model.predict(X_tr)
        mae_tr = mean_absolute_error(y_tr, y_hat_tr)
        r2_tr = r2_score(y_tr, y_hat_tr)

        # (4) 독립 test set 평가
        y_hat_te = model.predict(X_te)
        mae_te = mean_absolute_error(y_te, y_hat_te)
        r2_te = r2_score(y_te, y_hat_te)

        print(
            f"[RESULT] {bin_name} XGBoost 최종 성능 요약\n"
            f"  Train MAE = {mae_tr:.2f} 세, R² = {r2_tr:.3f}\n"
            f"  Test  MAE = {mae_te:.2f} 세, R² = {r2_te:.3f}"
        )

        # 모델 및 feature 저장 (향후 inference용)
        models_by_age[bin_name] = model
        features_by_age[bin_name] = feature_list

        # (5) test set 개별 예측 결과를 results_list에 누적
        df_res_bin = pd.DataFrame(
            {
                "age": y_te,
                "rnfl12_mean": df_bin.loc[idx_te, "rnfl12_mean"].values,
                "gc6_mean": df_bin.loc[idx_te, "gc6_mean"].values,
            },
            index=idx_te,
        )
        if "AL_R" in df_bin.columns:
            df_res_bin["AL_R"] = df_bin.loc[idx_te, "AL_R"].values

        df_res_bin["predicted_age"] = y_hat_te
        df_res_bin["age_bin"] = bin_name
        df_res_bin["split"] = "test"

        results_list.append(df_res_bin)

    if not models_by_age:
        raise RuntimeError("어떤 age-bin 도 모델 학습에 성공하지 못했습니다.")

    # -------------------------
    # 3-8. 전체 40–79세 test set에 대한 예측 결과 및 norm_table/ONAS 계산
    # -------------------------
    df_results = pd.concat(results_list, axis=0).sort_index()
    df_results["delta_age"] = df_results["predicted_age"] - df_results["age"]

    # Normative RNFL mean / SD는 전체 super-normal cohort에서 계산
    rnfl_mean = float(df_train["rnfl12_mean"].mean())
    rnfl_sd = float(df_train["rnfl12_mean"].std())

    # ONAS sigma = ΔAge SD (독립 test set 기준)
    onas_sigma = float(df_results["delta_age"].std())
    z = df_results["delta_age"] / onas_sigma
    df_results["onas_percentile"] = 100.0 * (1.0 - normal_cdf(z))

    # norm_table 저장
    norm_df = pd.DataFrame(
        {
            "rnfl_mean": [rnfl_mean],
            "rnfl_sd": [rnfl_sd],
            "onas_sigma": [onas_sigma],
        }
    )
    norm_df.to_csv(NORM_TABLE_PATH, index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    print("  Baseline은 따로 두지 않고, age-bin tuned 모델만 사용합니다.")
    print(
        f"[RESULT] Normative parameters (40–79세, relaxed super-normal, full OCT + sex)\n"
        f"  RNFL mean (12H average): {rnfl_mean:.2f} µm\n"
        f"  RNFL SD  (12H average): {rnfl_sd:.2f} µm\n"
        f"  ONAS sigma (Δage SD, test set 기준) : {onas_sigma:.2f} years"
    )
    print(f"[INFO] norm_table 저장 완료: {os.path.abspath(NORM_TABLE_PATH)}")

    # 결과 저장 (figure용, 독립 test set only)
    df_results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 결과 파일 저장 완료: {os.path.abspath(RESULTS_PATH)}")

    # 모델 / feature 저장
    joblib.dump(models_by_age, MODELS_BY_AGE_PATH)
    joblib.dump(features_by_age, FEATURES_BY_AGE_PATH)
    print(f"[INFO] 모델 저장 완료: {os.path.abspath(MODELS_BY_AGE_PATH)}")
    print(f"[INFO] feature 리스트 저장 완료: {os.path.abspath(FEATURES_BY_AGE_PATH)}")

    print("\n[INFO] 모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()
