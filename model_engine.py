"""
model_engine.py
Trains the WoE logistic regression pipeline once and caches it.
All Streamlit pages import from here.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "DataQuest26/DataQuest26/loan_book.csv"

FEATURES_TO_ANALYSE = {
    "dti_ratio": False,
    "credit_utilisation_pct": False,
    "age": False,
    "annual_income": False,
    "employment_length_years": False,
    "num_delinquencies_2yr": False,
    "months_since_oldest_account": False,
    "loan_to_income": False,
    "pct_accounts_current": False,
    "num_open_accounts": False,
    "num_hard_inquiries_6mo": False,
    "loan_amount": False,
    "has_delinquency": True,
    "high_risk_combo": True,
    "phone_verified": True,
    "home_ownership": True,
    "loan_purpose": True,
    "email_domain_type": True,
    "region": True,
}

NAVY  = "#0A1628"
TEAL  = "#007B8A"
GOLD  = "#C9A84C"
RED   = "#C0392B"
GREEN = "#1A6B3C"
MUTED = "#888888"
PALE  = "#E8F4F6"


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df["home_ownership"] = (
        df["home_ownership"].str.upper().replace({"RENTING": "RENT", "OWNER": "OWN"})
    )
    df["loan_purpose"] = (
        df["loan_purpose"].str.lower().str.replace(" ", "_").str.strip()
    )
    df["has_delinquency"] = df["months_since_last_delinquency"].notna().astype(int)
    df["months_since_last_delinquency"] = df["months_since_last_delinquency"].fillna(0)

    df_train = df[df["set"] == "train"].copy()
    df_test  = df[df["set"] == "test"].copy()

    impute_cols = ["annual_income", "employment_length_years", "num_open_accounts"]
    medians = df_train[impute_cols].median()
    for col in impute_cols:
        df_train[col] = df_train[col].fillna(medians[col])
        df_test[col]  = df_test[col].fillna(medians[col])

    p99 = df_train["annual_income"].quantile(0.99)
    p01 = df_train["annual_income"].quantile(0.01)
    for d in [df_train, df_test]:
        d["annual_income"] = d["annual_income"].clip(lower=p01, upper=p99)
        d["credit_utilisation_pct"] = d["credit_utilisation_pct"].clip(upper=100)

    for d in [df_train, df_test]:
        d["loan_to_income"]   = d["loan_amount"] / (d["annual_income"] + 1)
        d["high_risk_combo"]  = ((d["credit_utilisation_pct"] > 70) & (d["dti_ratio"] > 0.35)).astype(int)
        d["phone_verified"]   = d["phone_verified"].astype(int)

    return df, df_train, df_test


def woe_transform(df_fit, df_transform, feature, target, bins=6, is_categorical=False):
    eps = 1e-6
    total_e  = df_fit[target].sum()
    total_ne = len(df_fit) - total_e

    if is_categorical:
        grp = df_fit.groupby(feature)[target].agg(["sum", "count"])
        grp.columns = ["events", "total"]
        grp["non_events"] = grp["total"] - grp["events"]
        grp["pct_e"]  = grp["events"]     / (total_e  + eps)
        grp["pct_ne"] = grp["non_events"] / (total_ne + eps)
        grp["WoE"]    = np.log((grp["pct_ne"] + eps) / (grp["pct_e"] + eps))
        woe_map = grp["WoE"].to_dict()
        return df_transform[feature].map(woe_map).fillna(0), grp.reset_index()
    else:
        df_tmp = df_fit[[feature, target]].dropna().copy()
        try:
            _, bins_out = pd.qcut(df_tmp[feature], q=bins, retbins=True, duplicates="drop")
        except Exception:
            _, bins_out = pd.cut(df_tmp[feature], bins=bins, retbins=True, duplicates="drop")
        bins_out[0]  = -np.inf
        bins_out[-1] =  np.inf
        df_tmp["bin"] = pd.cut(df_tmp[feature], bins=bins_out)
        grp = df_tmp.groupby("bin", observed=False)[target].agg(["sum", "count"])
        grp.columns = ["events", "total"]
        grp["non_events"] = grp["total"] - grp["events"]
        grp["pct_e"]  = grp["events"]     / (total_e  + eps)
        grp["pct_ne"] = grp["non_events"] / (total_ne + eps)
        grp["WoE"]    = np.log((grp["pct_ne"] + eps) / (grp["pct_e"] + eps))
        woe_map = grp["WoE"].to_dict()
        med = df_fit[feature].median()
        binned = pd.cut(df_transform[feature].fillna(med), bins=bins_out)
        return binned.map(woe_map).fillna(0), grp.reset_index()


def calculate_iv(df, feature, target, bins=6, is_categorical=False):
    eps = 1e-6
    total_e  = df[target].sum()
    total_ne = len(df) - total_e

    if is_categorical:
        grp = df.groupby(feature)[target].agg(["sum", "count"])
    else:
        df_tmp = df[[feature, target]].dropna().copy()
        try:
            df_tmp["bin"] = pd.qcut(df_tmp[feature], q=bins, duplicates="drop")
        except Exception:
            df_tmp["bin"] = pd.cut(df_tmp[feature], bins=bins, duplicates="drop")
        grp = df_tmp.groupby("bin", observed=False)[target].agg(["sum", "count"])

    grp.columns = ["events", "total"]
    grp["non_events"] = grp["total"] - grp["events"]
    grp["pct_e"]  = grp["events"]     / (total_e  + eps)
    grp["pct_ne"] = grp["non_events"] / (total_ne + eps)
    grp["WoE"]    = np.log((grp["pct_ne"] + eps) / (grp["pct_e"] + eps))
    grp["IV_comp"]= (grp["pct_ne"] - grp["pct_e"]) * grp["WoE"]
    grp["default_rate"] = grp["events"] / (grp["total"] + eps)
    return grp.reset_index(), grp["IV_comp"].sum()


def train_model():
    df, df_train, df_test = load_and_clean()

    woe_numeric = [f for f, is_cat in FEATURES_TO_ANALYSE.items() if not is_cat]
    woe_cat     = [f for f, is_cat in FEATURES_TO_ANALYSE.items() if is_cat]

    X_tr_woe = pd.DataFrame(index=df_train.index)
    X_te_woe = pd.DataFrame(index=df_test.index)
    woe_bins = {}

    for feat in woe_numeric:
        if feat not in df_train.columns:
            continue
        tr_vals, grp = woe_transform(df_train, df_train, feat, "default_flag", bins=6, is_categorical=False)
        te_vals, _   = woe_transform(df_train, df_test,  feat, "default_flag", bins=6, is_categorical=False)
        X_tr_woe[f"{feat}_woe"] = tr_vals
        X_te_woe[f"{feat}_woe"] = te_vals
        woe_bins[feat] = grp

    for feat in woe_cat:
        if feat not in df_train.columns:
            continue
        tr_vals, grp = woe_transform(df_train, df_train, feat, "default_flag", is_categorical=True)
        te_vals, _   = woe_transform(df_train, df_test,  feat, "default_flag", is_categorical=True)
        X_tr_woe[f"{feat}_woe"] = tr_vals
        X_te_woe[f"{feat}_woe"] = te_vals
        woe_bins[feat] = grp

    X_tr_woe = X_tr_woe.fillna(0)
    X_te_woe = X_te_woe.fillna(0)
    y_tr = df_train["default_flag"].values
    y_te = df_test["default_flag"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(class_weight="balanced", max_iter=2000,
                                       random_state=42, C=0.5)),
    ])
    pipe.fit(X_tr_woe, y_tr)

    proba_te = pipe.predict_proba(X_te_woe)[:, 1]
    auc  = roc_auc_score(y_te, proba_te)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_te, proba_te)
    ks   = float(np.max(tpr - fpr))

    # IV table
    iv_results = {}
    for feat, is_cat in FEATURES_TO_ANALYSE.items():
        if feat in df_train.columns:
            try:
                _, iv = calculate_iv(df_train, feat, "default_flag", bins=6, is_categorical=is_cat)
                iv_results[feat] = iv
            except Exception:
                iv_results[feat] = 0.0

    iv_df = (
        pd.DataFrame({"Feature": list(iv_results.keys()), "IV": list(iv_results.values())})
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )

    def iv_label(v):
        if v < 0.02:   return "Useless"
        elif v < 0.10: return "Weak"
        elif v < 0.30: return "Medium"
        elif v < 0.50: return "Strong"
        else:          return "Very Strong"

    iv_df["Strength"] = iv_df["IV"].apply(iv_label)

    # Coefficients
    lr_model = pipe.named_steps["lr"]
    scaler   = pipe.named_steps["scaler"]
    feat_names = X_tr_woe.columns.tolist()
    coefs_scaled = lr_model.coef_[0]
    stds = scaler.scale_
    means = scaler.mean_
    intercept_adj = lr_model.intercept_[0] - np.sum(coefs_scaled * means / stds)
    coefs_woe = coefs_scaled / stds

    coef_df = pd.DataFrame({
        "Feature": feat_names,
        "Coefficient": coefs_woe,
        "AbsCoef": np.abs(coefs_woe),
    }).sort_values("AbsCoef", ascending=False).reset_index(drop=True)

    return {
        "pipe": pipe,
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "X_tr_woe": X_tr_woe,
        "X_te_woe": X_te_woe,
        "y_tr": y_tr,
        "y_te": y_te,
        "proba_te": proba_te,
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "fpr": fpr,
        "tpr": tpr,
        "iv_df": iv_df,
        "iv_results": iv_results,
        "coef_df": coef_df,
        "intercept": intercept_adj,
        "woe_bins": woe_bins,
        "feat_names": feat_names,
        "woe_numeric": woe_numeric,
        "woe_cat": woe_cat,
    }


def predict_applicant(model_data, applicant: dict) -> dict:
    """Score a single applicant dict. Returns probability, decision, scorecard."""
    pipe      = model_data["pipe"]
    df_train  = model_data["df_train"]
    woe_bins  = model_data["woe_bins"]
    feat_names = model_data["feat_names"]
    woe_numeric = model_data["woe_numeric"]
    woe_cat     = model_data["woe_cat"]

    # Derive engineered features
    income = applicant.get("annual_income", 30000)
    applicant["loan_to_income"]  = applicant.get("loan_amount", 15000) / max(income, 1)
    applicant["high_risk_combo"] = int(
        applicant.get("credit_utilisation_pct", 0) > 70 and
        applicant.get("dti_ratio", 0) > 0.35
    )
    applicant["phone_verified"]  = int(applicant.get("phone_verified", True))
    applicant["has_delinquency"] = int(applicant.get("has_delinquency", False))

    eps = 1e-6
    total_e  = df_train["default_flag"].sum()
    total_ne = len(df_train) - total_e

    woe_row = {}
    scorecard = []

    for feat in woe_numeric:
        fname = f"{feat}_woe"
        if fname not in feat_names:
            continue
        grp = woe_bins.get(feat)
        val = applicant.get(feat, np.nan)
        if grp is None or pd.isna(val):
            woe_val = 0.0
        else:
            bin_col = grp.columns[0]
            woe_val = 0.0
            for _, row in grp.iterrows():
                b = row[bin_col]
                if hasattr(b, "left") and b.left <= val <= b.right:
                    woe_val = row["WoE"]
                    break
        woe_row[fname] = woe_val
        scorecard.append({"feature": feat, "value": val, "woe": woe_val, "type": "numeric"})

    for feat in woe_cat:
        fname = f"{feat}_woe"
        if fname not in feat_names:
            continue
        grp = woe_bins.get(feat)
        val = applicant.get(feat, "")
        if grp is None:
            woe_val = 0.0
        else:
            cat_col = grp.columns[0]
            match = grp[grp[cat_col] == val]
            woe_val = float(match["WoE"].values[0]) if len(match) > 0 else 0.0
        woe_row[fname] = woe_val
        scorecard.append({"feature": feat, "value": val, "woe": woe_val, "type": "categorical"})

    X_app = pd.DataFrame([woe_row])[feat_names].fillna(0)
    prob   = float(pipe.predict_proba(X_app)[0, 1])
    score  = int(300 + (1 - prob) * 600)

    if score >= 650:
        risk_tier = "Low Risk"
        tier_color = "#1A6B3C"
    elif score >= 550:
        risk_tier = "Medium Risk"
        tier_color = "#C9A84C"
    else:
        risk_tier = "High Risk"
        tier_color = "#C0392B"

    sc_df = pd.DataFrame(scorecard)
    if not sc_df.empty:
        sc_df["abs_woe"] = sc_df["woe"].abs()
        sc_df = sc_df.sort_values("abs_woe", ascending=False)

    return {
        "probability": prob,
        "score": score,
        "risk_tier": risk_tier,
        "tier_color": tier_color,
        "decision": "APPROVED" if prob < 0.50 else "DECLINED",
        "scorecard": sc_df,
    }
