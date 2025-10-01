# app.py 

import os, json, re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
HAS_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False

# SHAP
import shap

# DiCE
import dice_ml
from dice_ml import Dice

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# ---------- .env + OpenAI (safe) ----------
from dotenv import load_dotenv
load_dotenv()  

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = "gpt-4o-mini"

OPENAI_AVAILABLE = False
_openai_mode = None
_openai_client = None
try:
    from openai import OpenAI  # SDK v1+
    if OPENAI_API_KEY:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        _openai_mode = "v1"
        OPENAI_AVAILABLE = True
except Exception:
    try:
        import openai  # legacy
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            _openai_mode = "legacy"
            OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False
        _openai_mode = None
        _openai_client = None


def get_openai_client():
    """Return a callable chat() or None if unavailable."""
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return None

    if _openai_mode == "v1":
        def _chat(model, messages, temperature=0.35, max_tokens=180):
            resp = _openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        return _chat

    if _openai_mode == "legacy":
        def _chat(model, messages, temperature=0.35, max_tokens=180):
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"].strip()
        return _chat

    return None


def format_suggestion_prompt(changes, patient_context):
    """
    Build a short, safe prompt for suggestions.
    """
    bullets = "\n".join([f"- {c['feature']}: {c['before']} → {c['after']}" for c in changes])
    ctx_keys = ['age', 'sex', 'fbs', 'exang', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'cp']
    ctx = ", ".join([f"{k}={patient_context.get(k)}" for k in ctx_keys if k in patient_context])

    return f"""
You are a health coach writing neutral, educational tips (not medical advice).

Patient context (for tone): {ctx}

Requested counterfactual adjustments:
{bullets}

For EACH bullet above, give ONE short actionable suggestion (lifestyle/diet/exercise/general habit or “discuss with a cardiologist” when appropriate).
- No diagnoses, no prescriptions.
- ≤ 24 words per bullet.
Return bullet points only.
"""


def get_ai_suggestions(chat_fn, changes, patient_context, st_obj=None):
    if not changes:
        return "No feature changes recommended."
    prompt = format_suggestion_prompt(changes, patient_context)
    if st_obj:
        with st_obj.spinner("Generating suggestions…"):
            return chat_fn(OPENAI_MODEL, [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=220)
    return chat_fn(OPENAI_MODEL, [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=220)
# ---------- end OpenAI block ----------


# --------------------------- Page & Theme --------------------------- #
st.set_page_config(
    page_title="Counterfactual Explorer — Heart Disease",
    page_icon="🫀",
    layout="wide"
)

# CSS tuned for both light and dark themes
st.markdown(
    """
    <style>
      :root {
        --card-bg-light: #ffffff;
        --card-bg-dark: #111418;
        --card-border-light: rgba(0,0,0,0.07);
        --card-border-dark: rgba(255,255,255,0.08);
        --muted-light: #4a4a4a;
        --muted-dark: #a8b3bd;
      }
      @media (prefers-color-scheme: dark) {
        .app-card {
          background: var(--card-bg-dark) !important;
          border-color: var(--card-border-dark) !important;
        }
        .muted { color: var(--muted-dark) !important; }
      }
      @media (prefers-color-scheme: light) {
        .app-card {
          background: var(--card-bg-light) !important;
          border-color: var(--card-border-light) !important;
        }
        .muted { color: var(--muted-light) !important; }
      }
      .app-card{
        border: 1px solid;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 16px;
      }
      .section-title{
        margin: 4px 0 12px 0;
      }
      .tight-caption{ margin-top: -6px; }
      .btn-wide > button{ width:100%; }
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"]{
        height: 42px; padding-top: 8px; border-radius: 10px 10px 0 0;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------- Config --------------------------- #
CSV_PATH = "heart_disease_uci.csv"

EXPECTED_CATEGORICAL = ['sex', 'origin', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
EXPECTED_CONTINUOUS  = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
RAW_TARGET = 'num'
BIN_TARGET = 'target'

IMMUTABLE_DEFAULT = ['sex', 'age', 'origin']
PERMITTED_RANGE_DEFAULT = {
    'trestbps': (90, 200),
    'chol':     (100, 400),
    'thalach':  (60, 220),
    'oldpeak':  (0.0, 6.5),
    'ca':       (0, 4)
}

N_SHAP_BACKGROUND = 50
TOPK_SHAP = 8
N_CF = 3

NICE_LABELS = {
    'age': 'Age (years)',
    'sex': 'Sex',
    'origin': 'Origin',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting BP (mmHg)',
    'chol': 'Cholesterol (mg/dL)',
    'fbs': 'Fasting Blood Sugar (>120 mg/dL)',
    'restecg': 'Resting ECG',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise-induced Angina',
    'oldpeak': 'ST Depression',
    'slope': 'ST Slope',
    'ca': '# Major Vessels',
    'thal': 'Thalassemia',
    BIN_TARGET: 'Target (1=disease, 0=none)'
}
def pretty(c): return NICE_LABELS.get(c, c)

# --------------------------- Data prep --------------------------- #
def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if 'dataset' in df.columns and 'origin' not in df.columns:
        rename_map['dataset'] = 'origin'
    if 'thalch' in df.columns and 'thalach' not in df.columns:
        rename_map['thalch'] = 'thalach'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_schema(df).copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    required = set(EXPECTED_CATEGORICAL + EXPECTED_CONTINUOUS + [RAW_TARGET])
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # normalize boolean-like -> 'True'/'False'
    for c in ['fbs', 'exang']:
        df[c] = df[c].apply(lambda v: 'True' if str(v).strip().lower() in ['true','1','yes'] else 'False')

    for c in EXPECTED_CATEGORICAL:
        df[c] = df[c].astype(str)

    # simple imputations safety net
    for c in EXPECTED_CONTINUOUS:
        if df[c].isna().any():
            df[c] = SimpleImputer(strategy='median').fit_transform(df[[c]])
    for c in EXPECTED_CATEGORICAL:
        if df[c].isna().any():
            df[c] = SimpleImputer(strategy='most_frequent').fit_transform(df[[c]])

    # binarize target
    df[BIN_TARGET] = (pd.to_numeric(df[RAW_TARGET], errors='coerce').fillna(0) > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET])
    return df

def build_preprocessor():
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', num_pipe, EXPECTED_CONTINUOUS),
            ('cat', cat_pipe, EXPECTED_CATEGORICAL)
        ],
        remainder='drop'
    )

@st.cache_resource(show_spinner=True)
def load_and_train_models(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Place heart_disease_uci.csv next to app.py")

    raw = pd.read_csv(csv_path)
    df = basic_clean(raw)

    X = df[EXPECTED_CATEGORICAL + EXPECTED_CONTINUOUS].copy()
    y = df[BIN_TARGET].copy()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}

    # Logistic
    logi = Pipeline(steps=[('prep', build_preprocessor()),
                           ('clf', LogisticRegression(max_iter=2000))])
    logi.fit(Xtr, ytr)
    acc = accuracy_score(yte, logi.predict(Xte))
    auc = roc_auc_score(yte, logi.predict_proba(Xte)[:,1])
    models['Logistic Regression'] = {'pipe': logi, 'acc': acc, 'auc': auc, 'Xtest': Xte, 'ytest': yte}

    # Random Forest
    rf = Pipeline(steps=[('prep', build_preprocessor()),
                         ('clf', RandomForestClassifier(n_estimators=300, random_state=42))])
    rf.fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:,1])
    models['Random Forest'] = {'pipe': rf, 'acc': acc, 'auc': auc, 'Xtest': Xte, 'ytest': yte}

    # XGB 
    if HAS_XGB:
        xgb = Pipeline(steps=[('prep', build_preprocessor()),
                              ('clf', XGBClassifier(n_estimators=300, max_depth=4,
                                                    subsample=0.9, colsample_bytree=0.9,
                                                    learning_rate=0.05, random_state=42,
                                                    eval_metric='logloss', use_label_encoder=False))])
        xgb.fit(Xtr, ytr)
        acc = accuracy_score(yte, xgb.predict(Xte))
        auc = roc_auc_score(yte, xgb.predict_proba(Xte)[:,1])
        models['XGBoost'] = {'pipe': xgb, 'acc': acc, 'auc': auc, 'Xtest': Xte, 'ytest': yte}

    return df, X, y, models

# --------------------------- SHAP helpers --------------------------- #
def _transform_and_names(pipe, X_df):
    pre = pipe.named_steps['prep']
    X_trans = pre.transform(X_df)
    try:
        trans_names = pre.get_feature_names_out()
    except Exception:
        trans_names = np.array([f"f{i}" for i in range(X_trans.shape[1])])

    groups = []
    for name in trans_names:
        if name.startswith("num__"):
            groups.append(name.replace("num__", ""))
        elif name.startswith("cat__"):
            raw = name.replace("cat__", "").split("_", 1)[0]
            groups.append(raw)
        else:
            groups.append(name)
    return X_trans, np.array(trans_names), np.array(groups)

def shap_topk_bar(pipe, X_df, instance_df, k=8, title="SHAP contributions (top-k)"):
    pre = pipe.named_steps['prep']
    clf = pipe.named_steps['clf']

    bg_raw = X_df.sample(min(len(X_df), N_SHAP_BACKGROUND), random_state=42)
    X_bg_t, trans_names, groups = _transform_and_names(pipe, bg_raw)
    X_inst_t, _, _ = _transform_and_names(pipe, instance_df)

    if hasattr(clf, "predict_proba"):
        f_t = lambda data: clf.predict_proba(data)[:, 1]
    else:
        f_t = lambda data: clf.predict(data)

    with st.spinner("Computing SHAP explanations..."):
        explainer = shap.Explainer(f_t, X_bg_t)
        sv = explainer(X_inst_t)[0].values

    sv_series = pd.Series(sv, index=trans_names)
    agg = sv_series.groupby(groups).sum().sort_values(key=np.abs, ascending=False)
    topk = agg.head(k)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ['#2F80ED' if v >= 0 else '#EB5757' for v in topk.values]
    ax.barh(range(len(topk))[::-1], topk.values[::-1],
            tick_label=[pretty(c) for c in topk.index[::-1]], color=colors)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("SHAP value (aggregated)")
    fig.tight_layout()

    return fig, topk, None

# --------------------------- DiCE helpers --------------------------- #
def dice_cf(pipeline, raw_df, query_instance, immutables, permitted_range, n_cf=N_CF, desired_class=0, method="genetic"):
    d = dice_ml.Data(
        dataframe=raw_df,
        continuous_features=EXPECTED_CONTINUOUS,
        categorical_features=EXPECTED_CATEGORICAL,
        outcome_name=BIN_TARGET
    )
    m = dice_ml.Model(model=pipeline, backend="sklearn")
    exp = Dice(d, m, method=method)
    features_can_vary = [f for f in EXPECTED_CONTINUOUS + EXPECTED_CATEGORICAL if f not in immutables]
    return exp.generate_counterfactuals(
        query_instance,
        total_CFs=n_cf,
        desired_class=desired_class,
        features_to_vary=features_can_vary,
        permitted_range=permitted_range
    )

def feasibility_chip(col, before, after, immutables, permitted_range):
    if col in immutables: return "⛔ immutable"
    rng = permitted_range.get(col)
    if rng is None: return "✔ achievable"
    try:
        delta = abs(float(after) - float(before))
        span = max(abs(rng[1]-rng[0]), 1e-9)
        return "✖ tough" if (delta/span) > 0.35 else "✔  achievable"
    except Exception:
        return "✔  achievable"

def english_explanation(b_row, a_row, pred_b, pred_a):
    changes = []
    for c in b_row.index:
        if b_row[c] != a_row[c]:
            changes.append(f"{pretty(c)}: {b_row[c]} → {a_row[c]}")
    to_text = {0:"no disease (0)", 1:"disease (1)"}
    if not changes:
        return f"Prediction changes from {to_text[pred_b]} to {to_text[pred_a]} without feature changes."
    return "If you adjust " + "; ".join(changes) + f", the prediction changes from {to_text[pred_b]} to {to_text[pred_a]}."

def pdf_report(filename, meta, prediction, prob, fig_paths, delta_tables, explanations, ai_suggestions_blocks):
    c = canvas.Canvas(filename, pagesize=A4)
    w, h = A4; y = h - 50
    c.setFont("Helvetica-Bold", 14); c.drawString(40, y, "Counterfactual Report — Heart Disease (Demo)"); y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"); y -= 12
    c.drawString(40, y, f"Model: {meta['model_name']} | Version: {meta['model_version']} | CF method: {meta['cf_method']}"); y -= 12
    c.drawString(40, y, f"Immutables: {', '.join(meta['immutables'])}"); y -= 12
    c.drawString(40, y, f"Permitted ranges: {json.dumps(meta['permitted_range'])[:1000]}"); y -= 18
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, f"Prediction: {prediction} | Probability (disease): {prob:.3f}"); y -= 16

    for p in fig_paths:
        try:
            img = ImageReader(p)
            c.drawImage(img, 40, y-200, width=300, height=200, preserveAspectRatio=True, mask='auto')
            y -= 210
            if y < 120: c.showPage(); y = h - 60
        except Exception:
            pass

    for i, (df, expl, sugg) in enumerate(zip(delta_tables, explanations, ai_suggestions_blocks), 1):
        if y < 120: c.showPage(); y = h - 60
        c.setFont("Helvetica-Bold", 11); c.drawString(40, y, f"Counterfactual {i}:"); y -= 14
        c.setFont("Helvetica", 9)
        for _, row in df.head(12).iterrows():
            c.drawString(40, y, f"- {row['Feature']}: {row['Before']} → {row['After']} (Δ={row['Delta']}) [{row['Feasibility']}]"); y -= 12
            if y < 120: c.showPage(); y = h - 60
        if len(df) > 12: c.drawString(40, y, f"... ({len(df)-12} more changes)"); y -= 12
        c.drawString(40, y, f"Explanation: {expl}"); y -= 12
        c.drawString(40, y, "Suggestions:"); y -= 12
        for line in sugg.splitlines():
            c.drawString(46, y, line[:110]); y -= 11
            if y < 120: c.showPage(); y = h - 60

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, y, "DISCLAIMER: Educational demo. Not medical advice.")
    c.save()

# --------------------------- App UI --------------------------- #
st.title("🫀 Counterfactual Explorer for Heart Risk")

with st.container():
    with st.spinner("Loading data & training models (first run only)…"):
        df, X, y, models = load_and_train_models(CSV_PATH)

# Sidebar — configuration
with st.sidebar:
    st.header("⚙️ Settings")

    if not OPENAI_API_KEY:
        st.warning("OpenAI key not found in .env. Suggestions will be disabled.", icon="⚠️")
    else:
        st.success("OpenAI suggestions enabled via .env", icon="✅")

    st.subheader("Immutables & Ranges")
    immutables = st.multiselect(
        "Immutable features",
        options=EXPECTED_CATEGORICAL + EXPECTED_CONTINUOUS,
        default=IMMUTABLE_DEFAULT
    )

    permitted_range = {}
    for c, (lo, hi) in PERMITTED_RANGE_DEFAULT.items():
        lo_v = st.number_input(f"{pretty(c)} min", value=float(lo))
        hi_v = st.number_input(f"{pretty(c)} max", value=float(hi))
        if lo_v > hi_v: st.error(f"{c}: min > max")
        permitted_range[c] = (lo_v, hi_v)

    desired_class = st.selectbox("Desired class for CFs", [0,1], index=0)
    cf_method = st.selectbox("DiCE method", ["genetic","random"], index=0)

# Model picker row
model_names = list(models.keys())
top_row = st.container()
with top_row:
    pick_col, metrics_col = st.columns([2, 1])
    with pick_col:
        choice = st.selectbox("Choose Model for prediction:", model_names, index=0)
    with metrics_col:
        m1, m2 = st.columns(2)
        with m1: st.metric("Accuracy", f"{models[choice]['acc']*100:.2f}%")
        with m2: st.metric("ROC AUC", f"{models[choice]['auc']:.3f}")

pipe = models[choice]['pipe']; Xtest = models[choice]['Xtest']

# pick a positive case (for risk reduction demos)
def pick_positive_case(pipe, X_test):
    preds = pipe.predict(X_test)
    pos = np.where(preds==1)[0]
    return X_test.iloc[[pos[0]]] if len(pos)>0 else X_test.iloc[[0]]

query = pick_positive_case(pipe, Xtest)

# Editable Patient Form (3 columns)
st.subheader("🎯 Customise Patient Details")

def cat_options(col): return sorted(list(map(str, X[col].unique())))

colA, colB, colC = st.columns(3)

with colA:
    age_val    = st.number_input(pretty('age'), value=float(query['age'].iloc[0]), step=1.0)
    sex_val    = st.selectbox(pretty('sex'), options=cat_options('sex'),
                              index=cat_options('sex').index(query['sex'].iloc[0]))
    origin_val = st.selectbox(pretty('origin'), options=cat_options('origin'),
                              index=cat_options('origin').index(query['origin'].iloc[0]))

with colB:
    trestbps_val = st.slider(pretty('trestbps'), 90, 200, int(query['trestbps'].iloc[0]))
    chol_val     = st.slider(pretty('chol'), 100, 400, int(query['chol'].iloc[0]))
    thalach_val  = st.slider(pretty('thalach'), 60, 220, int(query['thalach'].iloc[0]))
    oldpeak_val  = st.slider(pretty('oldpeak'), 0.0, 6.5, float(query['oldpeak'].iloc[0]), step=0.1)
    ca_val       = st.slider(pretty('ca'), 0, 4, int(query['ca'].iloc[0]))  # Major vessels

with colC:
    cp_val       = st.selectbox(pretty('cp'), options=cat_options('cp'),
                                index=cat_options('cp').index(query['cp'].iloc[0]))
    exang_val    = st.selectbox(pretty('exang'), options=['True','False'],
                                index=0 if query['exang'].iloc[0]=='True' else 1)
    fbs_val      = st.selectbox(pretty('fbs'), options=['True','False'],
                                index=0 if query['fbs'].iloc[0]=='True' else 1)
    slope_val    = st.selectbox(pretty('slope'), options=cat_options('slope'),
                                index=cat_options('slope').index(query['slope'].iloc[0]))
    thal_val     = st.selectbox(pretty('thal'), options=cat_options('thal'),
                                index=cat_options('thal').index(query['thal'].iloc[0]))

user_row = pd.DataFrame([{
    'age': age_val, 'sex': sex_val, 'origin': origin_val, 'cp': cp_val,
    'trestbps': trestbps_val, 'chol': chol_val, 'thalach': thalach_val,
    'fbs': fbs_val, 'restecg': query['restecg'].iloc[0],
    'exang': exang_val, 'oldpeak': oldpeak_val, 'slope': slope_val,
    'ca': ca_val, 'thal': thal_val
}], columns=X.columns)

pred_proba = pipe.predict_proba(user_row)[0,1] if hasattr(pipe,'predict_proba') else float(pipe.predict(user_row))
pred_label = int(pred_proba >= 0.5)
st.info(f"**Prediction:** {'Disease (1)' if pred_label==1 else 'No disease (0)'} | **Prob(disease):** {pred_proba:.3f}")

# SHAP for current patient
fig_shap, _, _ = shap_topk_bar(pipe, X, user_row, k=TOPK_SHAP, title="Top contributions (current patient)")
st.pyplot(fig_shap)
st.caption("Bars left/right show negative/positive contribution to predicted risk.")

# Generate CFs
lcol, rcol = st.columns([1,1])
with lcol:
    if st.button("✨ Generate Counterfactuals"):
        raw_for_dice = pd.concat([X, y.rename(BIN_TARGET)], axis=1)
        cf_obj = dice_cf(pipe, raw_for_dice, user_row, immutables, permitted_range,
                         n_cf=N_CF, desired_class=desired_class, method=cf_method)
        st.session_state['cf_result'] = cf_obj
        st.session_state['query_snapshot'] = user_row.copy()
        st.session_state.setdefault('audit', []).append({
            "time": datetime.now().isoformat(timespec='seconds'),
            "model_name": choice,
            "model_version": "v1-demo",
            "cf_method": cf_method,
            "immutables": immutables,
            "permitted_range": permitted_range
        })
with rcol:
    st.caption("Tip: set desired class = 0 for risk reduction.")

cf_obj = st.session_state.get('cf_result')
ai_chat = get_openai_client()

if cf_obj is not None:
    query_snapshot = st.session_state.get('query_snapshot', user_row)
    cf_df = cf_obj.cf_examples_list[0].final_cfs_df

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("🧪 Counterfactuals")

    explanations, delta_tables, fig_paths, ai_suggestions_blocks = [], [], [], []

    # Use enumerate for unique keys and tidy numbering
    for idx, (_, row) in enumerate(cf_df.iterrows(), 1):
        st.markdown(f"**CF #{idx}**")

        after_row = row[X.columns]

        pb = pipe.predict_proba(query_snapshot)[0,1] if hasattr(pipe,'predict_proba') else float(pipe.predict(query_snapshot))
        pa = pipe.predict_proba(pd.DataFrame([after_row], columns=X.columns))[0,1] if hasattr(pipe,'predict_proba') else float(pipe.predict(pd.DataFrame([after_row], columns=X.columns)))
        lb, la = int(pb>=0.5), int(pa>=0.5)

        # deltas table
        recs, change_list_for_ai = [], []
        for c in X.columns:
            b, a = query_snapshot.iloc[0][c], after_row[c]
            if b != a:
                try: delta = round(float(a)-float(b), 3)
                except: delta = "—"
                recs.append({"Feature": pretty(c), "Before": b, "After": a,
                             "Delta": delta, "Feasibility": feasibility_chip(c, b, a, immutables, permitted_range)})
                change_list_for_ai.append({"feature": pretty(c), "before": b, "after": a})
        delta_df = pd.DataFrame(recs) if recs else pd.DataFrame(columns=["Feature","Before","After","Delta","Feasibility"])

        st.dataframe(delta_df, use_container_width=True)

        msg = english_explanation(query_snapshot.iloc[0], after_row, lb, la)
        st.success(msg)
        explanations.append(msg); delta_tables.append(delta_df)

        # AI suggestions
        if ai_chat is None:
            st.warning("OpenAI suggestions disabled — install `openai` and set OPENAI_API_KEY in .env.", icon="⚠️")
            ai_text = "(No suggestions — OpenAI unavailable)"
        else:
            patient_ctx = {k: user_row.iloc[0][k] for k in user_row.columns}
            ai_text = get_ai_suggestions(ai_chat, change_list_for_ai, patient_ctx, st_obj=st)
            st.info(ai_text)
        ai_suggestions_blocks.append(ai_text)

        # SHAP after CF
        fig_after, _, _ = shap_topk_bar(pipe, X, pd.DataFrame([after_row], columns=X.columns),
                                        k=TOPK_SHAP, title="Top contributions after applying CF")
        st.pyplot(fig_after)

        # Save image for PDF
        png_path = f"shap_cf_{idx}.png"; fig_after.savefig(png_path, dpi=150, bbox_inches='tight')
        fig_paths.append(png_path)

        # Apply CF (unique key via idx)
        with st.container():
            st.markdown('<div class="btn-wide">', unsafe_allow_html=True)
            if st.button(f"Apply CF #{idx}", key=f"apply_{idx}"):
                for c in X.columns:
                    user_row.at[0, c] = after_row[c]
                st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

    # PDF download (includes AI suggestions)
    st.markdown('<div class="btn-wide">', unsafe_allow_html=True)
    if st.button("📄 Build PDF Report"):
        current_png = "shap_current.png"; fig_shap.savefig(current_png, dpi=150, bbox_inches='tight')
        meta = {"model_name": choice, "model_version": "v1-demo",
                "cf_method": cf_method, "immutables": immutables, "permitted_range": permitted_range}
        pdf_name = f"cf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_report(pdf_name, meta, pred_label, pred_proba, [current_png]+fig_paths, delta_tables, explanations, ai_suggestions_blocks)
        with open(pdf_name, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf_name, mime="application/pdf", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs: Model comparison / consensus / uncertainty
st.subheader("🔀 Model Comparison & Consensus")

tabs = st.tabs([*models.keys(), "Consensus CFs", "Uncertainty"])

for i, name in enumerate(models.keys()):
    with tabs[i]:
        mdl = models[name]; p = mdl['pipe']
        proba = p.predict_proba(user_row)[0,1] if hasattr(p,'predict_proba') else float(p.predict(user_row))
        lab = int(proba>=0.5)
        st.info(f"**{name}** → Prediction: {'Disease (1)' if lab==1 else 'No disease (0)'} | Prob={proba:.3f} | Acc={mdl['acc']*100:.2f}% | AUC={mdl['auc']:.3f}")
        fig_cmp, _, _ = shap_topk_bar(p, X, user_row, k=TOPK_SHAP, title=f"{name} — SHAP top-k")
        st.pyplot(fig_cmp)

with tabs[len(models.keys())]:
    st.caption("Features that appear in CF changes for ≥2 models.")
    raw_for_dice = pd.concat([X, y.rename(BIN_TARGET)], axis=1)
    changed_sets = []
    for name, mdl in models.items():
        p = mdl['pipe']
        cf_tmp = dice_cf(p, raw_for_dice, user_row, immutables, permitted_range,
                         n_cf=1, desired_class=desired_class, method=cf_method)
        after = cf_tmp.cf_examples_list[0].final_cfs_df.iloc[0][X.columns]
        ch = set([c for c in X.columns if user_row.iloc[0][c] != after[c]])
        changed_sets.append(ch)
        st.write(f"**{name} changed**: " + (", ".join(pretty(c) for c in ch) if ch else "(none)"))
    tally = {}
    for s in changed_sets:
        for f in s: tally[f] = tally.get(f,0)+1
    consensus = [f for f,k in tally.items() if k>=2]
    if consensus: st.success("Consensus (≥2 models): " + ", ".join(pretty(c) for c in consensus))
    else: st.warning("No consensus found.")

with tabs[len(models.keys())+1]:
    probs = []
    for mdl in models.values():
        p = mdl['pipe']
        try: probs.append(p.predict_proba(user_row)[0,1])
        except: probs.append(float(p.predict(user_row)))
    mean_p, std_p = float(np.mean(probs)), float(np.std(probs))
    st.write(f"**Mean prob(disease)**: {mean_p:.3f} ± {std_p:.3f}")
    fig, ax = plt.subplots(figsize=(6.2,2.6))
    ax.bar([0], [mean_p], yerr=[std_p], capsize=8)
    ax.set_xticks([0]); ax.set_xticklabels(['Ensemble']); ax.set_ylim(0,1)
    ax.set_ylabel("Prob(disease)"); ax.set_title("Uncertainty across models")
    fig.tight_layout()
    st.pyplot(fig)

# Audit Trail
st.subheader("📝 Audit Trail")
audit = st.session_state.get('audit', [])
st.json(audit if audit else [])
st.markdown('> **Disclaimer:** Educational demo. Not medical advice.')
