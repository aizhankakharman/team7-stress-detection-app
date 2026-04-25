from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Team 7 Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --glass:rgba(255,255,255,.72);
  --blue:#2563eb; --cyan:#06b6d4; --purple:#7c3aed; --pink:#e11d48; --green:#16a34a;
}
html, body, [class*="css"] {font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;}
.block-container {padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1360px;}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 7%, rgba(124,58,237,.25) 0, transparent 27%),
    radial-gradient(circle at 92% 10%, rgba(6,182,212,.25) 0, transparent 31%),
    radial-gradient(circle at 70% 87%, rgba(37,99,235,.18) 0, transparent 28%),
    linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #ecfeff 100%);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #020617 0%, #111827 52%, #1e1b4b 100%);
  border-right: 1px solid rgba(255,255,255,.12);
}
[data-testid="stSidebar"] * {color: #f8fafc !important;}
[data-testid="stSidebar"] .stButton>button {
  background: linear-gradient(135deg, #38bdf8, #818cf8);
  color:white !important; border:0; box-shadow: 0 12px 30px rgba(56,189,248,.22);
}
.hero {
  position: relative; overflow: hidden; min-height: 235px;
  border-radius: 38px; padding: 34px 38px;
  background: linear-gradient(135deg, rgba(2,6,23,.98) 0%, rgba(30,64,175,.96) 44%, rgba(8,145,178,.94) 100%);
  color:white; box-shadow: 0 30px 90px rgba(15,23,42,.23); margin-bottom: 20px;
  border: 1px solid rgba(255,255,255,.18);
}
.hero:before {content:""; position:absolute; inset:-2px; background: radial-gradient(circle at 86% 20%, rgba(255,255,255,.24), transparent 22%), radial-gradient(circle at 72% 78%, rgba(255,255,255,.14), transparent 30%); pointer-events:none;}
.orb {position:absolute; border-radius:50%; filter: blur(.2px); opacity:.72; animation: floaty 6s ease-in-out infinite;}
.orb.one {width:130px; height:130px; right:70px; top:28px; background:rgba(255,255,255,.15);} 
.orb.two {width:72px; height:72px; right:225px; bottom:30px; background:rgba(103,232,249,.22); animation-delay:1.6s;}
@keyframes floaty {0%,100%{transform:translateY(0)} 50%{transform:translateY(-14px)}}
.hero-content {position:relative; z-index:2; max-width: 900px;}
.hero h1 {font-size: 54px; line-height: 1.02; margin: 14px 0 14px 0; letter-spacing: -1.8px; font-weight: 950;}
.hero p {font-size: 18px; opacity: .94; margin: 0; max-width: 900px;}
.pill {display:inline-flex; align-items:center; gap:7px; padding: 9px 14px; border-radius:999px; background:rgba(255,255,255,.14); color:white; font-weight:850; margin-right:8px; font-size:13px; border:1px solid rgba(255,255,255,.22); backdrop-filter: blur(12px);}
.glass-card, .card {
  background: var(--glass); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  padding: 24px; border: 1px solid rgba(255,255,255,.72); border-radius: 30px;
  box-shadow: 0 20px 55px rgba(15, 23, 42, .10); margin-bottom: 18px;
}
.card h3, .glass-card h3 {margin:0 0 8px 0; color:var(--ink); font-size:22px; font-weight:900; letter-spacing:-.3px;}
.muted {color:var(--muted); font-size:14px; line-height:1.55;}
.section-label {font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:#2563eb; font-weight:950; margin-bottom:8px;}
.kpi-grid {display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; margin: 6px 0 18px 0;}
.kpi {background:rgba(255,255,255,.78); border:1px solid rgba(226,232,240,.85); border-radius:24px; padding:17px; box-shadow:0 14px 35px rgba(15,23,42,.06);} 
.kpi .label {font-size:12px; color:#64748b; font-weight:850; text-transform:uppercase; letter-spacing:.08em;}
.kpi .value {font-size:30px; font-weight:950; color:#0f172a; margin-top:6px; letter-spacing:-.6px;}
.signal-grid {display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px;}
.signal {background:#fff; border:1px solid #e2e8f0; border-radius:22px; padding:14px 15px;}
.signal .name {font-weight:900; color:#0f172a; font-size:14px;}
.signal .value {font-weight:950; font-size:24px; color:#1d4ed8; margin-top:4px;}
.result-card {padding: 28px; border-radius: 34px; margin: 6px 0 18px 0; box-shadow: 0 22px 62px rgba(15,23,42,.13); position:relative; overflow:hidden; border:1px solid rgba(255,255,255,.75);} 
.result-card:after {content:""; position:absolute; width:190px; height:190px; border-radius:50%; right:-55px; top:-65px; background:rgba(255,255,255,.45);} 
.result-stress {background: linear-gradient(135deg,#fff1f2 0%, #ffedd5 52%, #ffffff 100%); border-left: 10px solid #e11d48;}
.result-normal {background: linear-gradient(135deg,#ecfdf5 0%, #eff6ff 58%, #ffffff 100%); border-left: 10px solid #16a34a;}
.result-title {font-size: 17px; font-weight:950; color:#334155; margin-bottom:7px; text-transform:uppercase; letter-spacing:.06em;}
.big-number {font-size: 46px; font-weight: 1000; color:#0f172a; margin:0; letter-spacing:-1.2px;}
.confidence {font-size:18px; margin-top:9px; color:#334155;}
.check-ok {background:#dcfce7; color:#166534; padding:10px 14px; border-radius:999px; font-weight:950; display:inline-block; border:1px solid #86efac;}
.check-bad {background:#ffe4e6; color:#9f1239; padding:10px 14px; border-radius:999px; font-weight:950; display:inline-block; border:1px solid #fda4af;}
.prob-row {margin:12px 0;}
.prob-label {display:flex; justify-content:space-between; font-weight:900; color:#0f172a; margin-bottom:7px;}
.prob-bg {height:16px; background:#e2e8f0; border-radius:999px; overflow:hidden; border:1px solid #cbd5e1;}
.prob-fill {height:100%; border-radius:999px; background:linear-gradient(90deg,#2563eb,#06b6d4,#7c3aed);}
[data-testid="stMetricValue"] {font-size: 26px; font-weight:950; color:#0f172a;}
.stButton>button {border-radius: 18px; font-weight:950; padding:.78rem 1rem; border:1px solid #cbd5e1;}
.stButton>button[kind="primary"] {background: linear-gradient(135deg,#2563eb,#7c3aed); border:0; box-shadow:0 18px 35px rgba(37,99,235,.24);} 
[data-testid="stDataFrame"] {border-radius: 18px; overflow:hidden; border:1px solid #e2e8f0;}
.footer {text-align:center; color:#64748b; padding-top:18px; font-size:13px;}
hr.soft {border:0; height:1px; background:linear-gradient(90deg,transparent,#cbd5e1,transparent); margin:18px 0;}
@media(max-width: 900px){.hero h1{font-size:38px}.kpi-grid{grid-template-columns:repeat(2,1fr)}.signal-grid{grid-template-columns:1fr}}

/* Fix sidebar number input visibility */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="input"] input {
  color: #0f172a !important;
  background: #ffffff !important;
  -webkit-text-fill-color: #0f172a !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] {background:#ffffff !important; border-radius:14px !important;}
[data-testid="stSidebar"] button svg {color:#0f172a !important; fill:#0f172a !important;}
.recommend-card {position:relative; overflow:hidden; border-radius:34px; padding:26px; margin:6px 0 18px 0; color:white; box-shadow:0 22px 62px rgba(15,23,42,.16); border:1px solid rgba(255,255,255,.34);}
.recommend-card:before {content:""; position:absolute; width:220px; height:220px; border-radius:50%; right:-70px; top:-80px; background:rgba(255,255,255,.16);} 
.recommend-card:after {content:""; position:absolute; width:110px; height:110px; border-radius:50%; right:135px; bottom:-45px; background:rgba(255,255,255,.13);} 
.rec-stress-high {background:linear-gradient(135deg,#450a0a 0%,#9f1239 50%,#f97316 100%);} 
.rec-stress-mid {background:linear-gradient(135deg,#431407 0%,#b45309 54%,#fb7185 100%);} 
.rec-stress-low {background:linear-gradient(135deg,#172554 0%,#7c2d12 52%,#f59e0b 100%);} 
.rec-stable {background:linear-gradient(135deg,#052e2b 0%,#047857 52%,#38bdf8 100%);} 
.rec-content {position:relative; z-index:2;}
.rec-kicker {font-size:12px; text-transform:uppercase; letter-spacing:.14em; opacity:.88; font-weight:950; margin-bottom:10px;}
.rec-title {font-size:32px; font-weight:1000; letter-spacing:-.8px; margin:0 0 8px 0;}
.rec-subtitle {font-size:16px; opacity:.92; max-width:900px; margin-bottom:18px;}
.rec-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:14px;}
.rec-item {background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.24); border-radius:22px; padding:14px; backdrop-filter:blur(14px);}
.rec-item b {display:block; font-size:14px; margin-bottom:4px;}
.rec-item span {font-size:13px; opacity:.9;}
.breathing-card {background:rgba(255,255,255,.78); border:1px solid rgba(226,232,240,.9); border-radius:30px; padding:22px; margin-bottom:18px; box-shadow:0 16px 45px rgba(15,23,42,.08);}
.breathing-circle {width:112px; height:112px; margin:10px auto 8px auto; border-radius:50%; background:radial-gradient(circle,rgba(37,99,235,.18),rgba(6,182,212,.38)); border:2px solid rgba(37,99,235,.25); animation:breathe 4s ease-in-out infinite;}
@keyframes breathe {0%,100%{transform:scale(.82); opacity:.72} 50%{transform:scale(1.1); opacity:1}}
.disclaimer-box {background:#f8fafc; border:1px solid #e2e8f0; border-radius:20px; padding:13px 16px; color:#64748b; font-size:13px; margin-bottom:18px;}
@media(max-width:900px){.rec-grid{grid-template-columns:1fr}.rec-title{font-size:25px}}

</style>
""", unsafe_allow_html=True)

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "stress_model.pkl"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.pkl"
SELECTED_PATH = ARTIFACT_DIR / "selected_features.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"
BACKGROUND_PATH = ARTIFACT_DIR / "shap_background.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.pkl"
SAMPLES_PATH = ARTIFACT_DIR / "prepared_samples.csv"

@st.cache_resource
def load_artifacts():
    needed = [MODEL_PATH, FEATURES_PATH, LABEL_ENCODER_PATH, BACKGROUND_PATH, METRICS_PATH, SAMPLES_PATH]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        return None, missing
    return {
        "model": joblib.load(MODEL_PATH),
        "feature_columns": joblib.load(FEATURES_PATH),
        "selected_features": joblib.load(SELECTED_PATH) if SELECTED_PATH.exists() else None,
        "label_encoder": joblib.load(LABEL_ENCODER_PATH),
        "background": joblib.load(BACKGROUND_PATH),
        "metrics": joblib.load(METRICS_PATH),
        "samples": pd.read_csv(SAMPLES_PATH),
    }, []

artifacts, missing = load_artifacts()

st.markdown("""
<div class="hero">
  <div class="orb one"></div><div class="orb two"></div>
  <div class="hero-content">
    <span class="pill">🫀 Physiological Signals</span><span class="pill">🌲 Random Forest</span><span class="pill">🔎 SHAP Explainability</span>
    <h1>Stress Detection Intelligence</h1>
    <p>Interactive model dashboard for detecting stress from heart rate, electrodermal activity, temperature, and movement signals. The app shows the prediction, confidence, true-label check, and a local feature explanation for the selected sample.</p>
  </div>
</div>
""", unsafe_allow_html=True)

if artifacts is None:
    st.error("Model files are not ready yet. First run: python train_and_export_model.py")
    st.write("Missing files:", missing)
    st.stop()

model = artifacts["model"]
feature_columns = artifacts["feature_columns"]
selected_features = artifacts["selected_features"]
label_encoder = artifacts["label_encoder"]
background = artifacts["background"]
metrics = artifacts["metrics"]
samples = artifacts["samples"]

class_names = [str(c) for c in label_encoder.classes_]

def pretty_label(label):
    s = str(label)
    if s == "2":
        return "Stress"
    if s == "1":
        return "Non-stress"
    return s

def is_stress_label(label):
    return str(label) == "2" or str(label).lower() == "stress"


def get_wellness_plan(predicted_label, stress_probability):
    if is_stress_label(predicted_label):
        if stress_probability >= 0.85:
            return {"css":"rec-stress-high","level":"High stress signal","title":"Recovery plan recommended","message":"The selected physiological pattern looks strongly stress-related. A short reset can make the app more helpful after prediction.","steps":[("Pause","Stop the current task for 2–5 minutes."),("Breathe","Take five slow deep breaths."),("Reduce load","Move away from noise, screen, or multitasking."),("Hydrate","Drink water and relax your shoulders."),("Check pattern","If high stress repeats often, discuss it with a specialist."),("Monitor","Run another sample later and compare the result.")]}
        if stress_probability >= 0.70:
            return {"css":"rec-stress-mid","level":"Moderate stress signal","title":"Short reset suggested","message":"The model detects elevated stress probability. The recommendation is simple and educational, not a medical diagnosis.","steps":[("Micro-break","Take a 2-minute pause."),("Stretch","Relax neck, shoulders, and hands."),("Breathing","Try slow inhale and exhale cycles."),("Environment","Lower distractions around you."),("Workload","Continue with one task at a time."),("Re-check","Observe if the next sample becomes stable.")]}
        return {"css":"rec-stress-low","level":"Mild stress signal","title":"Light wellness action","message":"The model output is close to stress but not extreme. A small pause may be enough.","steps":[("Pause","Take a short break."),("Posture","Sit comfortably and relax shoulders."),("Hydration","Drink water if needed."),("Focus","Avoid multitasking for a few minutes."),("Movement","Do a quick stretch."),("Track","Compare with another sample.")]}
    return {"css":"rec-stable","level":"Stable signal","title":"Healthy rhythm detected","message":"The selected physiological pattern looks stable according to the model. The app can still suggest healthy maintenance habits.","steps":[("Keep pace","Continue your current rhythm."),("Short breaks","Use regular breaks to prevent fatigue."),("Hydrate","Maintain water intake."),("Posture","Keep shoulders and neck relaxed."),("Routine","Balance focus time and rest."),("Monitor","Check another sample if needed.")]}


if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0

with st.sidebar:
    st.markdown("### Control Center")
    max_idx = len(samples) - 1
    chosen = st.number_input("Sample row", min_value=0, max_value=max_idx, value=int(st.session_state.sample_index), step=1)
    st.session_state.sample_index = int(chosen)
    if st.button("🎲 Load random sample", use_container_width=True):
        st.session_state.sample_index = int(np.random.randint(0, max_idx + 1))
        st.rerun()
    edit_mode = st.toggle("Manual edit mode", value=False)
    st.markdown("---")
    st.markdown("### Model")
    st.write("Random Forest classifier")
    st.write("Local SHAP explanation")

sample_row = samples.iloc[[st.session_state.sample_index]].copy()
true_label = sample_row["label"].iloc[0] if "label" in sample_row.columns else None
input_df = sample_row.drop(columns=["label"], errors="ignore")
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

important_inputs = [
    "HR_mean", "HR_min", "HR_max", "HR_median",
    "EDA_mean", "EDA_min", "EDA_max", "EDA_median",
    "TEMP_mean", "TEMP_min", "TEMP_max", "TEMP_median",
    "ACC_Z_min", "EDA_mean_log", "EDA_max_log", "acc_wrist_mag",
]
important_inputs = [c for c in important_inputs if c in input_df.columns]

# Top KPI strip
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Test F1-macro</div><div class="value">{metrics.get('test_f1_macro', 0):.4f}</div></div>
  <div class="kpi"><div class="label">Test accuracy</div><div class="value">{metrics.get('test_accuracy', 0):.4f}</div></div>
  <div class="kpi"><div class="label">ROC-AUC</div><div class="value">{metrics.get('test_roc_auc', 0):.4f}</div></div>
  <div class="kpi"><div class="label">PR-AUC</div><div class="value">{metrics.get('test_pr_auc', 0):.4f}</div></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, .9], gap="large")

with left:
    st.markdown('<div class="glass-card"><div class="section-label">Input</div><h3>Selected physiological sample</h3><p class="muted">One prepared dataset row is passed to the trained model. You can choose a sample row or adjust key values manually.</p>', unsafe_allow_html=True)
    if true_label is not None:
        st.info(f"Dataset row: {st.session_state.sample_index}  ·  True label: {true_label} ({pretty_label(true_label)})")
    if edit_mode:
        cols = st.columns(2)
        for i, col in enumerate(important_inputs):
            val = float(input_df[col].iloc[0])
            with cols[i % 2]:
                input_df.loc[input_df.index[0], col] = st.number_input(col, value=val, format="%.5f")
    else:
        preview_cols = important_inputs[:6]
        signal_html = '<div class="signal-grid">'
        for col in preview_cols:
            signal_html += f'<div class="signal"><div class="name">{col}</div><div class="value">{float(input_df[col].iloc[0]):.3f}</div></div>'
        signal_html += '</div><hr class="soft">'
        st.markdown(signal_html, unsafe_allow_html=True)
        st.dataframe(input_df[important_inputs].T.rename(columns={input_df.index[0]: "value"}), use_container_width=True, height=360)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card"><div class="section-label">Overview</div><h3>Model health</h3><p class="muted">The app uses the exported Week 3 Random Forest pipeline and evaluates results on a hold-out test set.</p>', unsafe_allow_html=True)
    if "confusion_matrix" in metrics:
        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=[f"True {pretty_label(c)}" for c in class_names], columns=[f"Pred {pretty_label(c)}" for c in class_names])
        st.dataframe(cm_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Run prediction", type="primary", use_container_width=True)

if predict_clicked:
    prediction_encoded = int(model.predict(input_df)[0])
    predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_confidence = float(probabilities[prediction_encoded])

    stress_card = "result-stress" if is_stress_label(predicted_label) else "result-normal"
    emoji = "⚠️" if is_stress_label(predicted_label) else "✅"

    st.markdown(f"""
    <div class="result-card {stress_card}">
      <div class="result-title">{emoji} Prediction result</div>
      <p class="big-number">{pretty_label(predicted_label)}</p>
      <p class="confidence">Predicted label: <b>{predicted_label}</b> · Confidence: <b>{predicted_confidence:.2%}</b></p>
    </div>
    """, unsafe_allow_html=True)

    check_col, prob_col = st.columns([.82, 1.18], gap="large")
    with check_col:
        st.markdown('<div class="glass-card"><div class="section-label">Validation</div><h3>Prediction check</h3>', unsafe_allow_html=True)
        if true_label is not None and not edit_mode:
            if str(predicted_label) == str(true_label):
                st.markdown('<span class="check-ok">Correct on this sample</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="check-bad">Incorrect on this sample</span>', unsafe_allow_html=True)
            st.write(f"True label: **{true_label} ({pretty_label(true_label)})**")
            st.write(f"Predicted: **{predicted_label} ({pretty_label(predicted_label)})**")
        else:
            st.write("Correctness check is available for original dataset rows.")
        st.markdown('</div>', unsafe_allow_html=True)
    with prob_col:
        st.markdown('<div class="glass-card"><div class="section-label">Confidence</div><h3>Class probabilities</h3>', unsafe_allow_html=True)
        for cls, prob in zip(class_names, probabilities):
            st.markdown(f"""
            <div class="prob-row">
              <div class="prob-label"><span>{pretty_label(cls)} · label {cls}</span><span>{prob:.2%}</span></div>
              <div class="prob-bg"><div class="prob-fill" style="width:{prob*100:.2f}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    stress_probability = 0.0
    for cls, prob in zip(class_names, probabilities):
        if is_stress_label(cls):
            stress_probability = float(prob)
            break

    plan = get_wellness_plan(predicted_label, stress_probability)
    rec_items = "".join([
        f'<div class="rec-item"><b>{title}</b><span>{text}</span></div>'
        for title, text in plan["steps"]
    ])
    st.markdown(f"""
    <div class="recommend-card {plan['css']}">
      <div class="rec-content">
        <div class="rec-kicker">Smart wellness recommendation</div>
        <div class="rec-title">{plan['title']}</div>
        <div class="rec-subtitle"><b>{plan['level']}</b> · {plan['message']}</div>
        <div class="rec-grid">{rec_items}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🫁 30-second breathing reset", expanded=False):
        st.markdown("""
        <div class="breathing-card">
          <div class="breathing-circle"></div>
          <p style="text-align:center; color:#334155; font-weight:800; margin:0;">Inhale slowly · hold · exhale slowly</p>
          <p style="text-align:center; color:#64748b; margin-top:6px;">This is an educational wellness suggestion, not a medical treatment.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="disclaimer-box">Educational disclaimer: this dashboard is a machine learning demo for coursework. It does not provide medical diagnosis or medical advice.</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-label">Explainability</div><h3>Local SHAP explanation</h3><p class="muted">This chart explains the selected sample only. Features at the top had the strongest influence on the model output.</p>', unsafe_allow_html=True)
    try:
        transformed_input = model[:-1].transform(input_df)
        final_estimator = model.named_steps["classifier"]
        names = selected_features if selected_features else [f"feature_{i}" for i in range(transformed_input.shape[1])]
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(transformed_input)
        expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            vals = shap_values[prediction_encoded][0]
            base_val = expected_value[prediction_encoded] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        elif len(np.array(shap_values).shape) == 3:
            vals = shap_values[0, :, prediction_encoded]
            base_val = expected_value[prediction_encoded] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            vals = shap_values[0]
            base_val = expected_value[prediction_encoded] if isinstance(expected_value, (list, np.ndarray)) else expected_value

        exp = shap.Explanation(values=vals, base_values=base_val, data=transformed_input[0], feature_names=names)
        fig = plt.figure(figsize=(11.5, 6.2))
        shap.plots.waterfall(exp, show=False, max_display=12)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        top = pd.DataFrame({"Feature": names, "SHAP value": vals, "Absolute impact": np.abs(vals)})
        top = top.sort_values("Absolute impact", ascending=False).head(10)
        st.caption("Top influential features for this sample")
        st.dataframe(top, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning("SHAP plot could not be generated for this sample.")
        st.exception(e)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="glass-card"><div class="section-label">Ready</div><h3>Choose a sample and run prediction</h3><p class="muted">The dashboard will show the predicted stress state, confidence, validation check, probability bars, and local SHAP feature explanation.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Team 7 · Physiological Stress Detection App</div>', unsafe_allow_html=True)
