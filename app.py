# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import json
import re
import ast
from dotenv import load_dotenv

load_dotenv()

# ---- Config ----
MODEL_PATH = "models/final_logistic_pipeline.joblib"
META_PATH  = "models/model_meta.joblib"

st.set_page_config(layout="wide", page_title="Attrition Risk Demo")

# ---- Load artifacts ----
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file at {MODEL_PATH}. Place final_logistic_pipeline.joblib in models/")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing meta file at {META_PATH}. Place model_meta.joblib in models/")

    pipe = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)

    threshold = float(meta.get("threshold", 0.5))
    feature_names = meta.get("feature_names", None)
    numeric_cols = meta.get("numeric_cols", None)
    categorical_cols = meta.get("categorical_cols", None)

    return pipe, threshold, feature_names, numeric_cols, categorical_cols

try:
    pipe, THRESHOLD, FEATURE_NAMES, NUMERIC_COLS, CATEGORICAL_COLS = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

preproc = pipe.named_steps.get('preproc')
clf = pipe.named_steps.get('clf')

# ---- UI ----
st.title("Attrition Risk — Upload CSV to Rank Employees")
st.markdown(
    "Upload a CSV with the same column names used during model training. "
    "The app returns ranked attrition probabilities and per-employee top contributing features (approximate, linear contributions)."
)

# unique key to avoid duplicate-id errors on reruns
uploaded = st.file_uploader("Upload CSV file (raw employee data)", type=["csv"], key="employee_csv_uploader")
if uploaded is None:
    st.info("Upload a CSV file to start. Use the original dataset CSV for testing.")
    st.stop()

# ---- Read CSV ----
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.write("Data preview (first rows):")
st.dataframe(df.head())

# ---- Validate required columns (best-effort) ----
required_cols = []
if NUMERIC_COLS and CATEGORICAL_COLS:
    required_cols = list(NUMERIC_COLS) + list(CATEGORICAL_COLS)

missing = []
if required_cols:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"The uploaded CSV is missing these required columns: {missing}")
        if st.checkbox("Auto-fill missing columns with sensible defaults and continue"):
            for c in missing:
                if NUMERIC_COLS and c in NUMERIC_COLS:
                    df[c] = df.get(c, pd.Series(dtype=float)).astype(float).fillna(df[c].median() if c in df.columns else 0.0)
                else:
                    df[c] = df.get(c, 'Unknown')
            st.success("Attempted to fill missing columns. Re-running prediction.")
        else:
            st.stop()

# ---- Predict probabilities ----
try:
    probs = pipe.predict_proba(df)[:, 1]
except Exception as e:
    st.error(f"Model prediction failed. Likely column mismatch or invalid values. Error: {e}")
    st.stop()

df_result = df.copy()
df_result['attrition_prob'] = probs
df_result['predicted_attrition'] = (df_result['attrition_prob'] >= THRESHOLD).astype(int)
df_ranked = df_result.sort_values('attrition_prob', ascending=False).reset_index(drop=True)

# show only top 3 in preview
st.subheader("Top 3 highest-risk employees (by predicted probability)")
st.dataframe(df_ranked.head(3))

# ---- Compute linear contributions (fast, for logistic) ----
def compute_contribs(pipe, raw_df, feature_names):
    Xtr = pipe.named_steps['preproc'].transform(raw_df)
    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
    coefs = pipe.named_steps['clf'].coef_[0]
    contribs = Xtr * coefs
    return pd.DataFrame(contribs, columns=feature_names, index=raw_df.index)

# reconstruct feature names if missing
if FEATURE_NAMES is None:
    try:
        FEATURE_NAMES = []
        if NUMERIC_COLS:
            FEATURE_NAMES.extend(list(NUMERIC_COLS))
        if preproc is not None and hasattr(preproc, "transformers_"):
            for name, transformer, columns in preproc.transformers_:
                try:
                    enc = transformer
                    if hasattr(transformer, "named_steps") and 'onehot' in transformer.named_steps:
                        enc = transformer.named_steps['onehot']
                    if hasattr(enc, "get_feature_names_out"):
                        FEATURE_NAMES.extend(list(enc.get_feature_names_out(columns)))
                    else:
                        FEATURE_NAMES.extend(list(columns))
                except Exception:
                    continue
        if not FEATURE_NAMES:
            FEATURE_NAMES = None
    except Exception:
        FEATURE_NAMES = None

contrib_df = None
if FEATURE_NAMES is not None:
    try:
        contrib_df = compute_contribs(pipe, df_result, FEATURE_NAMES)
        st.subheader("Per-employee top contributing features (top 3)")
        for i in range(min(3, len(df_ranked))):
            st.markdown(f"**Rank {i+1} — Probability {df_ranked.loc[i,'attrition_prob']:.3f}**")
            idx = df_ranked.index[i]
            if idx not in contrib_df.index:
                st.warning("Index mismatch when computing contributions for this row; skipping.")
                continue
            row_pos = contrib_df.loc[idx].nlargest(5)
            row_neg = contrib_df.loc[idx].nsmallest(5)
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top positive contributors (increase risk)")
                st.table(pd.DataFrame({'feature': row_pos.index, 'contribution': row_pos.values}))
            with col2:
                st.write("Top negative contributors (decrease risk)")
                st.table(pd.DataFrame({'feature': row_neg.index, 'contribution': row_neg.values}))
    except Exception as e:
        st.warning(f"Could not compute per-row contributions: {e}")
else:
    st.info("Per-row contributions not available (feature names missing).")

# ---- OpenRouter/OpenAI client (single Generate for top 3) ----
openai_available = True
try:
    from openai import OpenAI
except Exception:
    openai_available = False

PII_COLS = {"name", "employee_name", "email", "phone", "employee_id", "emp_id", "id"}

def mask_row_by_colnames(row, pii_cols=PII_COLS):
    out = {}
    for c, v in row.items():
        if str(c).lower() in pii_cols:
            h = hashlib.sha1(str(v).encode("utf-8")).hexdigest()[:8] if not pd.isna(v) else None
            out[c] = f"<masked:{h}>" if h is not None else None
        else:
            out[c] = v
    return out

def infer_original_value_for_feat(orig_row, feat_name):
    if feat_name in orig_row.index:
        return orig_row[feat_name]
    if "_" in feat_name:
        parts = feat_name.split("_", 1)
        col = parts[0]
        if col in orig_row.index:
            return f"{col} == {orig_row[col]}"
        else:
            return f"{feat_name} (one-hot; original col not found)"
    return "<original value not available>"

def build_employee_prompt_json(orig_row, contrib_ser, top_k=5):
    masked = mask_row_by_colnames(orig_row.to_dict())
    lines = []
    lines.append("You are an HR analytics assistant. Return output ONLY as JSON (no extra text).")
    lines.append("Produce a JSON object with three fields:")
    lines.append('  - "summary": a short (2-line) executive summary string,')
    lines.append('  - "suggestions": a JSON array of exactly 4 short actionable suggestion strings (prioritized highest to lowest),')
    lines.append('  - "caveat": a short data/model caveat string.')
    lines.append("")
    lines.append("Employee context (masked):")
    shortlist = []
    for c in ["employee_id","id","JobRole","Department","Age","Gender"]:
        if c in orig_row.index:
            shortlist.append(c)
    if not shortlist:
        shortlist = list(orig_row.index[:6])
    for c in shortlist:
        lines.append(f"- {c}: {masked.get(c, orig_row.get(c,''))}")
    lines.append("")
    lines.append("Top positive contributors (increase risk):")
    for f,v in contrib_ser.nlargest(top_k).items():
        actual = infer_original_value_for_feat(orig_row, f)
        lines.append(f"- {f}: contribution={v:.4f}; actual={actual}")
    lines.append("")
    lines.append("Top negative contributors (decrease risk):")
    for f,v in contrib_ser.nsmallest(top_k).items():
        actual = infer_original_value_for_feat(orig_row, f)
        lines.append(f"- {f}: contribution={v:.4f}; actual={actual}")
    lines.append("")
    lines.append(f"Predicted attrition probability: {orig_row.get('attrition_prob',0):.3f}")
    lines.append("")
    lines.append("Tone: neutral, concise, prioritized by impact.")
    lines.append("")
    lines.append("IMPORTANT: Output only valid JSON and nothing else. Example:")
    lines.append('{"summary":"...", "suggestions":["s1","s2","s3","s4"], "caveat":"..."}')
    return "\n".join(lines)

def call_openrouter_chat(prompt, model="meta-llama/llama-3-8b-instruct", max_tokens=800, temperature=0.0):
    if not openai_available:
        return {"error": "openai package not installed"}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "Missing OPENAI_API_KEY environment variable"}
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # robust extraction
    assistant_text = None
    try:
        assistant_text = getattr(resp.choices[0].message, "content", None)
    except Exception:
        pass
    if not assistant_text:
        try:
            msg = resp.choices[0].message
            if isinstance(msg, dict) and "content" in msg:
                assistant_text = msg["content"]
        except Exception:
            pass
    if not assistant_text:
        try:
            assistant_text = getattr(resp.choices[0], "text", None)
        except Exception:
            pass
    if assistant_text is None:
        return {"error": "Unexpected response format", "raw": str(resp)[:2000]}
    return {"text": assistant_text}

# robust JSON extractor: tries direct json.loads, code-fence stripping, substring extraction, ast.literal_eval fallback
def extract_json_from_text(txt: str):
    if not isinstance(txt, str):
        return None
    cleaned = txt.strip()
    # remove common prefix
    cleaned = re.sub(r"(?i)^here is the output:\s*", "", cleaned)
    # remove leading/trailing code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # attempt direct json loads first
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # try to find the first {...} substring that looks like JSON
    m = re.search(r"\{(?:[^{}]|(?R))*\}", cleaned, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            # try ast literal eval as fallback (handles single quotes)
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass
    # try to find a Python dict-like substring
    # This tries a looser approach: locate first "{" and last "}"
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end+1]
        # replace smart quotes
        candidate = candidate.replace("“", '"').replace("”", '"').replace("'", '"')
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass
    return None

# ---- Single generate button for top 3 ----
st.markdown("### Suggestions & summaries (top 3)")
st.markdown("Click the button below to generate structured summaries for all top 3 employees.")

if contrib_df is None or FEATURE_NAMES is None:
    st.info("Per-row contributions or feature names not available; cannot generate suggestions.")
else:
    if st.button("Generate summaries for top 3"):
        outputs = []
        for i in range(min(3, len(df_ranked))):
            idx = df_ranked.index[i]
            if idx not in contrib_df.index:
                outputs.append({"i": i, "error": "contribution data missing"})
                continue
            contrib_row = contrib_df.loc[idx]
            orig_row = df_ranked.loc[i]
            prompt = build_employee_prompt_json(orig_row, contrib_row, top_k=5)
            with st.spinner(f"Generating summary for Rank {i+1}..."):
                resp = call_openrouter_chat(prompt)
            if isinstance(resp, dict) and resp.get("error"):
                outputs.append({"i": i, "error": resp.get("error"), "raw": resp.get("raw")})
                continue
            text = resp.get("text", "") if isinstance(resp, dict) else str(resp)
            # try robust extraction
            parsed = extract_json_from_text(text)
            if isinstance(parsed, dict):
                # normalize suggestions to list of strings
                summary = parsed.get("summary", "").strip()
                suggestions = parsed.get("suggestions", [])
                if isinstance(suggestions, str):
                    # split by newlines or bullets
                    suggestions = [s.strip() for s in re.split(r"[\n\r•\-]+", suggestions) if s.strip()]
                # keep at most 4, pad if needed
                suggestions = [str(s).strip() for s in suggestions][:4]
                suggestions += ["(no suggestion)"] * max(0, 4 - len(suggestions))
                caveat = parsed.get("caveat", "").strip()
                outputs.append({"i": i, "summary": summary, "suggestions": suggestions, "caveat": caveat})
            else:
                # fallback: try to interpret text as JSON string (sometimes the model returns a JSON-looking line)
                try:
                    maybe = json.loads(text.strip())
                    if isinstance(maybe, dict):
                        parsed = maybe
                        summary = parsed.get("summary", "").strip()
                        suggestions = parsed.get("suggestions", [])
                        if isinstance(suggestions, str):
                            suggestions = [s.strip() for s in re.split(r"[\n\r•\-]+", suggestions) if s.strip()]
                        suggestions = [str(s).strip() for s in suggestions][:4]
                        suggestions += ["(no suggestion)"] * max(0, 4 - len(suggestions))
                        caveat = parsed.get("caveat", "").strip()
                        outputs.append({"i": i, "summary": summary, "suggestions": suggestions, "caveat": caveat})
                        continue
                except Exception:
                    pass
                # final fallback: render raw cleaned text
                cleaned = text.replace("Here is the output:", "").strip()
                outputs.append({"i": i, "raw_text": cleaned})

        # Render outputs in identical format (all three)
        for out in outputs:
            i = out["i"]
            prob = df_ranked.loc[i, "attrition_prob"]
            title = f"Employee {i+1} — Rank {i+1} (Probability {prob:.3f})"
            with st.expander(title, expanded=True):
                if out.get("error"):
                    st.error(f"Employee {i+1}: {out['error']}")
                    if out.get("raw"):
                        st.write(out["raw"])
                    continue
                if "summary" in out:
                    st.markdown("**Executive Summary (2 lines)**")
                    st.write(out["summary"])
                    st.markdown("**Actionable Suggestions (4 prioritized bullets)**")
                    for idx, s in enumerate(out["suggestions"], start=1):
                        st.markdown(f"**{idx}.** {s}")
                    st.markdown("**Data/Model Caveat**")
                    st.write(out.get("caveat", ""))
                else:
                    # fallback raw text - still wrap in same headings so it looks consistent
                    st.markdown("**Executive Summary (raw)**")
                    st.markdown(out.get("raw_text", "(no output)"))

# ---- Download ranked CSV (top 3) ----
csv_bytes = df_ranked.head(3).to_csv(index=False).encode('utf-8')
st.download_button("Download ranked CSV (top 3)", data=csv_bytes, file_name="ranked_attrition_top3.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Notes:** Do not upload sensitive personal data without consent. This demo does not persist uploads on disk.")
