# streamlit_app_pretty.py
"""
Prettier Streamlit dashboard for your Phase I Call Analysis.
Run:
    streamlit run streamlit_app_pretty.py [optional_audio.wav]
"""

import os
import sys
import tempfile
import json
from pathlib import Path

import streamlit as st
import plotly.express as px
import pandas as pd

# Import new pipeline
from src.pipeline import AudioPipeline
from src.config import CONFIG

# ---------- Page config & CSS ----------
st.set_page_config(page_title="INTELLISCORE", layout="wide")
st.markdown(
    """
    <style>
    .transcript-box {
        background: #0b1221;
        color: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        font-family: "Courier New", monospace;
        white-space: pre-wrap;
    }
    .small-muted { color: #7a7f87; font-size: 0.9rem; }
    .score-pill { padding:6px 10px; border-radius:999px; background:#f1f5f9; }
    .download-links a { margin-right: 12px; color: #0366d6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def save_outputs_to_dir(outdir="outputs"):
    files = {}
    for fname in ["transcript.txt", "analysis.json", "report.csv", "report.html"]:
        p = os.path.join(outdir, fname)
        if os.path.exists(p):
            files[fname] = p
    return files

def plot_score_donut(scores_dict):
    df = pd.DataFrame({
        "Metric": list(scores_dict.keys()),
        "Score": list(scores_dict.values())
    })
    color_map = {
        "Listening": "#2f86ff",     # blue
        "Communication": "#2cbf4b", # green
        "Persuasion": "#f4b400",    # yellow
        "Outcome": "#ff4b4b"        # red
    }
    fig = px.pie(df, names='Metric', values='Score', hole=0.45,
                 color='Metric', color_discrete_map=color_map)
    fig.update_traces(textinfo='none', hoverinfo='label+percent')
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
    return fig

def plot_sentiment_bar(sentiment):
    labels = ["Positive", "Neutral", "Negative"]
    vals = [0.0, 0.0, 0.0]
    lab = (sentiment or {}).get("label", "UNKNOWN").upper()
    sc = float((sentiment or {}).get("score", 0.0))
    if lab == "POSITIVE":
        vals[0] = sc
    elif lab == "NEUTRAL":
        vals[1] = sc
    elif lab == "NEGATIVE":
        vals[2] = sc
    df = pd.DataFrame({"Sentiment": labels, "Score": vals})
    fig = px.bar(df, x="Sentiment", y="Score", text="Score")
    fig.update_layout(yaxis=dict(range=[0,1]), margin=dict(l=10, r=10, t=10, b=10))
    return fig

def pretty_transcript_html(transcript, turns):
    # Use turns to colorize and show confidence
    out_lines = []
    for t in turns:
        txt = t.get("text", "")
        role_raw = t.get("role")
        speaker_default = t.get("speaker", "SPEAKER_UNKNOWN")
        
        # Determine display label
        if role_raw and role_raw.upper() in ["COLLECTOR", "DEBTOR"]:
            display_role = role_raw.upper()
        else:
            display_role = f"{t.get('speaker', 'SPEAKER')} (role undetermined)"

        conf = t.get("confidence", 0.0)
        
        # Styles
        style_extra = ""
        # Low confidence (< 0.5) gets a warning border
        # if conf < 0.6 and conf > 0:
        #    style_extra = "border-left: 3px solid #ff4b4b;"
            
        conf_badge = "" 
        # Hidden for demo clarity
        # if conf > 0:
        #    conf_badge = f"<span style='font-size:0.7em; opacity:0.6; margin-left:8px'>({int(conf*100)}% conf)</span>"
        
        if display_role == "COLLECTOR":
             out_lines.append(f'<div style="background:#073642;margin:6px 0;padding:8px;border-radius:6px;{style_extra}"><strong style="color:#9be7ff">COLLECTOR:</strong> {txt} {conf_badge}</div>')
        elif display_role == "DEBTOR":
             out_lines.append(f'<div style="background:#0f291e;margin:6px 0;padding:8px;border-radius:6px;{style_extra}"><strong style="color:#4ade80">DEBTOR:</strong> {txt} {conf_badge}</div>')
        else:
             # Unknown / Fallback
             out_lines.append(f'<div style="background:#111827;color:#e6eef8;margin:6px 0;padding:8px;border-radius:6px;{style_extra}"><strong style="color:#888">{display_role}:</strong> {txt} {conf_badge}</div>')
    return "<div>" + "\n".join(out_lines) + "</div>"

# ---------- UI: header ----------
st.title("📞 Phase II — AI Call Analysis")
st.markdown("A visual, user-friendly front-end for our analysis pipeline.")

# support passing file via argv
file_arg = None
if len(sys.argv) > 1:
    if len(sys.argv) >= 2:
        maybe = sys.argv[-1]
        if os.path.exists(maybe):
            file_arg = maybe

# sidebar controls
st.sidebar.header("Controls")
method = st.sidebar.selectbox("Transcription method", ["mock", "whisper_local"], 
                              index=0 if CONFIG.use_api == "mock" else 1)

# Dynamically update config based on selection (runtime hack for demo)
CONFIG.use_api = method

auto_analyze = st.sidebar.checkbox("Auto-analyze passed file (if any)", value=True)

# upload area
col_top_left, col_top_right = st.columns([3, 1])
with col_top_left:
    if file_arg:
        st.info(f"File passed via command: `{Path(file_arg).name}`")
    uploaded_file = st.file_uploader("Upload audio file (wav/mp3/m4a/flac/ogg) or pass via command", type=["wav", "mp3", "m4a", "flac", "ogg"])
    selected_file_path = None

    if uploaded_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        selected_file_path = tmp.name
    elif file_arg:
        selected_file_path = file_arg

    if selected_file_path:
        st.audio(read_bytes(selected_file_path))
        analyze_btn = st.button("Analyze Now")
        if auto_analyze and file_arg:
            analyze_btn = True  

        if analyze_btn:
            with st.spinner("Running analysis pipeline (Phase II)..."):
                pipeline = AudioPipeline()
                results = pipeline.process_file(selected_file_path)
                
                # Unpack results
                intent = results.get("intent")
                entities = results.get("entities", {})
                scores = results.get("scores", {})
                sentiment = results.get("sentiment", {})
                speech_emotion = results.get("speech_emotion", {})
                turns = results.get("turns", [])
                transcript = results.get("transcript", "")
                
                # Metrics / Confidence Stats
                avg_conf = sum(t.get("confidence", 0) for t in turns) / len(turns) if turns else 0

            # top summary row
            st.markdown("---")
            r1, r2 = st.columns([2, 3])
            with r1:
                st.subheader("Call Summary")
                st.write(f"**Intent:** {intent}")
                st.write("**Detected amounts:**", ", ".join(entities.get("Amount") or ["-"]))
                st.write("**Detected dates:**", ", ".join(entities.get("Date") or ["-"]))
                st.write("**Modes:**", ", ".join(entities.get("Mode") or ["-"]))
                
                st.markdown("<div class='small-muted'>Files saved in `outputs/`</div>", unsafe_allow_html=True)

            with r2:
                # charts side-by-side
                c1, c2 = st.columns([1,1])
                with c1:
                    st.subheader("Scores")
                    fig = plot_score_donut(scores)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.subheader("Sentiment")
                    fig2 = plot_sentiment_bar(sentiment)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown("**Speech emotion**:")
                    st.write(speech_emotion.get("label_pretty") if speech_emotion else "UNKNOWN")
                    if speech_emotion:
                        st.write(f"Confidence: {speech_emotion.get('score', 0):.2f}")

            st.markdown("---")

            # Transcript area
            left, right = st.columns([2.2, 1.0])
            with left:
                st.subheader("Call Details — Transcript")
                st.markdown(pretty_transcript_html(transcript, turns), unsafe_allow_html=True)
                st.markdown("<div style='margin-top:8px' class='small-muted'>Confidence scores indicate alignment/overlap quality.</div>", unsafe_allow_html=True)

            with right:
                st.subheader("Entities & Timeline")
                st.markdown("**Entities (postprocessed)**")
                st.json(entities)
                st.markdown("**Turn timeline**")
                try:
                    timeline = []
                    for i, t in enumerate(turns):
                        timeline.append({
                            "turn": i+1,
                            "role": t.get("role", "unknown"),
                            "start": t.get("start", ""),
                            "dur": f'{t.get("end", 0)-t.get("start", 0):.1f}s',
                            "conf": t.get("confidence", 0)
                        })
                    df_t = pd.DataFrame(timeline)
                    st.dataframe(df_t)
                except Exception:
                    st.write("No timeline available.")

            # downloads
            st.markdown("---")
            st.subheader("Downloads")
            saved = save_outputs_to_dir("outputs")
            if saved:
                dlc = st.columns(len(saved))
                for (fname, p), col in zip(saved.items(), dlc):
                    with col:
                        st.download_button(label=f"Download {fname}", data=read_bytes(p), file_name=fname)
            else:
                st.write("No output files found (look in outputs/).")

    else:
        st.info("Upload an audio file or pass one via the command line: `streamlit run streamlit_app_pretty.py example.wav`")

# small footer
st.markdown("---")
st.caption("Dashboard powered by Phase II new architecture. Roles inferred using language model.")
