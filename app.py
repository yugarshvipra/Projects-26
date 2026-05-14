"""
app.py — ATS Resume Analyzer & Feedback System
Run with:  streamlit run app.py
"""

import io
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from security import validate_upload, FileValidationError
from resume_parser import parse_resume
from analyzer import analyze, AnalysisResult, ATSScore

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)


# ── Custom CSS ────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        /* ── App Background ── */
        .stApp {
            background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%);
            color: #e8eaf6;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1d2e 0%, #141625 100%);
            border-right: 1px solid #2d3057;
        }
        section[data-testid="stSidebar"] * {
            color: #c5cae9 !important;
        }

        /* ── Header ── */
        .ats-header {
            font-family: 'DM Serif Display', serif;
            font-size: 2.8rem;
            font-weight: 400;
            background: linear-gradient(90deg, #7c83f7, #a78bfa, #e879f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .ats-subheader {
            font-size: 1rem;
            color: #7986cb;
            margin-bottom: 2rem;
            letter-spacing: 0.04em;
        }

        /* ── Metric Cards ── */
        .metric-card {
            background: linear-gradient(135deg, #1e2138 0%, #252847 100%);
            border: 1px solid #3d4280;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin: 0.4rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(124,131,247,0.2);
        }
        .metric-label {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #7986cb;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-family: 'DM Serif Display', serif;
            font-size: 2.2rem;
            color: #e8eaf6;
            line-height: 1;
        }
        .metric-bar-bg {
            height: 6px;
            background: #2d3057;
            border-radius: 4px;
            margin-top: 0.7rem;
        }
        .metric-bar-fill {
            height: 6px;
            border-radius: 4px;
        }

        /* ── Keyword Pills ── */
        .pill-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.8rem 0;
        }
        .pill-matched {
            background: rgba(74, 222, 128, 0.12);
            border: 1px solid rgba(74, 222, 128, 0.4);
            color: #4ade80;
            border-radius: 20px;
            padding: 0.25rem 0.85rem;
            font-size: 0.82rem;
            font-weight: 500;
        }
        .pill-missing {
            background: rgba(248, 113, 113, 0.10);
            border: 1px solid rgba(248, 113, 113, 0.35);
            color: #f87171;
            border-radius: 20px;
            padding: 0.25rem 0.85rem;
            font-size: 0.82rem;
            font-weight: 500;
        }
        .pill-neutral {
            background: rgba(124, 131, 247, 0.10);
            border: 1px solid rgba(124, 131, 247, 0.3);
            color: #7c83f7;
            border-radius: 20px;
            padding: 0.25rem 0.85rem;
            font-size: 0.82rem;
            font-weight: 500;
        }

        /* ── Section Labels ── */
        .section-title {
            font-family: 'DM Serif Display', serif;
            font-size: 1.35rem;
            color: #c5cae9;
            margin: 1.6rem 0 0.6rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid #2d3057;
        }

        /* ── Tip / Alert Boxes ── */
        .tip-box {
            background: rgba(124,131,247,0.07);
            border-left: 3px solid #7c83f7;
            border-radius: 0 10px 10px 0;
            padding: 0.75rem 1rem;
            margin: 0.4rem 0;
            font-size: 0.9rem;
            color: #c5cae9;
        }
        .issue-box {
            background: rgba(248,113,113,0.06);
            border-left: 3px solid #f87171;
            border-radius: 0 10px 10px 0;
            padding: 0.75rem 1rem;
            margin: 0.4rem 0;
            font-size: 0.9rem;
            color: #fca5a5;
        }
        .positive-box {
            background: rgba(74,222,128,0.06);
            border-left: 3px solid #4ade80;
            border-radius: 0 10px 10px 0;
            padding: 0.75rem 1rem;
            margin: 0.4rem 0;
            font-size: 0.9rem;
            color: #86efac;
        }

        /* ── Upload Area ── */
        .stFileUploader > div {
            background: rgba(30, 33, 56, 0.7) !important;
            border: 2px dashed #3d4280 !important;
            border-radius: 12px !important;
        }

        /* ── Text Area ── */
        .stTextArea textarea {
            background: #1e2138 !important;
            border: 1px solid #3d4280 !important;
            color: #e8eaf6 !important;
            border-radius: 10px !important;
            font-family: 'DM Sans', sans-serif !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, #7c83f7, #a78bfa) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.6rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            letter-spacing: 0.03em !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 24px rgba(124,131,247,0.4) !important;
        }

        /* ── Divider ── */
        hr { border-color: #2d3057 !important; }

        /* ── Verdict banner ── */
        .verdict-banner {
            background: linear-gradient(135deg, #1e2138, #252847);
            border: 1px solid #3d4280;
            border-radius: 14px;
            padding: 1.2rem 1.6rem;
            margin: 1rem 0;
            font-size: 1.05rem;
            color: #c5cae9;
            font-weight: 500;
        }

        /* Hide default Streamlit branding in main content */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Plotly Gauge ──────────────────────────────────────────────────────────────
def _render_gauge(score: float) -> go.Figure:
    if score >= 75:
        color = "#4ade80"
    elif score >= 50:
        color = "#facc15"
    elif score >= 30:
        color = "#fb923c"
    else:
        color = "#f87171"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "%",
                "font": {"size": 42, "color": "#e8eaf6", "family": "DM Serif Display"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#3d4280",
                    "tickfont": {"color": "#7986cb", "size": 11},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#1e2138",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "rgba(248,113,113,0.08)"},
                    {"range": [30, 55], "color": "rgba(251,146,60,0.08)"},
                    {"range": [55, 75], "color": "rgba(250,204,21,0.08)"},
                    {"range": [75, 100],"color": "rgba(74,222,128,0.08)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=10),
        height=280,
        font={"color": "#e8eaf6"},
    )
    return fig


# ── Score Breakdown Radar ─────────────────────────────────────────────────────
def _render_radar(ats_score: ATSScore) -> go.Figure:
    categories = ["Content Match", "Keywords", "Formatting", "Completeness", "Contact"]
    values = [
        ats_score.content_match,
        ats_score.keyword_coverage,
        ats_score.formatting,
        ats_score.completeness,
        ats_score.contact_info,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(124,131,247,0.18)",
            line={"color": "#7c83f7", "width": 2},
            marker={"color": "#a78bfa", "size": 7},
            name="Your Resume",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[80, 80, 80, 80, 80, 80],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(74,222,128,0.05)",
            line={"color": "#4ade80", "width": 1.5, "dash": "dot"},
            name="Target (80%)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color="#7986cb",
                gridcolor="#2d3057",
                tickfont={"size": 9, "color": "#7986cb"},
            ),
            angularaxis=dict(color="#7986cb", gridcolor="#2d3057"),
        ),
        showlegend=True,
        legend=dict(
            font={"color": "#c5cae9", "size": 11},
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=40, b=40),
        height=340,
        font={"color": "#e8eaf6"},
    )
    return fig


# ── Helper: Metric Card ───────────────────────────────────────────────────────
def _metric_card(label: str, value: float, color: str) -> None:
    bar_pct = int(value)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.0f}%</div>
            <div class="metric-bar-bg">
                <div class="metric-bar-fill"
                     style="width:{bar_pct}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Helper: Pill Tags ─────────────────────────────────────────────────────────
def _render_pills(items: list[str], style: str) -> None:
    pills = "".join(
        f'<span class="pill-{style}">{item}</span>' for item in items
    )
    st.markdown(f'<div class="pill-container">{pills}</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'DM Serif Display\',serif; font-size:1.5rem; '
            'color:#7c83f7; margin-bottom:0.5rem;">🎯 ATS Analyzer</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("### 📖 How It Works")
        st.markdown("""
1. **Upload** your resume (PDF or DOCX, ≤ 5 MB)
2. **Paste** the job description you're targeting
3. Click **Analyze** to run the full ATS scan
4. Review your **score**, **keyword gaps**, and **formatting tips**
        """)

        st.markdown("---")
        st.markdown("### 🔒 Privacy & Security")
        st.markdown("""
- All processing is **in-memory only** — your file is **never saved to disk**
- No data is transmitted to third-party services
- PII is detected and flagged, not stored
- Session data is cleared when you close the tab
        """)

        st.markdown("---")
        st.markdown("### 📊 Score Breakdown")
        st.markdown("""
| Component | Weight |
|-----------|--------|
| Content Match | 40% |
| Keyword Coverage | 30% |
| Formatting | 15% |
| Section Completeness | 10% |
| Contact Info | 5% |
        """)

        st.markdown("---")
        st.markdown("### ✅ ATS Best Practices")
        st.markdown("""
- Use standard section headings
- Avoid tables and multi-column layouts
- Mirror keywords from the job description
- Quantify achievements with numbers
- Save as `.docx` or text-based `.pdf`
- Keep to 1–2 pages
        """)


# ── Results Rendering ─────────────────────────────────────────────────────────
def _render_results(result: AnalysisResult) -> None:
    score = result.score

    # ── Verdict Banner ────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="verdict-banner">🏆 {result.verdict}</div>',
        unsafe_allow_html=True,
    )

    # ── Row 1: Gauge + Radar ──────────────────────────────────────────────────
    col_gauge, col_radar = st.columns([1, 1.3])

    with col_gauge:
        st.markdown('<div class="section-title">Overall ATS Score</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            _render_gauge(score.overall),
            use_container_width=True,
            key="gauge",
        )

    with col_radar:
        st.markdown('<div class="section-title">Score Breakdown</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            _render_radar(score),
            use_container_width=True,
            key="radar",
        )

    st.markdown("---")

    # ── Row 2: 5 metric cards ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Dimension Scores</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _metric_card("Content Match", score.content_match, "#7c83f7")
    with c2:
        _metric_card("Keywords", score.keyword_coverage, "#a78bfa")
    with c3:
        _metric_card("Formatting", score.formatting, "#4ade80")
    with c4:
        _metric_card("Completeness", score.completeness, "#facc15")
    with c5:
        _metric_card("Contact Info", score.contact_info, "#fb923c")

    st.markdown("---")

    # ── Row 3: Keywords ───────────────────────────────────────────────────────
    kw_col1, kw_col2 = st.columns(2)

    with kw_col1:
        st.markdown('<div class="section-title">✅ Matched Keywords</div>',
                    unsafe_allow_html=True)
        if result.matched_keywords:
            _render_pills(result.matched_keywords[:25], "matched")
        else:
            st.info("No keyword matches found.")

    with kw_col2:
        st.markdown('<div class="section-title">❌ Missing Keywords</div>',
                    unsafe_allow_html=True)
        if result.missing_keywords:
            _render_pills(result.missing_keywords[:25], "missing")
            st.caption(
                f"{len(result.missing_keywords)} JD keywords absent from your resume. "
                "Add the most relevant ones naturally."
            )
        else:
            st.success("🎉 Great — your resume covers all key JD terms!")

    st.markdown("---")

    # ── Row 4: Formatting audit ───────────────────────────────────────────────
    fmt_col1, fmt_col2 = st.columns(2)

    with fmt_col1:
        st.markdown('<div class="section-title">⚠️ Formatting Issues</div>',
                    unsafe_allow_html=True)
        if result.formatting_issues:
            for issue in result.formatting_issues:
                st.markdown(
                    f'<div class="issue-box">{issue}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="positive-box">✅ No critical formatting issues found.</div>',
                unsafe_allow_html=True,
            )

    with fmt_col2:
        st.markdown('<div class="section-title">✅ Formatting Wins</div>',
                    unsafe_allow_html=True)
        for pos in result.formatting_positives:
            st.markdown(
                f'<div class="positive-box">{pos}</div>',
                unsafe_allow_html=True,
            )

        if result.missing_sections:
            st.markdown('<div class="section-title">📋 Missing Sections</div>',
                        unsafe_allow_html=True)
            _render_pills(result.missing_sections, "neutral")

    st.markdown("---")

    # ── Row 5: Improvement Tips ───────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 Actionable Recommendations</div>',
                unsafe_allow_html=True)
    for tip in result.improvement_tips:
        st.markdown(
            f'<div class="tip-box">{tip}</div>',
            unsafe_allow_html=True,
        )

    # ── Row 6: Resume Top Terms ───────────────────────────────────────────────
    if result.resume_top_terms:
        st.markdown('<div class="section-title">🔍 Your Resume\'s Top Terms</div>',
                    unsafe_allow_html=True)
        _render_pills(result.resume_top_terms, "neutral")


# ── Parsed Resume Details Expander ───────────────────────────────────────────
def _render_parsed_details(parsed) -> None:
    with st.expander("📄 Parsed Resume Details (click to expand)", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**📇 Contact Info Detected**")
            st.write(f"📧 Emails: {', '.join(parsed.emails) if parsed.emails else '—'}")
            st.write(f"📞 Phones: {', '.join(parsed.phones) if parsed.phones else '—'}")
            st.write(
                f"🔗 LinkedIn: {', '.join(parsed.linkedin_urls) if parsed.linkedin_urls else '—'}"
            )
            st.write(
                f"🐙 GitHub: {', '.join(parsed.github_urls) if parsed.github_urls else '—'}"
            )
            st.write(f"👤 Name (detected): {parsed.name or '—'}")

        with cols[1]:
            st.markdown("**📐 Document Stats**")
            st.write(f"📝 Word Count: {parsed.word_count:,}")
            st.write(f"🔤 Character Count: {parsed.char_count:,}")
            st.write(
                f"📑 Sections Found: "
                f"{', '.join(parsed.sections_found) if parsed.sections_found else '—'}"
            )

        st.markdown("**🔒 PII Detected (categories only)**")
        if parsed.pii_inventory:
            pii_items = " | ".join(
                f"{k}: {v}" for k, v in parsed.pii_inventory.items()
            )
            st.markdown(
                f'<div class="tip-box">{pii_items}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "These PII categories were detected in your document. "
                "Actual values are never stored or logged by this system."
            )
        else:
            st.write("No common PII categories detected.")

        if parsed.skills:
            st.markdown("**🛠 Skills Detected**")
            _render_pills(parsed.skills[:30], "neutral")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()
    _render_sidebar()

    # Header
    st.markdown(
        '<div class="ats-header">ATS Resume Analyzer</div>'
        '<div class="ats-subheader">'
        'Parse · Score · Optimise — get past the bots and in front of humans'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Input Section ─────────────────────────────────────────────────────────
    input_col, jd_col = st.columns([1, 1.1], gap="large")

    with input_col:
        st.markdown("### 📎 Upload Resume")
        uploaded_file = st.file_uploader(
            "PDF or DOCX (max 5 MB)",
            type=["pdf", "docx"],
            help="Your file is processed entirely in-memory and never saved to disk.",
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.success(f"✅ File loaded: **{uploaded_file.name}**")

    with jd_col:
        st.markdown("### 📋 Job Description")
        job_description = st.text_area(
            "Paste the full job description here",
            height=220,
            placeholder="Copy and paste the job description you are applying for...",
            label_visibility="collapsed",
        )

    st.markdown("")

    # Centre the analyze button
    _, btn_col, _ = st.columns([1, 1.2, 1])
    with btn_col:
        analyze_clicked = st.button("🚀 Analyze Resume", type="primary")

    st.markdown("---")

    # ── Analysis Pipeline ─────────────────────────────────────────────────────
    if analyze_clicked:
        # Input validation
        errors = []
        if not uploaded_file:
            errors.append("Please upload a resume file.")
        if not job_description or len(job_description.strip()) < 30:
            errors.append("Please paste a job description (at least 30 characters).")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        with st.spinner("🔍 Parsing resume…"):
            try:
                file_bytes = uploaded_file.read()
                # Security validation
                validate_upload(file_bytes, uploaded_file.name)
            except FileValidationError as exc:
                st.error(f"🚫 File rejected: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error reading file: {exc}")
                logger.exception("File read error")
                st.stop()

            # Parse
            parsed = parse_resume(file_bytes, uploaded_file.name)

            # Wipe raw bytes from memory immediately (security)
            del file_bytes

        if parsed.error:
            st.error(f"⚠️ {parsed.error}")
            if parsed.is_image_based:
                st.info(
                    "💡 Tip: If your PDF was created from a scan, use an OCR tool "
                    "(e.g. Adobe Acrobat, Google Drive) to convert it to a "
                    "selectable-text PDF, then re-upload."
                )
            st.stop()

        with st.spinner("🧠 Running ATS analysis…"):
            result = analyze(parsed, job_description)

        if result.error:
            st.error(f"⚠️ {result.error}")
            st.stop()

        # Render parsed details
        _render_parsed_details(parsed)

        st.markdown("")

        # Render full results dashboard
        _render_results(result)

    else:
        # Landing placeholder
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 1rem; color: #7986cb;">
                <div style="font-size:3.5rem; margin-bottom:1rem;">🎯</div>
                <div style="font-family:'DM Serif Display',serif; font-size:1.6rem;
                     color:#c5cae9; margin-bottom:0.8rem;">
                    Ready to Beat the Bots?
                </div>
                <div style="font-size:1rem; max-width:520px; margin:0 auto; line-height:1.7;">
                    Upload your resume, paste the job description, and get an
                    instant ATS compatibility score with actionable, prioritised
                    recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
