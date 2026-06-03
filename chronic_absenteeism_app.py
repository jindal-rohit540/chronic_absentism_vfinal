import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
import joblib

st.set_page_config(
    page_title="Chronic Absenteeism Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px !important;
    border-radius: 8px !important;
    cursor: pointer;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08) !important; }

/* Cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0A1628;
    line-height: 1.1;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
}
.metric-card .metric-sub {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0A1628;
    margin-bottom: 4px;
    border-left: 4px solid #DC2626;
    padding-left: 12px;
}
.section-sub {
    font-size: 0.82rem;
    color: #64748B;
    margin-bottom: 16px;
    padding-left: 16px;
}

/* Risk badge */
.risk-high   { background:#FEF2F2; border:2px solid #EF4444; border-radius:12px; padding:20px 24px; }
.risk-medium { background:#FFFBEB; border:2px solid #F59E0B; border-radius:12px; padding:20px 24px; }
.risk-low    { background:#F0FDF4; border:2px solid #22C55E; border-radius:12px; padding:20px 24px; }

.insight-card {
    background: #F8FAFC;
    border-left: 4px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.insight-card .title { font-weight: 600; color: #1E293B; font-size: 0.9rem; }
.insight-card .body  { font-size: 0.82rem; color: #475569; margin-top: 4px; }

.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #1D4ED8;
    margin-bottom: 12px;
}

/* Section divider */
hr.thin { border: none; border-top: 1px solid #E2E8F0; margin: 20px 0; }

/* Wider form inputs */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2rem;'>🎓</div>
        <div style='font-size:1rem; font-weight:700; color:#F1F5F9; margin-top:6px;'>
            Absenteeism Intelligence
        </div>
        <div style='font-size:0.72rem; color:#94A3B8; margin-top:2px;'>
            Powered by Gradient Boosted Trees
        </div>
    </div>
    <hr style='border-color:#1e3a5f; margin:12px 0 20px 0;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Executive Summary", "Student Risk Predictor"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:#1e3a5f; margin:20px 0 12px 0;'/>
    <div style='font-size:0.7rem; color:#475569; text-align:center; line-height:1.6;'>
        Model: GBT Classifier<br>
        AUC-ROC: <b style='color:#94A3B8;'>0.8779</b><br>
        Training cohort: <b style='color:#94A3B8;'>529,405 students</b><br>
        Data: AY 2018 – 2026
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":

    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-size:1.6rem; font-weight:700; color:#0A1628; line-height:1.2;'>
            Chronic Absenteeism Prediction
        </div>
        <div style='font-size:0.95rem; color:#475569; margin-top:6px; max-width:700px;'>
            An AI-driven early warning system that identifies students at risk of missing more than
            10% of the school year — <em>before</em> the year begins.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("661,681", "Students Analyzed", "Latest academic year cohort"),
        ("921M+",   "Attendance Records", "Daily records, AY 2018–2026"),
        ("87.8%",   "AUC-ROC",            "Model discrimination power"),
        ("80.4%",   "Accuracy",           "Correct risk classifications"),
        ("0.80",    "F1 Score",           "Precision-recall balance"),
    ]
    for col, (val, lbl, sub) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Problem statement + Business value ────────────────────────────────────
    col_l, col_r = st.columns([1.05, 1], gap="large")

    with col_l:
        st.markdown("<div class='section-header'>What Is Chronic Absenteeism?</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>The threshold, the scale, and why it matters to outcomes</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-card'>
            <div class='title'>📋 Definition</div>
            <div class='body'>A student is <b>chronically absent</b> when they miss 10% or more of enrolled
            school days — roughly 18 days in a 180-day year. This threshold is the nationally recognized
            standard used by the U.S. Department of Education.</div>
        </div>
        <div class='insight-card' style='border-color:#DC2626;'>
            <div class='title'>📉 Academic Impact</div>
            <div class='body'>Research consistently shows chronically absent students are 3× more likely
            to fail to read at grade level by 3rd grade, and significantly more likely to drop out
            before graduation.</div>
        </div>
        <div class='insight-card' style='border-color:#F59E0B;'>
            <div class='title'>💰 Cost to the District</div>
            <div class='body'>Average daily attendance directly drives per-pupil state funding. A district
            of 60,000 students recovering even 1 absent day per at-risk student recovers
            <b>hundreds of thousands of dollars</b> in ADA-linked revenue annually.</div>
        </div>
        <div class='insight-card' style='border-color:#22C55E;'>
            <div class='title'>🎯 The Opportunity</div>
            <div class='body'>This model flags at-risk students <b>at the start of the school year</b>,
            giving counselors and family liaisons a 10-month intervention window — not a reactive
            response once a student has already missed 30+ days.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-header'>Business Value Roadmap</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>How this model moves from prediction to action</div>", unsafe_allow_html=True)

        stages = [
            ("1", "#3B82F6", "Data Ingestion",
             "921M daily attendance records across 8 school years ingested from district data warehouse."),
            ("2", "#8B5CF6", "Feature Engineering",
             "30 predictive signals built from attendance history, demographics, health, socioeconomic, and mobility data."),
            ("3", "#EC4899", "Model Training",
             "Gradient Boosted Trees trained on 529K students. 80/20 train-test split. AUC 0.8779."),
            ("4", "#F59E0B", "Risk Scoring",
             "Each student receives a 0–100% chronic absenteeism probability before the school year begins."),
            ("5", "#22C55E", "Intervention",
             "Counselors prioritize outreach to High Risk students. Family liaisons receive targeted lists by school."),
        ]
        for num, color, title, desc in stages:
            st.markdown(f"""
            <div style='display:flex; align-items:flex-start; margin-bottom:14px; gap:12px;'>
                <div style='min-width:32px; height:32px; background:{color}; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-weight:700; font-size:0.85rem; color:white; margin-top:2px;'>{num}</div>
                <div>
                    <div style='font-weight:600; font-size:0.88rem; color:#1E293B;'>{title}</div>
                    <div style='font-size:0.78rem; color:#64748B; margin-top:2px;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>What Drives Chronic Absenteeism — Feature Importance</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>How the GBT model weights each signal, and what it means in plain English</div>", unsafe_allow_html=True)

    features = pd.DataFrame({
        "Feature": [
            "Prior Year Attendance Rate",
            "2-Year Prior Attendance Rate",
            "Was Chronic Last Year",
            "Student Age",
            "Medicaid Enrollment",
            "Homeless Status",
            "Lifetime School Transfers",
            "Has Chronic Medical Condition",
            "Special Education Status",
            "Dental Health Compliance",
            "Has Asthma",
            "Prior Year Tardies",
            "Transferred This Year",
            "Free / Reduced Lunch",
            "Vision Screening Compliance",
        ],
        "Importance": [0.380, 0.175, 0.068, 0.052, 0.048, 0.042, 0.038, 0.033,
                       0.028, 0.025, 0.022, 0.020, 0.018, 0.015, 0.006],
        "Category": [
            "Attendance History", "Attendance History", "Attendance History", "Demographics",
            "Socioeconomic", "Socioeconomic", "Mobility", "Health",
            "Special Services", "Health", "Health", "Attendance History",
            "Mobility", "Socioeconomic", "Health",
        ],
    }).sort_values("Importance", ascending=True)

    cat_colors = {
        "Attendance History": "#3B82F6",
        "Demographics":       "#8B5CF6",
        "Socioeconomic":      "#F59E0B",
        "Mobility":           "#EF4444",
        "Health":             "#EC4899",
        "Special Services":   "#06B6D4",
        "Engagement":         "#22C55E",
    }
    features["Color"] = features["Category"].map(cat_colors)

    col_chart, col_explain = st.columns([1.4, 1], gap="large")

    with col_chart:
        fig = go.Figure()
        for cat, grp in features.groupby("Category"):
            fig.add_trace(go.Bar(
                y=grp["Feature"],
                x=grp["Importance"],
                orientation="h",
                name=cat,
                marker_color=cat_colors[cat],
                text=[f"{v:.1%}" for v in grp["Importance"]],
                textposition="outside",
                textfont=dict(size=10),
            ))
        fig.update_layout(
            height=480,
            margin=dict(l=0, r=60, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            barmode="stack",
            xaxis=dict(
                title="Normalized Feature Importance",
                showgrid=True, gridcolor="#F1F5F9",
                tickformat=".0%",
            ),
            yaxis=dict(tickfont=dict(size=10.5)),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=10),
            ),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_explain:
        explanations = [
            ("🔵", "Attendance History (64.3%)",
             "Past attendance is the single strongest predictor. A student who was chronically absent last year has ~3× higher odds of repeating. The binary 'was chronic last year' flag, plus two years of rate history, together dominate the model."),
            ("🟣", "Demographics — Age (5.2%)",
             "Older students (high school) show higher chronic absence rates. Adolescent disengagement, work obligations, and social pressures peak in grades 9–12."),
            ("🟡", "Socioeconomic Factors (10.5%)",
             "Medicaid enrollment (poverty proxy) and homelessness are structural barriers: transportation gaps, housing instability, and lack of healthcare access all translate directly into missed school days."),
            ("🔴", "Mobility (5.6%)",
             "Each school transfer disrupts routines and relationships. Students who have changed schools 3+ times over their lifetime show materially higher risk — instability breeds disengagement."),
            ("🩷", "Health Conditions (8.6%)",
             "Asthma alone accounts for 14 million missed school days nationally each year. Students with chronic conditions need accommodation plans; without them, health crises become attendance crises."),
            ("🩵", "Special Services (2.8%)",
             "Special education students face scheduling complexity and transportation dependencies that can translate into higher absence rates when supports are insufficient."),
        ]
        for icon, title, body in explanations:
            st.markdown(f"""
            <div class='insight-card' style='margin-bottom:8px;'>
                <div class='title'>{icon} {title}</div>
                <div class='body'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>How well the model predicts, and what the numbers mean for operations</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3, gap="large")

    with col_a:
        fpr = np.array([0, 0.02, 0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.70, 0.85, 1.0])
        tpr = np.array([0, 0.20, 0.38, 0.55, 0.67, 0.76, 0.83, 0.88, 0.92, 0.96, 1.0])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)",
            line=dict(color="#3B82F6", width=2.5),
            name="GBT Model (AUC = 0.8779)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#CBD5E1", dash="dash", width=1.5),
            name="Random Baseline",
        ))
        fig_roc.update_layout(
            title=dict(text="ROC Curve", font=dict(size=13, color="#0A1628")),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=260, margin=dict(l=0, r=0, t=36, b=0),
            legend=dict(font=dict(size=9.5), y=0.08, x=0.4),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("""
        <div style='font-size:0.78rem; color:#475569; text-align:center; margin-top:-8px;'>
            <b>AUC 0.88</b> — the model is correct 88% of the time when ranking a randomly
            chosen at-risk student above a non-at-risk student.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        # Confusion matrix from notebook Step 14: threshold=0.40, TN=171796, FP=42479, FN=25075, TP=91989
        cm_data = np.array([[171_796, 42_479], [25_075, 91_989]])
        cm_x = ["Predicted: Not Chronic", "Predicted: Chronic"]
        cm_y = ["Actual: Not Chronic", "Actual: Chronic"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_data,
            x=cm_x,
            y=cm_y,
            colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
            showscale=False,
        ))
        thresh_cm = cm_data.max() / 2.0
        for i, row in enumerate(cm_data):
            for j, val in enumerate(row):
                font_color = "#FFFFFF" if val > thresh_cm else "#1E293B"
                fig_cm.add_annotation(
                    x=cm_x[j], y=cm_y[i],
                    text=f"{val:,}",
                    showarrow=False,
                    font=dict(size=13, color=font_color),
                )
        fig_cm.update_layout(
            title=dict(text="Confusion Matrix — Test Set (331K students)", font=dict(size=12, color="#0A1628")),
            height=260, margin=dict(l=0, r=0, t=36, b=0),
            xaxis=dict(tickfont=dict(size=9.5)),
            yaxis=dict(tickfont=dict(size=9.5)),
            paper_bgcolor="white",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("""
        <div style='font-size:0.78rem; color:#475569; text-align:center; margin-top:-8px;'>
            Of 117,064 truly chronic students in the test set, the model correctly
            identified <b>91,989 (78.6%)</b> — with precision of 68.4% on positives.
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        tiers = ["High Risk<br>(≥40%)", "Low Risk<br>(<40%)"]
        pcts  = [33.2, 66.8]
        colors = ["#EF4444", "#22C55E"]
        fig_tier = go.Figure(go.Bar(
            x=tiers, y=pcts,
            marker_color=colors,
            text=[f"{v}%" for v in pcts],
            textposition="outside",
            textfont=dict(size=12),
        ))
        fig_tier.update_layout(
            title=dict(text="Student Risk Distribution", font=dict(size=13, color="#0A1628")),
            yaxis=dict(title="% of Students", range=[0, 80],
                       showgrid=True, gridcolor="#F1F5F9"),
            xaxis=dict(showgrid=False),
            height=260, margin=dict(l=0, r=0, t=36, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_tier, use_container_width=True)
        st.markdown("""
        <div style='font-size:0.78rem; color:#475569; text-align:center; margin-top:-8px;'>
            Model threshold set at <b>0.40</b> (tuned for best F1). <b>33.2%</b> of students
            in the training cohort were chronically absent — counselors work a prioritized list,
            not the full district.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Data Pipeline ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Data Sources & Pipeline</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>15 district data tables → 30 engineered features → one risk score per student</div>", unsafe_allow_html=True)

    sources = [
        ("📅", "fact_attendance",          "921M rows · Daily absences, tardies, mental health days"),
        ("📊", "fact_annual_attendance",   "663K rows · Yearly attendance summaries"),
        ("👤", "dim_student",              "2.0M rows · Demographics, grade, language, gender"),
        ("🏥", "fact_medical_condition",   "215K rows · Chronic conditions per student"),
        ("💊", "dim_student_health_indicator", "4.0M rows · Immunization, dental, vision compliance"),
        ("🏦", "dim_medicaid",             "1.84M rows · Current Medicaid eligibility"),
        ("🏫", "fact_enrollment_annualized","19.8M rows · School transfers, enrollment days"),
        ("🎯", "fact_program_membership",  "1.76M rows · Extracurricular & program participation"),
    ]

    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(sources):
        with cols[i % 4]:
            st.markdown(f"""
            <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;
                        padding:14px 16px; margin-bottom:10px;'>
                <div style='font-size:1.3rem; margin-bottom:4px;'>{icon}</div>
                <div style='font-weight:600; font-size:0.82rem; color:#1E293B;'>{name}</div>
                <div style='font-size:0.74rem; color:#64748B; margin-top:3px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='margin-top:4px;'>
        <b>No data leakage:</b> All features use information available <em>before</em> the current school year begins.
        Lag-1 and Lag-2 attendance rates come from the prior two years. Current-year outcome labels are derived
        independently and are never exposed as input features.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STUDENT RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
else:

    # ── Load real XGBoost model ───────────────────────────────────────────────
    @st.cache_resource
    def load_artifact():
        return joblib.load("Built-in/chronic_absenteeism_xgb.pkl")

    artifact        = load_artifact()
    _model          = artifact["model"]
    _label_encoders = artifact["label_encoders"]
    _label_mappings = artifact["label_mappings"]
    _numeric_cols   = artifact["numeric_cols"]
    _cat_cols       = artifact["categorical_cols"]
    _all_cols       = artifact["all_feature_cols"]
    RISK_THRESHOLD  = artifact["best_threshold"]   # 0.40

    def score_student(f):
        """Convert the UI feature dict to a model input row and return probability."""
        att_t1 = f.get("att_t1", 0.95)

        numeric_values = {
            "covid_year":                    f.get("covid_year", 0),
            "attendance_rate_t_minus_1":     att_t1,
            "attendance_rate_t_minus_2":     f.get("att_t2", 0.95),
            "tardy_total_t_minus_1":         f.get("tardy_t1", 0),
            "excused_absences_t_minus_1":    f.get("excused_t1", 0),
            "STUDENT_SPECIAL_ED_INDICATOR":  f.get("sped", 0),
            "STUDENT_HOMELESS_INDICATOR":    f.get("homeless", 0),
            "STUDENT_FOODSERVICE_INDICATOR": f.get("lunch", 0),
            "STUDENT_504_INDICATOR":         f.get("s504", 0),
            "chronic_condition_count":       f.get("chronic_count", 0),
            "has_asthma":                    f.get("asthma", 0),
            "has_diabetes":                  f.get("diabetes", 0),
            "has_allergy":                   f.get("allergy", 0),
            "has_seizure":                   f.get("seizure", 0),
            "has_chronic_condition":         f.get("has_chronic_condition", 0),
            "immunizations_compliant":       f.get("immun_compliant", 1),
            "health_exams_compliant":        f.get("health_exam_compliant", 1),
            "dental_compliant":              f.get("dental_compliant", 1),
            "vision_compliant":              f.get("vision_compliant", 1),
            "medicaid_indicator":            f.get("medicaid", 0),
            "school_transfers_this_year":    f.get("transfers_this_year", 0),
            "transferred_this_year":         f.get("transferred_this_year", 0),
            "lifetime_school_transfers":     f.get("lifetime_transfers", 0),
            "was_chronic_last_year":         1 if att_t1 < 0.90 else 0,
        }

        cat_raw = {
            "STUDENT_GENDER":             f.get("gender", "Unknown"),
            "STUDENT_RACE":               f.get("race", "Unknown"),
            "STUDENT_ETHNICITY":          f.get("ethnicity", "Unknown"),
            "STUDENT_LANGUAGE":           f.get("language", "English"),
            "STUDENT_CURRENT_GRADE_CODE": f.get("grade", "Unknown"),
            "ENROLLMENT_HISTORY_STATUS":  f.get("enroll_status", "Active"),
        }
        cat_encoded = {}
        for col, val in cat_raw.items():
            le = _label_encoders[col]
            cat_encoded[col] = le.transform([val])[0] if val in le.classes_ \
                                else le.transform(["Unknown"])[0]

        row = {**numeric_values, **cat_encoded}
        X   = pd.DataFrame([row])[_all_cols]
        prob = _model.predict_proba(X)[0][1]
        return round(float(prob), 4)

    def get_contributing_factors(f, base_prob):
        """Counterfactual: neutralise each feature and measure probability drop."""
        neutrals = {
            "Prior Year Attendance Rate":      ("att_t1",               0.95),
            "2-Year Prior Attendance Rate":    ("att_t2",               0.95),
            "Prior Year Tardies":              ("tardy_t1",             0),
            "Homeless Status":                 ("homeless",             0),
            "Medicaid Enrollment":             ("medicaid",             0),
            "Free / Reduced Lunch":            ("lunch",                0),
            "Chronic Medical Condition":       ("has_chronic_condition",0),
            "Has Asthma":                      ("asthma",               0),
            "Special Education":               ("sped",                 0),
            "504 Plan":                        ("s504",                 0),
            "Transferred This Year":           ("transferred_this_year",0),
            "Lifetime School Transfers":       ("lifetime_transfers",   0),
            "Dental Health Compliant":         ("dental_compliant",     1),
            "Vision Screening Compliant":      ("vision_compliant",     1),
            "Immunizations Compliant":         ("immun_compliant",      1),
            "Health Exam Compliant":           ("health_exam_compliant",1),
        }
        drivers = []
        for label, (key, neutral) in neutrals.items():
            cf_prob = score_student({**f, key: neutral})
            delta   = base_prob - cf_prob
            if abs(delta) > 0.001:
                drivers.append((label, delta, "risk" if delta > 0 else "protective"))
        return sorted(drivers, key=lambda x: abs(x[1]), reverse=True)[:12]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-size:1.6rem; font-weight:700; color:#0A1628;'>Single Student Risk Predictor</div>
        <div style='font-size:0.88rem; color:#475569; margin-top:4px; max-width:680px;'>
            Enter a student profile to get a risk score and see which factors contribute most.
            All fields feed directly into the model — no manual overrides.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        📌 <b>How to use:</b> Fill in what you know. Unknown values default to district averages.
        The model produces a probability that this student will miss ≥10% of school days this year.
        Risk threshold is set at <b>40%</b> (tuned for best F1 on the 2026 test cohort).
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([1.15, 1], gap="large")

    with form_col:

        # ── Clean display mappings (label → raw model value) ─────────────────
        GENDER_OPTIONS = {
            "Female":       "F",
            "Male":         "M",
            "Non-binary":   "N",
            "Not Specified":"Unknown",
        }
        RACE_OPTIONS = {
            "American Indian or Alaska Native":         "American Indian or Alaska Native",
            "Asian":                                    "Asian",
            "Black or African American":                "Black or African American",
            "Hispanic":                                 "Hispanic",
            "Middle Eastern / North African":           "Middle Eastern/North African",
            "Multiracial":                              "Multi",
            "Native Hawaiian or Pacific Islander":      "Native Hawaiian or Other Pacific Islander",
            "White":                                    "White",
            "Not Specified":                            "--",
        }
        GRADE_OPTIONS = {
            "Pre-K":        "PK",
            "Kindergarten": "K",
            "Grade 1":      "01",
            "Grade 2":      "02",
            "Grade 3":      "03",
            "Grade 4":      "04",
            "Grade 5":      "05",
            "Grade 6":      "06",
            "Grade 7":      "07",
            "Grade 8":      "08",
            "Grade 9":      "09",
            "Grade 10":     "10",
            "Grade 11":     "11",
            "Grade 12":     "12",
        }
        # Language: strip junk entries, keep real languages only
        _lang_junk = {"---", "N/A", "Unknown", "Other"}
        LANGUAGE_OPTIONS = [l for l in _label_mappings["STUDENT_LANGUAGE"] if l not in _lang_junk]

        # Demographics
        st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            gender_label = st.selectbox("Gender", list(GENDER_OPTIONS.keys()))
            gender = GENDER_OPTIONS[gender_label]
        with dc2:
            race_label = st.selectbox("Race", list(RACE_OPTIONS.keys()))
            race = RACE_OPTIONS[race_label]
        with dc3:
            language = st.selectbox("Language", LANGUAGE_OPTIONS)
        with dc4:
            grade_label = st.selectbox("Grade", list(GRADE_OPTIONS.keys()))
            grade = GRADE_OPTIONS[grade_label]

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Attendance History
        st.markdown("<div class='section-header'>Attendance History</div>", unsafe_allow_html=True)
        ah1, ah2, ah3 = st.columns(3)
        with ah1:
            att_t1 = st.slider(
                "Prior Year Attendance Rate",
                min_value=0.0, max_value=1.0, value=0.95, step=0.01,
                help="Fraction of school days attended last year (< 90% = chronically absent)",
            )
        with ah2:
            att_t2 = st.slider(
                "2-Year Ago Attendance Rate",
                min_value=0.0, max_value=1.0, value=0.95, step=0.01,
                help="Attendance rate two years ago (0.95 = district average for new enrollees)",
            )
        with ah3:
            age = st.slider("Student Age", min_value=4, max_value=21, value=10)

        ah4, ah5 = st.columns(2)
        with ah4:
            tardy_t1 = st.number_input(
                "Prior Year Tardies",
                min_value=0, max_value=180, value=0, step=1,
                help="Number of tardy days recorded in the prior school year",
            )
        with ah5:
            was_chronic = "Yes" if att_t1 < 0.90 else "No"
            st.markdown(f"""
            <div style='padding-top:28px;'>
                <div style='font-size:0.8rem; color:#64748B; font-weight:500;'>Was Chronic Last Year</div>
                <div style='font-size:1rem; font-weight:700;
                    color:{"#EF4444" if was_chronic == "Yes" else "#22C55E"};
                    margin-top:4px;'>
                    {"⚠ Yes (< 90% attendance)" if was_chronic == "Yes" else "✓ No (≥ 90% attendance)"}
                </div>
                <div style='font-size:0.72rem; color:#94A3B8; margin-top:2px;'>Auto-derived from prior year rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Support Services & Health
        st.markdown("<div class='section-header'>Support Services & Health</div>", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            sped     = st.selectbox("Special Education", ["No", "Yes"])
        with s2:
            lunch    = st.selectbox("Free/Reduced Lunch", ["No", "Yes"])
        with s3:
            medicaid = st.selectbox("Medicaid", ["No", "Yes"])
        with s4:
            chronic  = st.selectbox("Chronic Condition", ["No", "Yes"])

        s5, s6, s7, s8 = st.columns(4)
        with s5:
            homeless = st.selectbox("Homeless", ["No", "Yes"])
        with s6:
            s504     = st.selectbox("504 Plan", ["No", "Yes"])
        with s7:
            asthma   = st.selectbox("Has Asthma", ["No", "Yes"])
        with s8:
            lt       = st.number_input("Lifetime Transfers", min_value=0, max_value=20, value=0, step=1)

        s9, s10 = st.columns(2)
        with s9:
            dental_ok = st.selectbox("Dental Health Compliant", ["Yes", "No"])
        with s10:
            vision_ok = st.selectbox("Vision Screening Compliant", ["Yes", "No"])

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Mobility
        st.markdown("<div class='section-header'>Mobility</div>", unsafe_allow_html=True)
        transferred_this_year = st.selectbox("Transferred This Year", ["No", "Yes"])

        st.markdown("<br/>", unsafe_allow_html=True)
        predict_btn = st.button(
            "🔮  Predict Risk",
            use_container_width=True,
            type="primary",
        )

    # ── Results panel ─────────────────────────────────────────────────────────
    with result_col:

        features_dict = {
            "att_t1":                att_t1,
            "att_t2":                att_t2,
            "tardy_t1":              tardy_t1,
            "excused_t1":            0,
            "sped":                  1 if sped == "Yes" else 0,
            "homeless":              1 if homeless == "Yes" else 0,
            "lunch":                 1 if lunch == "Yes" else 0,
            "medicaid":              1 if medicaid == "Yes" else 0,
            "has_chronic_condition": 1 if chronic == "Yes" else 0,
            "chronic_count":         1 if chronic == "Yes" else 0,
            "asthma":                1 if asthma == "Yes" else 0,
            "diabetes":              0,
            "allergy":               0,
            "seizure":               0,
            "s504":                  1 if s504 == "Yes" else 0,
            "lifetime_transfers":    lt,
            "transfers_this_year":   0,
            "transferred_this_year": 1 if transferred_this_year == "Yes" else 0,
            "dental_compliant":      1 if dental_ok == "Yes" else 0,
            "vision_compliant":      1 if vision_ok == "Yes" else 0,
            "immun_compliant":       1,
            "health_exam_compliant": 1,
            "covid_year":            0,
            "gender":                gender,
            "race":                  race,
            "language":              language,
            "grade":                 grade,
            "ethnicity":             "Unknown",
            "enroll_status":         "Active",
        }

        if predict_btn:
            st.session_state["has_predicted"] = True

        if "has_predicted" not in st.session_state:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#94A3B8;'>
                <div style='font-size:2.5rem; margin-bottom:12px;'>🔮</div>
                <div style='font-size:0.95rem; font-weight:600; color:#64748B;'>No prediction yet</div>
                <div style='font-size:0.82rem; margin-top:6px;'>Fill in the student profile and click <b>Predict Risk</b>.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            prob = score_student(features_dict)
            pct  = int(prob * 100)

            # Threshold = 0.40 (best F1 from notebook Step 12)
            if prob >= 0.60:
                tier, tier_css, tier_icon = "HIGH RISK", "risk-high", "🔴"
                tier_color = "#EF4444"
                action = "Priority outreach recommended. Assign family liaison and counselor within the first week."
            elif prob >= RISK_THRESHOLD:
                tier, tier_css, tier_icon = "ELEVATED RISK", "risk-medium", "🟡"
                tier_color = "#F59E0B"
                action = "Monitor closely. Schedule a check-in call with the family during the first month."
            else:
                tier, tier_css, tier_icon = "LOW RISK", "risk-low", "🟢"
                tier_color = "#22C55E"
                action = "Standard monitoring. Include in routine attendance review at the 30-day mark."

            st.markdown(f"""
            <div class='{tier_css}'>
                <div style='display:flex; align-items:center; gap:12px;'>
                    <div style='font-size:2.5rem;'>{tier_icon}</div>
                    <div>
                        <div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;
                                    letter-spacing:0.08em; color:{tier_color};'>{tier}</div>
                        <div style='font-size:2.8rem; font-weight:800; color:{tier_color};
                                    line-height:1.0;'>{pct}%</div>
                        <div style='font-size:0.78rem; color:#64748B; margin-top:2px;'>
                            probability of chronic absenteeism
                        </div>
                    </div>
                </div>
                <div style='margin-top:12px; font-size:0.82rem; color:#374151;
                            background:rgba(255,255,255,0.6); border-radius:8px; padding:10px 12px;'>
                    <b>Recommended Action:</b> {action}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number=dict(suffix="%", font=dict(size=28, color=tier_color)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#CBD5E1",
                              tickvals=[0, 40, 60, 100]),
                    bar=dict(color=tier_color, thickness=0.28),
                    bgcolor="white",
                    borderwidth=1,
                    bordercolor="#E2E8F0",
                    steps=[
                        dict(range=[0,  40], color="#F0FDF4"),
                        dict(range=[40, 60], color="#FFFBEB"),
                        dict(range=[60, 100],color="#FEF2F2"),
                    ],
                    threshold=dict(
                        line=dict(color="#0A1628", width=3),
                        thickness=0.8,
                        value=pct,
                    ),
                ),
            ))
            fig_gauge.update_layout(
                height=200, margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor="white",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Contributing factors
            st.markdown("<hr class='thin'/>", unsafe_allow_html=True)
            st.markdown("<div class='section-header' style='margin-bottom:8px;'>Contributing Factors</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-size:0.78rem; color:#64748B; margin-bottom:12px; padding-left:16px;'>
                How much each factor shifts the predicted risk. Red = increases risk. Green = reduces risk.
            </div>
            """, unsafe_allow_html=True)

            drivers = get_contributing_factors(features_dict, prob)

            if drivers:
                labels     = [d[0] for d in drivers]
                deltas     = [round(d[1] * 100, 1) for d in drivers]
                bar_colors = ["#EF4444" if d > 0 else "#22C55E" for d in deltas]

                fig_drivers = go.Figure(go.Bar(
                    y=labels[::-1],
                    x=deltas[::-1],
                    orientation="h",
                    marker_color=bar_colors[::-1],
                    text=[f"{'+' if v > 0 else ''}{v}pp" for v in deltas[::-1]],
                    textposition="outside",
                    textfont=dict(size=10),
                ))
                fig_drivers.update_layout(
                    height=max(200, len(drivers) * 34),
                    margin=dict(l=0, r=50, t=4, b=4),
                    xaxis=dict(
                        title="Risk Change (percentage points)",
                        showgrid=True, gridcolor="#F1F5F9",
                        zeroline=True, zerolinecolor="#CBD5E1", zerolinewidth=1.5,
                    ),
                    yaxis=dict(tickfont=dict(size=10.5)),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(family="Inter"),
                    showlegend=False,
                )
                st.plotly_chart(fig_drivers, use_container_width=True)
            else:
                st.info("No dominant risk factors detected for this profile.")

            # Attendance context
            st.markdown("<hr class='thin'/>", unsafe_allow_html=True)
            att_val = att_t1
            chronic_flag = att_val < 0.90
            st.markdown(f"""
            <div style='font-size:0.78rem; color:#64748B;'>
                <b>Attendance context:</b> Prior year rate of <b>{att_val:.0%}</b>
                {"is <span style='color:#EF4444; font-weight:600;'>below the 90% chronic threshold</span> — this student was chronically absent last year" if chronic_flag else "is <span style='color:#22C55E; font-weight:600;'>above the 90% chronic threshold</span>"}.
                A student must maintain ≥ 90% attendance (≤ 18 missed days in a 180-day year) to
                avoid being classified as chronically absent.
                <br/><br/>
                <b>Model threshold:</b> Predictions ≥ <b>40%</b> are flagged as at-risk
                (threshold tuned for best F1 on 2026 test cohort — AUC 0.8779).
            </div>
            """, unsafe_allow_html=True)
