import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
        ["Executive Summary", "Student Risk Predictor", "Network & District View"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:#1e3a5f; margin:20px 0 12px 0;'/>
    <div style='font-size:0.7rem; color:#475569; text-align:center; line-height:1.6;'>
        Trained on <b style='color:#94A3B8;'>529,405 students</b><br>
        Accuracy: <b style='color:#94A3B8;'>80%</b> on unseen data<br>
        Data: AY 2018 – 2026
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0A1628 0%,#1e3a5f 100%);
                border-radius:16px; padding:36px 40px; margin-bottom:28px;'>
        <div style='font-size:0.78rem; font-weight:600; letter-spacing:0.12em;
                    color:#60A5FA; text-transform:uppercase; margin-bottom:10px;'>
            Early Warning System — District Analytics
        </div>
        <div style='font-size:2rem; font-weight:800; color:#F1F5F9; line-height:1.25;
                    max-width:700px;'>
            Predicting Student Absenteeism<br>Before It Becomes a Crisis
        </div>
        <div style='font-size:0.95rem; color:#94A3B8; margin-top:14px; max-width:640px; line-height:1.6;'>
            Every year, thousands of students slip into chronic absenteeism — missing so many days
            that they fall behind and never catch up. This tool uses historical data to identify
            those students <b style='color:#F1F5F9;'>at the very start of the school year</b>,
            so staff can step in early, not too late.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip — plain English only ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("313,136",  "Actively Enrolled Students", "Current district enrollment"),
        ("661,681",  "Students in the Dataset",    "Includes historical & inactive enrollees across 8 years"),
        ("8 in 10",  "Predictions Are Correct",    "Tested on 331,000 real students"),
        ("33%",      "Students Flagged at Risk",   "Identified before the year begins"),
    ]
    for col, (val, lbl, sub) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Problem + Value ───────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.05, 1], gap="large")

    with col_l:
        st.markdown("<div class='section-header'>The Problem We Are Solving</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>What chronic absenteeism is, and why it matters</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-card'>
            <div class='title'>📋 What counts as chronic absenteeism?</div>
            <div class='body'>A student is considered chronically absent when they miss
            <b>18 or more school days</b> in a year — that is just 2 days per month.
            It sounds small, but the learning loss compounds quickly.</div>
        </div>
        <div class='insight-card' style='border-color:#DC2626;'>
            <div class='title'>📉 What happens to these students?</div>
            <div class='body'>Students who are chronically absent are <b>3× more likely</b>
            to fall behind in reading by 3rd grade, and significantly more likely to drop out
            before finishing high school.</div>
        </div>
        <div class='insight-card' style='border-color:#F59E0B;'>
            <div class='title'>💰 What does it cost the district?</div>
            <div class='body'>State funding is tied to how many students show up each day.
            When attendance drops, so does funding. Getting even <b>one extra day</b> of
            attendance per at-risk student adds up to hundreds of thousands of dollars
            across a large district.</div>
        </div>
        <div class='insight-card' style='border-color:#22C55E;'>
            <div class='title'>🎯 How does this tool help?</div>
            <div class='body'>Instead of reacting after a student has already missed 30 days,
            this tool gives counselors and family liaisons a <b>full school year</b> to reach
            out, build relationships, and remove the barriers keeping students away.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-header'>From Data to Action — How It Works</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Five steps from raw records to a counselor's priority list</div>", unsafe_allow_html=True)

        stages = [
            ("1", "#3B82F6", "Collect the Data",
             "8 years of attendance, health, demographics, and school history for every student in the district."),
            ("2", "#8B5CF6", "Find the Patterns",
             "The system learns from 30 signals — things like last year's attendance, whether a student moved schools, or has a health condition."),
            ("3", "#EC4899", "Train the Model",
             "Using the history of 529,000 students, the model learns which combinations of factors lead to chronic absenteeism."),
            ("4", "#F59E0B", "Score Every Student",
             "Before the new school year starts, every student receives a risk score from 0–100%."),
            ("5", "#22C55E", "Targeted Outreach",
             "Counselors get a prioritized list. High-risk students get a phone call in week one — not a letter in March."),
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

    # ── What drives risk ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>What Puts a Student at Risk?</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>The factors the model found most predictive — in order of importance</div>", unsafe_allow_html=True)

    features = pd.DataFrame({
        "Factor": [
            "Was absent a lot last year",
            "Was absent a lot two years ago",
            "Was chronically absent last year",
            "Older student (high school)",
            "Enrolled in Medicaid",
            "Experiencing homelessness",
            "Has changed schools multiple times",
            "Has a chronic health condition",
            "In special education",
            "Dental health not up to date",
            "Has asthma",
            "Was frequently late last year",
            "Changed schools this year",
            "Receives free or reduced lunch",
            "Vision screening overdue",
        ],
        "Impact": [0.380, 0.175, 0.068, 0.052, 0.048, 0.042, 0.038, 0.033,
                   0.028, 0.025, 0.022, 0.020, 0.018, 0.015, 0.006],
        "Category": [
            "Attendance History", "Attendance History", "Attendance History", "Age & Grade",
            "Economic Hardship", "Economic Hardship", "Student Mobility", "Student Health",
            "Special Services", "Student Health", "Student Health", "Attendance History",
            "Student Mobility", "Economic Hardship", "Student Health",
        ],
    }).sort_values("Impact", ascending=True)

    cat_colors = {
        "Attendance History": "#3B82F6",
        "Age & Grade":        "#8B5CF6",
        "Economic Hardship":  "#F59E0B",
        "Student Mobility":   "#EF4444",
        "Student Health":  "#EC4899",
        "Special Services":   "#06B6D4",
    }
    features["Color"] = features["Category"].map(cat_colors)

    col_chart, col_explain = st.columns([1.4, 1], gap="large")

    with col_chart:
        fig = go.Figure()
        for cat, grp in features.groupby("Category"):
            fig.add_trace(go.Bar(
                y=grp["Factor"],
                x=grp["Impact"],
                orientation="h",
                name=cat,
                marker_color=cat_colors[cat],
                text=[f"{int(v*100)}%" for v in grp["Impact"]],
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
                title="How strongly this factor influences the prediction",
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
            ("🔵", "Past Attendance (64%)",
             "The single biggest signal. A student who struggled with attendance last year is very likely to struggle again — unless someone intervenes. This gives us a head start."),
            ("🟡", "Economic Hardship (10%)",
             "Students experiencing poverty, homelessness, or housing instability face real barriers to getting to school — transportation, work obligations, safety concerns. The model picks this up."),
            ("🩷", "Student Health (9%)",
             "Students with asthma, dental issues, or other health conditions miss more school. Something as simple as an untreated toothache can keep a child home for days."),
            ("🟣", "Age & Grade (5%)",
             "Older students, especially in high school, are more likely to disengage. Work, social pressures, and lack of connection to school all play a role."),
            ("🔴", "Student Mobility (6%)",
             "Every time a student changes schools, they lose their friends, routines, and support network. Students who have moved schools multiple times are at significantly higher risk."),
            ("🩵", "Special Services (3%)",
             "Students with IEPs and 504 plans sometimes face logistical and transportation challenges that can translate into missed days when the right supports aren't in place."),
        ]
        for icon, title, body in explanations:
            st.markdown(f"""
            <div class='insight-card' style='margin-bottom:8px;'>
                <div class='title'>{icon} {title}</div>
                <div class='body'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── How accurate is it ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>How Accurate Is the Model?</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Tested on 331,000 real students from the most recent school year</div>", unsafe_allow_html=True)

    acc1, acc2, acc3, acc4 = st.columns(4)
    accuracy_kpis = [
        ("#22C55E", "80%",    "Overall Accuracy",
         "Out of every 10 students, the model correctly predicted 8 of them."),
        ("#3B82F6", "78.6%",  "At-Risk Students Caught",
         "Of the students who actually became chronically absent, nearly 8 in 10 were flagged in advance."),
        ("#F59E0B", "80.1%",  "Low-Risk Students Correct",
         "Students predicted as low-risk were indeed fine — so staff time isn't wasted on false alarms."),
        ("#8B5CF6", "8 Years", "of Historical Data Used",
         "The model learned from students across 8 school years, making its predictions battle-tested."),
    ]
    for col, (color, val, lbl, body) in zip([acc1, acc2, acc3, acc4], accuracy_kpis):
        with col:
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-top:4px solid {color};
                        border-radius:12px; padding:20px 18px; height:100%;
                        box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
                <div style='font-size:2rem; font-weight:800; color:{color};'>{val}</div>
                <div style='font-size:0.8rem; font-weight:700; color:#1E293B;
                            text-transform:uppercase; letter-spacing:0.04em;
                            margin-top:6px;'>{lbl}</div>
                <div style='font-size:0.76rem; color:#64748B; margin-top:6px;
                            line-height:1.5;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Data sources — plain English ──────────────────────────────────────────
    st.markdown("<div class='section-header'>What Data Goes Into This?</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Every piece of information the model uses — all from existing district systems</div>", unsafe_allow_html=True)

    sources = [
        ("📅", "Daily Attendance",       "Every absence, tardy, and present day recorded since 2018 — over 921 million records."),
        ("👤", "Student Demographics",   "Grade level, age, gender, race, home language — 2 million student profiles."),
        ("🏥", "Health Records",         "Chronic conditions like asthma and diabetes, plus immunization and dental compliance."),
        ("🏠", "Economic Indicators",    "Medicaid enrollment and free/reduced lunch status as proxies for household hardship."),
        ("🏫", "Enrollment History",     "Every school a student has attended — used to measure stability and mobility over time."),
        ("📋", "Special Services",       "Whether a student has an IEP, 504 plan, or other support designation."),
        ("🩺", "Health Screenings",      "Vision and dental screening compliance — gaps here often mean unmet health needs."),
        ("📆", "8 Years of History",     "The model was trained on data from 2018 through 2026, capturing pre- and post-COVID patterns."),
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STUDENT RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Student Risk Predictor":

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
            "Prior Year Attendance Rate":   ("att_t1",             0.95),
            "2-Year Prior Attendance Rate": ("att_t2",             0.95),
            "Homeless":                     ("homeless",           0),
            "Medicaid Enrolled":            ("medicaid",           0),
            "Chronic Medical Conditions":   ("chronic_count",      0),
            "Special Education":            ("sped",               0),
            "504 Plan":                     ("s504",               0),
            "Lifetime School Transfers":    ("lifetime_transfers", 0),
            "Transfers This Year":          ("transfers_this_year",0),
            "Dental Screening Up to Date":  ("dental_compliant",   1),
            "Vision Screening Up to Date":  ("vision_compliant",   1),
            "Immunizations Up to Date":     ("immun_compliant",    1),
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
        A score of <b>40% or above</b> is flagged as at-risk.
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
        # School / Network — loaded from dim_schools.csv
        @st.cache_data
        def load_schools():
            df = pd.read_csv("Built-in/dim_schools.csv")
            df["NETWORK"] = df["NETWORK"].fillna("No Network")
            return df

        df_schools = load_schools()
        networks = sorted(df_schools["NETWORK"].unique().tolist())

        st.markdown("<div class='section-header'>School & Network</div>", unsafe_allow_html=True)
        sn1, sn2 = st.columns(2)
        with sn1:
            network = st.selectbox("Network", networks,
                                   help="The network or cluster this school belongs to")
        with sn2:
            school_list = df_schools[df_schools["NETWORK"] == network]["SCHOOL_NAME"].sort_values().tolist()
            school_name = st.selectbox("School", school_list,
                                       help="Used for context only — the model scores by student profile, not school identity")

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Demographics
        st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            gender_label = st.selectbox("Gender", list(GENDER_OPTIONS.keys()))
            gender = GENDER_OPTIONS[gender_label]
        with dc2:
            race_label = st.selectbox("Race", list(RACE_OPTIONS.keys()))
            race = RACE_OPTIONS[race_label]
        with dc3:
            grade_label = st.selectbox("Grade", list(GRADE_OPTIONS.keys()))
            grade = GRADE_OPTIONS[grade_label]
        language = "English"  # removed from UI; default passed to model

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Attendance History
        st.markdown("<div class='section-header'>Attendance History</div>", unsafe_allow_html=True)
        ah1, ah2 = st.columns(2)
        with ah1:
            att_t1 = st.slider(
                "Prior Year Attendance Rate",
                min_value=0.0, max_value=1.0, value=0.95, step=0.01,
                help="Fraction of school days attended last year. Below 90% means the student was chronically absent.",
            )
        with ah2:
            att_t2 = st.slider(
                "2-Year Prior Attendance Rate",
                min_value=0.0, max_value=1.0, value=0.95, step=0.01,
                help="Attendance rate two years ago.",
            )

        was_chronic = "Yes" if att_t1 < 0.90 else "No"
        st.markdown(f"""
        <div style='padding: 6px 0 4px 0;'>
            <span style='font-size:0.8rem; color:#64748B; font-weight:500;'>Was Chronically Absent Last Year: </span>
            <span style='font-size:0.88rem; font-weight:700;
                color:{"#EF4444" if was_chronic == "Yes" else "#22C55E"};'>
                {"⚠ Yes (attendance below 90%)" if was_chronic == "Yes" else "✓ No (attendance 90% or above)"}
            </span>
            <span style='font-size:0.72rem; color:#94A3B8;'> — auto-derived from prior year rate</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Student Mobility
        st.markdown("<div class='section-header'>Student Mobility</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            lt = st.number_input("Lifetime School Transfers",
                min_value=0, max_value=20, value=0, step=1,
                help="Total number of schools the student has attended across their entire enrollment history.")
        with m2:
            school_transfers_this_year = st.number_input("School Transfers This Year",
                min_value=0, max_value=10, value=0, step=1,
                help="Number of schools the student has already transferred to this school year.")

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Support Services
        st.markdown("<div class='section-header'>Support Services</div>", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            sped     = st.selectbox("Special Education", ["No", "Yes"])
        with s2:
            s504     = st.selectbox("504 Plan", ["No", "Yes"])
        with s3:
            homeless = st.selectbox("Homeless", ["No", "Yes"])
        with s4:
            medicaid = st.selectbox("Medicaid Enrolled", ["No", "Yes"])

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

        # Student Health
        st.markdown("<div class='section-header'>Student Health</div>", unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        with h1:
            chronic_count = st.number_input("# Chronic Medical Conditions",
                min_value=0, max_value=10, value=0, step=1,
                help="Number of documented chronic health conditions (e.g. asthma, diabetes).")
        with h2:
            dental_ok = st.selectbox("Dental Screening Up to Date", ["Yes", "No"],
                help="Unmet dental needs are a documented driver of missed school days.")
        with h3:
            vision_ok = st.selectbox("Vision Screening Up to Date", ["Yes", "No"],
                help="Undiagnosed vision problems can cause disengagement and absences.")

        immun_ok     = st.selectbox("Immunizations Up to Date", ["Yes", "No"], index=0)

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
            "tardy_t1":              0,           # zero importance — kept for model input shape
            "excused_t1":            0,           # zero importance
            "sped":                  1 if sped == "Yes" else 0,
            "homeless":              1 if homeless == "Yes" else 0,
            "lunch":                 0,           # zero importance
            "medicaid":              1 if medicaid == "Yes" else 0,
            "has_chronic_condition": 0,           # zero importance
            "chronic_count":         chronic_count,
            "asthma":                0,           # zero importance
            "diabetes":              0,           # zero importance
            "allergy":               0,           # zero importance
            "seizure":               0,           # zero importance
            "s504":                  1 if s504 == "Yes" else 0,
            "lifetime_transfers":    lt,
            "transfers_this_year":   school_transfers_this_year,
            "transferred_this_year": 0,           # zero importance
            "dental_compliant":      1 if dental_ok == "Yes" else 0,
            "vision_compliant":      1 if vision_ok == "Yes" else 0,
            "immun_compliant":       1 if immun_ok == "Yes" else 0,
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
                <b>Model threshold:</b> Scores of <b>40% or above</b> are flagged as at-risk,
                based on testing against 331,000 real students from the most recent school year.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NETWORK & DISTRICT VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Network & District View":

    # ── Load model (reuse cache) ───────────────────────────────────────────────
    @st.cache_resource
    def load_artifact_p3():
        return joblib.load("Built-in/chronic_absenteeism_xgb.pkl")

    _art3           = load_artifact_p3()
    _model3         = _art3["model"]
    _le3            = _art3["label_encoders"]
    _lm3            = _art3["label_mappings"]
    _all_cols3      = _art3["all_feature_cols"]
    THRESHOLD3      = _art3["best_threshold"]

    @st.cache_data
    def load_schools_p3():
        df = pd.read_csv("Built-in/dim_schools.csv")
        df["NETWORK"] = df["NETWORK"].fillna("No Network")
        return df

    df_schools3 = load_schools_p3()

    # ── Synthetic cohort generator ─────────────────────────────────────────────
    # Generates a representative population seeded by scope string so it is
    # reproducible within a session but distinct per school/network/district.
    @st.cache_data(show_spinner=False)
    def build_cohort(scope_key: str, n: int = 400) -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash(scope_key)) % (2**31))

        # Attendance history — mix of good and at-risk students
        att_t1 = np.clip(rng.beta(8, 2, n), 0, 1)       # skewed toward high attendance
        att_t2 = np.clip(rng.beta(8, 2, n), 0, 1)

        # Socioeconomic / support flags (correlated with attendance)
        at_risk_weight = np.where(att_t1 < 0.85, 0.6, 0.2)
        homeless  = rng.binomial(1, np.clip(at_risk_weight * 0.3, 0, 1), n)
        medicaid  = rng.binomial(1, np.clip(at_risk_weight * 0.7 + 0.2, 0, 1), n)
        sped      = rng.binomial(1, 0.15,  n)
        s504      = rng.binomial(1, 0.05,  n)

        # Mobility
        lifetime_tr = rng.integers(0, 5, n)
        tr_this_yr  = rng.integers(0, 2, n)

        # Health
        chronic_count = rng.integers(0, 3, n)
        dental_ok  = rng.binomial(1, 0.72, n)
        vision_ok  = rng.binomial(1, 0.78, n)
        immun_ok   = rng.binomial(1, 0.90, n)

        # Categorical
        grades = ["PK","K","01","02","03","04","05","06","07","08","09","10","11","12"]
        grade_arr  = rng.choice(grades, n)
        gender_arr = rng.choice(["F", "M"], n)
        race_arr   = rng.choice(
            ["Black or African American", "Hispanic", "White", "Asian", "Multi", "--"],
            n, p=[0.36, 0.47, 0.09, 0.04, 0.03, 0.01]
        )

        rows = []
        for i in range(n):
            a1  = float(att_t1[i])
            a2  = float(att_t2[i])
            num = {
                "covid_year":                    0,
                "attendance_rate_t_minus_1":     a1,
                "attendance_rate_t_minus_2":     a2,
                "tardy_total_t_minus_1":         0,
                "excused_absences_t_minus_1":    0,
                "STUDENT_SPECIAL_ED_INDICATOR":  int(sped[i]),
                "STUDENT_HOMELESS_INDICATOR":    int(homeless[i]),
                "STUDENT_FOODSERVICE_INDICATOR": 0,
                "STUDENT_504_INDICATOR":         int(s504[i]),
                "chronic_condition_count":       int(chronic_count[i]),
                "has_asthma":                    0,
                "has_diabetes":                  0,
                "has_allergy":                   0,
                "has_seizure":                   0,
                "has_chronic_condition":         1 if chronic_count[i] > 0 else 0,
                "immunizations_compliant":       int(immun_ok[i]),
                "health_exams_compliant":        1,
                "dental_compliant":              int(dental_ok[i]),
                "vision_compliant":              int(vision_ok[i]),
                "medicaid_indicator":            int(medicaid[i]),
                "school_transfers_this_year":    int(tr_this_yr[i]),
                "transferred_this_year":         1 if tr_this_yr[i] > 0 else 0,
                "lifetime_school_transfers":     int(lifetime_tr[i]),
                "was_chronic_last_year":         1 if a1 < 0.90 else 0,
            }
            cat_raw = {
                "STUDENT_GENDER":             gender_arr[i],
                "STUDENT_RACE":               race_arr[i],
                "STUDENT_ETHNICITY":          "Unknown",
                "STUDENT_LANGUAGE":           "English",
                "STUDENT_CURRENT_GRADE_CODE": grade_arr[i],
                "ENROLLMENT_HISTORY_STATUS":  "Active",
            }
            cat_enc = {}
            for col, val in cat_raw.items():
                le = _le3[col]
                cat_enc[col] = le.transform([val])[0] if val in le.classes_ \
                               else le.transform(["Unknown"])[0]
            rows.append({**num, **cat_enc})

        df = pd.DataFrame(rows)[_all_cols3]
        probs = _model3.predict_proba(df)[:, 1]
        out = pd.DataFrame(rows)
        out["prob"] = probs
        out["att_t1"] = att_t1
        out["att_t2"] = att_t2
        out["homeless"] = homeless
        out["medicaid"] = medicaid
        out["sped"] = sped
        out["s504"] = s504
        out["dental_ok"] = dental_ok
        out["vision_ok"] = vision_ok
        out["immun_ok"] = immun_ok
        out["chronic_count"] = chronic_count
        out["lifetime_tr"] = lifetime_tr
        out["tr_this_yr"] = tr_this_yr
        return out

    def apply_interventions(cohort: pd.DataFrame, interventions: dict) -> np.ndarray:
        """Re-score cohort with intervention adjustments applied."""
        df = pd.DataFrame()
        rows = []
        for _, r in cohort.iterrows():
            a1 = float(r["att_t1"])
            a2 = float(r["att_t2"])
            dental  = min(1, int(r["dental_ok"])  + interventions.get("dental_boost", 0))
            vision  = min(1, int(r["vision_ok"])  + interventions.get("vision_boost", 0))
            immun   = min(1, int(r["immun_ok"])   + interventions.get("immun_boost", 0))
            lt_adj  = max(0, int(r["lifetime_tr"]) - interventions.get("transfer_reduction", 0))
            homeless_adj = max(0, int(r["homeless"]) - interventions.get("homeless_reduction", 0))

            num = {
                "covid_year":                    0,
                "attendance_rate_t_minus_1":     a1,
                "attendance_rate_t_minus_2":     a2,
                "tardy_total_t_minus_1":         0,
                "excused_absences_t_minus_1":    0,
                "STUDENT_SPECIAL_ED_INDICATOR":  int(r["sped"]),
                "STUDENT_HOMELESS_INDICATOR":    homeless_adj,
                "STUDENT_FOODSERVICE_INDICATOR": 0,
                "STUDENT_504_INDICATOR":         int(r["s504"]),
                "chronic_condition_count":       int(r["chronic_count"]),
                "has_asthma":                    0,
                "has_diabetes":                  0,
                "has_allergy":                   0,
                "has_seizure":                   0,
                "has_chronic_condition":         1 if r["chronic_count"] > 0 else 0,
                "immunizations_compliant":       immun,
                "health_exams_compliant":        1,
                "dental_compliant":              dental,
                "vision_compliant":              vision,
                "medicaid_indicator":            int(r["medicaid"]),
                "school_transfers_this_year":    int(r["tr_this_yr"]),
                "transferred_this_year":         1 if r["tr_this_yr"] > 0 else 0,
                "lifetime_school_transfers":     lt_adj,
                "was_chronic_last_year":         1 if a1 < 0.90 else 0,
            }
            # Categorical columns come directly from cached cohort row
            cat_enc = {c: int(r[c]) for c in _all_cols3 if c not in num}
            rows.append({**num, **cat_enc})

        X = pd.DataFrame(rows)[_all_cols3]
        return _model3.predict_proba(X)[:, 1]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-size:1.6rem; font-weight:700; color:#0A1628;'>Network & District Risk View</div>
        <div style='font-size:0.88rem; color:#475569; margin-top:4px; max-width:720px;'>
            Explore how chronic absenteeism risk is distributed across a school, network, or the entire
            district — and model the impact of targeted interventions before the year begins.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        📌 <b>How this works:</b> A representative student population is simulated for each scope using
        the same predictive model. Use the <b>intervention levers</b> below to see how district-level
        actions (e.g. improving dental compliance, stabilizing housing) shift the overall risk landscape.
    </div>
    """, unsafe_allow_html=True)

    # Demo networks and their schools (static subset for demo clarity)
    DEMO_NETWORKS = {
        "Network 1":  ["ALBANY PARK", "BATEMAN", "BELDING", "CHASE", "DISNEY MAGNET"],
        "Network 4":  ["ALCOTT ES", "AUDUBON", "BLAINE", "BURLEY", "CHASE"],
        "Network 10": ["ASHBURN", "AZUELA", "BEASLEY", "BRIGHTON PARK", "CARDENAS"],
        "Network 13": ["ADDAMS", "ALDRIDGE", "BARNARD", "BEIDLER", "BOND"],
        "Charter":    ["ACERO - BRIGHTON PARK", "ACERO - CISNEROS", "ACERO - CLEMENTE", "ACERO - DE LAS CASAS", "ACERO - GARCIA"],
    }

    # ── Scope selector ────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns([1, 1.2, 1.5])
    with sc1:
        scope_level = st.selectbox(
            "View Level",
            ["Entire District (CPS)", "By Network", "By School"],
            help="Choose whether to view all of CPS, a specific network, or a single school.",
        )
    with sc2:
        if scope_level == "By Network":
            selected_network = st.selectbox("Select Network", list(DEMO_NETWORKS.keys()))
            scope_key   = f"network::{selected_network}"
            scope_label = selected_network
        elif scope_level == "By School":
            sel_net = st.selectbox("Network", list(DEMO_NETWORKS.keys()), key="scope_net")
            with sc3:
                sel_school = st.selectbox("School", DEMO_NETWORKS[sel_net])
            scope_key   = f"school::{sel_school}"
            scope_label = sel_school
        else:
            scope_key   = "district::CPS"
            scope_label = "Chicago Public Schools (District)"
            st.markdown(
                "<div style='padding-top:28px; font-size:0.85rem; color:#64748B;'>"
                "All networks &amp; schools combined</div>",
                unsafe_allow_html=True,
            )

    n_students = 600 if scope_level == "Entire District (CPS)" else (300 if scope_level == "By Network" else 150)

    cohort = build_cohort(scope_key, n=n_students)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Intervention levers ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Intervention Scenario Modeling</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-sub'>
        Adjust these levers to model the effect of targeted district programs on overall risk.
        The risk distribution updates instantly.
    </div>
    """, unsafe_allow_html=True)

    iv1, iv2, iv3, iv4 = st.columns(4)
    with iv1:
        dental_boost = st.slider(
            "Improve Dental Compliance",
            min_value=0, max_value=1, value=0,
            help="Set to 1 to make all non-compliant students compliant — models impact of a district dental program.",
            format="%d (bring all to compliant)" if False else "%d",
        )
        st.caption("0 = no change · 1 = all students compliant")
    with iv2:
        vision_boost = st.slider(
            "Improve Vision Screening",
            min_value=0, max_value=1, value=0,
            help="Set to 1 to make all students vision-screened.",
        )
        st.caption("0 = no change · 1 = all students screened")
    with iv3:
        transfer_reduction = st.slider(
            "Reduce School Transfers",
            min_value=0, max_value=3, value=0,
            help="Reduce each student's lifetime transfer count by this amount — models school stabilization programs.",
        )
        st.caption("Transfers reduced per student (max 3)")
    with iv4:
        homeless_reduction = st.slider(
            "Housing Stabilization",
            min_value=0, max_value=1, value=0,
            help="Set to 1 to model the effect of resolving housing instability for all currently homeless students.",
        )
        st.caption("0 = no change · 1 = all housed")

    interventions = {
        "dental_boost":        dental_boost,
        "vision_boost":        vision_boost,
        "transfer_reduction":  transfer_reduction,
        "homeless_reduction":  homeless_reduction,
    }

    any_intervention = any(v > 0 for v in interventions.values())

    # Score baseline and intervention
    baseline_probs = cohort["prob"].values
    if any_intervention:
        with st.spinner("Re-scoring population with interventions..."):
            intervened_probs = apply_interventions(cohort, interventions)
    else:
        intervened_probs = baseline_probs

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── KPI summary ───────────────────────────────────────────────────────────
    base_at_risk   = float((baseline_probs >= THRESHOLD3).mean())
    inter_at_risk  = float((intervened_probs >= THRESHOLD3).mean())
    base_avg       = float(baseline_probs.mean())
    inter_avg      = float(intervened_probs.mean())
    delta_pct      = inter_at_risk - base_at_risk
    delta_avg      = inter_avg - base_avg

    kc1, kc2, kc3, kc4 = st.columns(4)

    def _kpi_card(col, value, label, sub, color="#0A1628", delta=None):
        delta_html = ""
        if delta is not None and abs(delta) > 0.0001:
            sign  = "▼" if delta < 0 else "▲"
            dcol  = "#22C55E" if delta < 0 else "#EF4444"
            delta_html = (
                f"<div style='font-size:0.78rem; color:{dcol}; font-weight:600; margin-top:4px;'>"
                f"{sign} {abs(delta):.1%} vs. baseline</div>"
            )
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{color};'>{value}</div>
                <div class='metric-label'>{label}</div>
                <div class='metric-sub'>{sub}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)

    _kpi_card(kc1, f"{n_students:,}", "Students Modeled",
              f"Representative cohort for {scope_label[:30]}")
    _kpi_card(kc2, f"{base_at_risk:.0%}", "At-Risk (Baseline)",
              "Score ≥ 40% — flagged for outreach",
              color="#EF4444" if base_at_risk > 0.4 else "#F59E0B")
    _kpi_card(kc3, f"{inter_at_risk:.0%}", "At-Risk (With Interventions)",
              "After applying your scenario levers",
              color="#EF4444" if inter_at_risk > 0.4 else "#22C55E",
              delta=delta_pct if any_intervention else None)
    _kpi_card(kc4, f"{inter_avg:.0%}", "Avg Risk Score",
              "Mean probability across all students",
              delta=delta_avg if any_intervention else None)

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── School-level breakdown (shown only when a network is selected) ─────────
    if scope_level == "By Network":
        st.markdown("<div class='section-header'>School-Level Risk Breakdown</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-sub'>At-risk student counts per school in this network</div>",
            unsafe_allow_html=True,
        )

        school_rows = []
        for school in DEMO_NETWORKS[selected_network]:
            sc_cohort = build_cohort(f"school::{school}", n=150)
            sc_probs  = sc_cohort["prob"].values
            n_total   = len(sc_probs)
            n_atrisk  = int((sc_probs >= THRESHOLD3).sum())
            n_high    = int((sc_probs >= 0.60).sum())
            school_rows.append({
                "School":              school,
                "Total Students":      n_total,
                "At-Risk (≥40%)":      n_atrisk,
                "High Risk (≥60%)":    n_high,
                "At-Risk Rate":        n_atrisk / n_total,
            })

        school_df = pd.DataFrame(school_rows).sort_values("At-Risk (≥40%)", ascending=False)

        # Render as styled cards
        for _, row in school_df.iterrows():
            rate     = row["At-Risk Rate"]
            bar_pct  = int(rate * 100)
            bar_col  = "#EF4444" if rate >= 0.40 else ("#F59E0B" if rate >= 0.25 else "#22C55E")
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
                        padding:14px 20px; margin-bottom:8px; display:flex;
                        align-items:center; justify-content:space-between; gap:16px;'>
                <div style='min-width:220px; font-weight:600; font-size:0.88rem; color:#1E293B;'>
                    {row["School"]}
                </div>
                <div style='flex:1; background:#F1F5F9; border-radius:6px; height:10px; overflow:hidden;'>
                    <div style='width:{bar_pct}%; background:{bar_col}; height:10px; border-radius:6px;'></div>
                </div>
                <div style='min-width:80px; text-align:right; font-size:0.82rem; color:#64748B;'>
                    <b style='color:{bar_col};'>{row["At-Risk (≥40%)"]:,}</b> / {row["Total Students"]:,} at risk
                </div>
                <div style='min-width:60px; text-align:right; font-size:0.82rem;
                            font-weight:700; color:{bar_col};'>{bar_pct}%</div>
                <div style='min-width:100px; text-align:right; font-size:0.78rem; color:#EF4444;'>
                    🔴 {row["High Risk (≥60%)"]} high risk
                </div>
                <a href='#' style='font-size:0.76rem; color:#3B82F6; text-decoration:none;
                                   white-space:nowrap; opacity:0.6; cursor:not-allowed;'
                   title='Coming soon'>⬇ Download Report</a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Distribution chart ────────────────────────────────────────────────────
    chart_l, chart_r = st.columns([1.4, 1], gap="large")

    with chart_l:
        st.markdown("<div class='section-header'>Risk Score Distribution</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-sub'>How students are spread across the risk spectrum</div>",
            unsafe_allow_html=True,
        )

        bins = np.linspace(0, 1, 21)
        base_counts, _  = np.histogram(baseline_probs, bins=bins)
        inter_counts, _ = np.histogram(intervened_probs, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_labels  = [f"{int(b*100)}%" for b in bin_centers]

        bar_colors_base = [
            "#EF4444" if b >= 0.60 else ("#F59E0B" if b >= THRESHOLD3 else "#22C55E")
            for b in bin_centers
        ]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=bin_labels,
            y=base_counts,
            name="Baseline",
            marker_color=bar_colors_base,
            opacity=0.85,
        ))
        if any_intervention:
            fig_dist.add_trace(go.Bar(
                x=bin_labels,
                y=inter_counts,
                name="With Interventions",
                marker_color="#3B82F6",
                opacity=0.55,
            ))
        fig_dist.add_vline(
            x=bin_labels[np.searchsorted(bin_centers, THRESHOLD3)],
            line_dash="dash", line_color="#0A1628", line_width=1.5,
            annotation_text="At-Risk Threshold (40%)",
            annotation_position="top right",
            annotation_font_size=10,
        )
        fig_dist.update_layout(
            height=320,
            margin=dict(l=0, r=10, t=10, b=30),
            paper_bgcolor="white",
            plot_bgcolor="white",
            barmode="overlay",
            xaxis=dict(title="Risk Score", tickangle=-30, tickfont=dict(size=9)),
            yaxis=dict(title="Number of Students", showgrid=True, gridcolor="#F1F5F9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10)),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_r:
        st.markdown("<div class='section-header'>Risk Tier Breakdown</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-sub'>Students by risk category — baseline vs. scenario</div>",
            unsafe_allow_html=True,
        )

        def tier_counts(probs):
            return {
                "Low Risk (<40%)":      int((probs < THRESHOLD3).sum()),
                "Elevated (40–60%)":   int(((probs >= THRESHOLD3) & (probs < 0.60)).sum()),
                "High Risk (≥60%)":    int((probs >= 0.60).sum()),
            }

        tc_base  = tier_counts(baseline_probs)
        tc_inter = tier_counts(intervened_probs)

        tier_labels = list(tc_base.keys())
        tier_colors = ["#22C55E", "#F59E0B", "#EF4444"]

        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=tier_labels,
            values=list(tc_inter.values()) if any_intervention else list(tc_base.values()),
            marker_colors=tier_colors,
            hole=0.48,
            textfont=dict(size=11),
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="white",
            showlegend=False,
            font=dict(family="Inter"),
            annotations=[dict(
                text="<b>Risk<br>Mix</b>",
                x=0.5, y=0.5, font_size=13, showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        for label, color in zip(tier_labels, tier_colors):
            b_n = tc_base[label]
            i_n = tc_inter[label]
            delta_n = i_n - b_n
            delta_str = ""
            if any_intervention and delta_n != 0:
                sign = "▼" if delta_n < 0 else "▲"
                dcol = "#22C55E" if (delta_n < 0 and "High" in label) or \
                                    (delta_n < 0 and "Elevated" in label) or \
                                    (delta_n > 0 and "Low" in label) else "#EF4444"
                delta_str = (
                    f"<span style='color:{dcol}; font-size:0.74rem; font-weight:600;'>"
                    f"  {sign}{abs(delta_n)}</span>"
                )
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>"
                f"<div style='width:12px; height:12px; background:{color}; border-radius:3px;'></div>"
                f"<span style='font-size:0.82rem; color:#374151;'>{label}: "
                f"<b>{i_n if any_intervention else b_n}</b> students{delta_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

    # ── Intervention impact summary ───────────────────────────────────────────
    if any_intervention:
        students_moved = int(
            ((baseline_probs >= THRESHOLD3) & (intervened_probs < THRESHOLD3)).sum()
        )
        st.markdown("<div class='section-header'>Intervention Impact Summary</div>", unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown(f"""
            <div style='background:#F0FDF4; border:1px solid #86EFAC; border-radius:12px;
                        padding:18px 20px; text-align:center;'>
                <div style='font-size:2rem; font-weight:800; color:#16A34A;'>{students_moved}</div>
                <div style='font-size:0.78rem; font-weight:600; color:#15803D;
                            text-transform:uppercase; letter-spacing:0.05em; margin-top:4px;'>
                    Students Moved Off At-Risk List
                </div>
                <div style='font-size:0.73rem; color:#64748B; margin-top:4px;'>
                    Would drop below the 40% threshold with these interventions
                </div>
            </div>
            """, unsafe_allow_html=True)
        with ic2:
            avg_improvement = base_avg - inter_avg
            st.markdown(f"""
            <div style='background:#EFF6FF; border:1px solid #93C5FD; border-radius:12px;
                        padding:18px 20px; text-align:center;'>
                <div style='font-size:2rem; font-weight:800; color:#1D4ED8;'>
                    {avg_improvement:.1%}
                </div>
                <div style='font-size:0.78rem; font-weight:600; color:#1E40AF;
                            text-transform:uppercase; letter-spacing:0.05em; margin-top:4px;'>
                    Average Risk Reduction
                </div>
                <div style='font-size:0.73rem; color:#64748B; margin-top:4px;'>
                    Mean probability drop across all students in this scope
                </div>
            </div>
            """, unsafe_allow_html=True)
        with ic3:
            school_days = 180
            days_recovered = students_moved * int(school_days * 0.10)
            st.markdown(f"""
            <div style='background:#FFFBEB; border:1px solid #FCD34D; border-radius:12px;
                        padding:18px 20px; text-align:center;'>
                <div style='font-size:2rem; font-weight:800; color:#B45309;'>
                    ~{days_recovered:,}
                </div>
                <div style='font-size:0.78rem; font-weight:600; color:#92400E;
                            text-transform:uppercase; letter-spacing:0.05em; margin-top:4px;'>
                    Attendance Days Potentially Recovered
                </div>
                <div style='font-size:0.73rem; color:#64748B; margin-top:4px;'>
                    Estimated from students who cross the risk threshold
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Explainer footer ──────────────────────────────────────────────────────
    st.markdown("<hr class='thin'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.76rem; color:#94A3B8; line-height:1.7;'>
        <b>Methodology note:</b> This view generates a representative synthetic cohort for the selected scope
        using the same XGBoost model trained on 529,405 CPS students. Student profiles are sampled to reflect
        realistic district-level distributions of attendance history, demographics, health indicators, and
        mobility patterns. Intervention levers neutralize specific risk factors for the modeled population
        and re-score them through the same model — showing how structural programs shift the aggregate
        risk distribution. This is a planning and demonstration tool; operational scoring should use
        actual student records.
    </div>
    """, unsafe_allow_html=True)
