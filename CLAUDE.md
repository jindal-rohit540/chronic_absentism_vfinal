# CPS Chronic Absenteeism — Project Reference

## What This Is
A Streamlit demo app (`chronic_absenteeism_app.py`) for Chicago Public Schools leadership showing an XGBoost model that predicts which students are at risk of chronic absenteeism (missing ≥10% of school days).

**Primary audience:** Non-technical CPS VPs and "key doers" from Nutrition, Attendance, Transportation, Facilities, etc.
**Deployed at:** Streamlit Cloud — repo `jindal-rohit540/chronic_absentism_vfinal`

---

## App Pages

| Page | File section | What it does |
|------|-------------|--------------|
| Executive Summary | `chronic_absenteeism_app.py` | Static overview, feature importance chart, accuracy KPIs |
| Student Risk Predictor | `chronic_absenteeism_app.py` | Single student scoring with live gauge + contributing factors waterfall |
| Network & District View | `chronic_absenteeism_app.py` | Aggregate risk view — filter by governance type, top 10 schools, drill to school-level student list |

**Key design rule:** The risk indicator must visibly move when inputs change — this is intentional demo behavior that builds credibility with stakeholders. Never remove it.

---

## Local Files

| File | Purpose |
|------|---------|
| `chronic_absenteeism_app.py` | Main Streamlit app |
| `fabric_generate_predictions.py` | Run in Fabric to regenerate `predictions_by_network.csv` |
| `Built-in/chronic_absenteeism_xgb.pkl` | XGBoost model artifact (288KB) |
| `Built-in/predictions_by_network.csv` | 342,018 student predictions — one row per student |
| `Built-in/dim_schools.csv` | 2,236 schools with SCHOOL_CODE, SCHOOL_NAME, NETWORK — used only in Page 2 |
| `fabric_inspect_schemas.py` | Helper to inspect Fabric table schemas |

---

## Model

- **Type:** XGBoost binary classifier
- **Target:** Chronic absenteeism (attendance rate < 90%)
- **Threshold:** 0.40 (best F1) — stored in artifact as `best_threshold`
- **Trained on:** 529,405 students, AY 2018–2026
- **Artifact keys:** `model`, `label_encoders`, `label_mappings`, `numeric_cols`, `categorical_cols`, `all_feature_cols`, `best_threshold`
- **30 features:** 24 numeric + 6 categorical (gender, race, ethnicity, language, grade, enrollment status)
- **Attendance rate formula:** `1 - (absent_value_sum / (absent_value_sum + attendance_value_sum))`

---

## Microsoft Fabric Environment

All data lives in `MDMF_COPY_RN.dbo` tables. Code runs in Fabric notebooks only — cannot run locally.

### Key Tables

| Table | Key columns | Notes |
|-------|------------|-------|
| `fact_attendance` | `STUDENT_KEY`, `ATTENDANCE_DATE`, `ABSENT_VALUE`, `ATTENDANCE_VALUE`, `EXCUSED_INDICATOR`, `TARDY_INDICATOR` | One row per student per day |
| `dim_student` | `STUDENT_KEY`, `STUDENT_GENDER`, `STUDENT_RACE`, `STUDENT_ETHNICITY`, `STUDENT_LANGUAGE`, `STUDENT_CURRENT_GRADE_CODE`, `STUDENT_SPECIAL_ED_INDICATOR`, `STUDENT_HOMELESS_INDICATOR`, `STUDENT_FOODSERVICE_INDICATOR`, `STUDENT_504_INDICATOR`, `ENROLLMENT_HISTORY_STATUS` | Demographics |
| `fact_enrollment_annualized` | `STUDENT_KEY`, `SCHOOL_KEY`, `SCHOOL_YEAR`, `ENROLLMENT_TYPE` | Use `ENROLLMENT_TYPE = 'PRIMARY'` to get main school |
| `dim_school` | `SCHOOL_KEY`, `SCHOOL_NAME`, `SCHOOL_CODE`, `NETWORK`, `GOVERNANCE` | Use `GOVERNANCE` (not `NETWORK`) for grouping |
| `dim_cohort_school` | `COHORT_SCHOOL_KEY`, `COHORT_NAME` | Filter `COHORT_NAME = 'All Schools'` for official CPS schools only |
| `dim_cohort_school_membership` | `SCHOOL_KEY`, `COHORT_SCHOOL_KEY` | Join table between dim_school and dim_cohort_school |
| `fact_student_medical_condition` | `STUDENT_KEY`, `MEDICAL_CONDITION_KEY` | |
| `dim_medical_condition` | `MEDICAL_CONDITION_KEY`, `MEDICAL_CONDITION_NAME` | |
| `dim_student_health_indicator` | `STUDENT_KEY`, `IMMUNIZATIONS_COMPLIANT_INDICATOR`, `HEALTH_EXAMS_COMPLIANT_INDICATOR`, `DENTAL_COMPLIANT_INDICATOR`, `VISION_SCREENING_COMPLIANT_INDICATOR` | |
| `dim_medicaid` | `STUDENT_KEY`, `CURRENT_IND` | Filter `CURRENT_IND == True` for active Medicaid |

### Official CPS School Filter (Conor Moloney's method)
Always use this to exclude non-CPS schools. 646 official schools confirmed.
```sql
SELECT GOVERNANCE, count(*)
FROM dim_school sch
INNER JOIN dim_cohort_school_membership csm ON sch.SCHOOL_KEY = csm.SCHOOL_KEY
INNER JOIN dim_cohort_school c ON c.COHORT_SCHOOL_KEY = csm.COHORT_SCHOOL_KEY
WHERE c.COHORT_NAME = 'All Schools'
GROUP BY sch.GOVERNANCE
ORDER BY 2
```
**Result (confirmed 2026-06-06):**
- District: 520
- Charter: 109
- ALOP: 8
- Contract: 6
- SAFE: 3

### Saving Files in Fabric
Save directly to `notebookutils.nbResPath/builtin/` — do NOT write to lakehouse path and shutil.copy (causes disk quota error).
```python
import os
builtin_dir = os.path.join(notebookutils.nbResPath, "builtin")
os.makedirs(builtin_dir, exist_ok=True)
save_path = os.path.join(builtin_dir, "filename.csv")
output.to_csv(save_path, index=False)
# Download from: Notebook Resources → builtin → filename.csv
```

### Loading the Model Artifact in Fabric
```python
import joblib, os
ARTIFACT_PATH = os.path.join(notebookutils.nbResPath, "builtin", "chronic_absenteeism_xgb.pkl")
artifact = joblib.load(ARTIFACT_PATH)
```

### Scoring Without Full toPandas (use mapInPandas)
Never do `full_df.toPandas()` on the feature DataFrame — too slow for 342K rows. Use `mapInPandas` to score on workers:
```python
from pyspark.sql.types import StructType, StructField, DoubleType

output_schema = StructType(final_df.schema.fields + [StructField("risk_score", DoubleType(), True)])

def score_partition(iterator):
    import joblib, pandas as pd, os
    art = joblib.load(ARTIFACT_PATH)
    for batch in iterator:
        X = batch[art["all_feature_cols"]].copy()
        # label-encode categoricals, then:
        batch["risk_score"] = art["model"].predict_proba(X)[:, 1].round(4)
        yield batch

scored_df = final_df.mapInPandas(score_partition, schema=output_schema)
```
Then do a slim `.toPandas()` on only the 12 output columns needed for the dashboard.

---

## predictions_by_network.csv Schema

Generated by `fabric_generate_predictions.py`. Lives in `Built-in/`.

| Column | Type | Description |
|--------|------|-------------|
| `GOVERNANCE` | string | District / Charter / ALOP / Contract / SAFE |
| `SCHOOL_NAME` | string | Full school name |
| `grade` | string | e.g. `01`, `02`, `PK`, `K` |
| `gender` | string | `M`, `F`, `N`, `U` |
| `race` | string | Full label e.g. "Hispanic", "Black or African American" |
| `risk_score` | float | 0.0–1.0 probability from model |
| `risk_tier` | string | Low Risk / Elevated Risk / High Risk |
| `prior_yr_attendance` | float | 0.0–1.0 attendance rate last year |
| `homeless` | int | 0 or 1 |
| `medicaid` | int | 0 or 1 |
| `sped` | int | 0 or 1 (special ed) |
| `chronic_condition_count` | int | Number of medical conditions on file |
| `student_id` | string | Anonymized ID e.g. `STU-00001` — real STUDENT_KEY is dropped |

---

## Regenerating Predictions

1. Open `fabric_generate_predictions.py` in a Fabric notebook
2. Make sure `chronic_absenteeism_xgb.pkl` is uploaded to Notebook Resources → builtin
3. Run the notebook — it takes several minutes (342K students × 30 features)
4. Download `predictions_by_network.csv` from Notebook Resources → builtin
5. Replace `Built-in/predictions_by_network.csv` locally
6. `git add Built-in/predictions_by_network.csv && git commit && git push`
7. Streamlit Cloud redeploys automatically

---

## Stakeholder Context

- **Sree Sundaram** — Director, requested district-wide view and confirmed "indicator moves" is a good demo gimmick
- **Conor Moloney** — Data lead, provided GOVERNANCE column guidance and official CPS school filter
- **Next step** — Roadshow with key doers (Nutrition, Attendance, Transportation, Facilities) to get feedback on what's missing
