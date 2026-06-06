"""
Fabric Notebook — Generate network/school level predictions
Run in Microsoft Fabric (PySpark kernel).
Output: predictions_by_network.csv — one row per student with GOVERNANCE, SCHOOL_NAME, risk_score, risk_tier

Per Conor Moloney:
  - Use GOVERNANCE (not NETWORK) as the grouping column
  - Only include official CPS schools by joining dim_cohort_school_membership
    where COHORT_NAME = 'All Schools'
  - Covers all networks + district-level rollup
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType
import numpy as np
import joblib
import os

TARGET_YEAR = 2025   # most recent completed school year

# ── 1. LOAD TABLES ────────────────────────────────────────────────────────────
print("Loading tables...")
fact_attendance  = spark.read.table("MDMF_COPY_RN.dbo.fact_attendance")
dim_student      = spark.read.table("MDMF_COPY_RN.dbo.dim_student")
fact_med         = spark.read.table("MDMF_COPY_RN.dbo.fact_student_medical_condition")
dim_med_cond     = spark.read.table("MDMF_COPY_RN.dbo.dim_medical_condition")
health_ind       = spark.read.table("MDMF_COPY_RN.dbo.dim_student_health_indicator")
dim_medicaid     = spark.read.table("MDMF_COPY_RN.dbo.dim_medicaid")
fact_enroll      = spark.read.table("MDMF_COPY_RN.dbo.fact_enrollment_annualized")
dim_school       = spark.read.table("MDMF_COPY_RN.dbo.dim_school")
dim_cohort_school         = spark.read.table("MDMF_COPY_RN.dbo.dim_cohort_school")
dim_cohort_school_membership = spark.read.table("MDMF_COPY_RN.dbo.dim_cohort_school_membership")
print("All tables loaded.")

# ── 2. OFFICIAL CPS SCHOOLS ONLY (Conor's filter) ─────────────────────────────
# Mirrors Conor's exact SQL:
#   SELECT GOVERNANCE, count(*)
#   FROM dim_school sch
#   INNER JOIN dim_cohort_school_membership csm ON sch.SCHOOL_KEY = csm.SCHOOL_KEY
#   INNER JOIN dim_cohort_school c ON c.COHORT_SCHOOL_KEY = csm.COHORT_SCHOOL_KEY
#   WHERE c.COHORT_NAME = 'All Schools'
#
# Use GOVERNANCE as the grouping column (not NETWORK).

cps_cohort = dim_cohort_school.filter(F.col("COHORT_NAME") == "All Schools")

official_cps_schools = (
    dim_school.alias("sch")
    .join(
        dim_cohort_school_membership.alias("csm"),
        F.col("sch.SCHOOL_KEY") == F.col("csm.SCHOOL_KEY"),
        how="inner"
    )
    .join(
        cps_cohort.alias("c"),
        F.col("c.COHORT_SCHOOL_KEY") == F.col("csm.COHORT_SCHOOL_KEY"),
        how="inner"
    )
    .select(
        F.col("sch.SCHOOL_KEY"),
        F.col("sch.SCHOOL_NAME"),
        F.col("sch.GOVERNANCE"),
    )
    .dropDuplicates(["SCHOOL_KEY"])
)

print(f"Official CPS schools: {official_cps_schools.count():,}")
print("Governance types (should match Conor's query output):")
official_cps_schools.groupBy("GOVERNANCE").count().orderBy("count", ascending=False).show(20, truncate=False)

# ── 3. GET STUDENTS ENROLLED IN OFFICIAL CPS SCHOOLS FOR TARGET YEAR ──────────
students_in_scope = (
    fact_enroll
    .filter(F.col("SCHOOL_YEAR") == TARGET_YEAR)
    .filter(F.col("ENROLLMENT_TYPE") == "PRIMARY")
    .join(official_cps_schools, on="SCHOOL_KEY", how="inner")
    .groupBy("STUDENT_KEY")
    .agg(
        F.first("SCHOOL_NAME",  ignorenulls=True).alias("SCHOOL_NAME"),
        F.first("GOVERNANCE",   ignorenulls=True).alias("GOVERNANCE"),
    )
)
print(f"Students in scope: {students_in_scope.count():,}")

# ── 4. BUILD ATTENDANCE SPINE (same logic as training notebook) ───────────────
fact_attendance = fact_attendance.withColumn(
    "school_year",
    F.when(F.month("ATTENDANCE_DATE") >= 8, F.year("ATTENDANCE_DATE") + 1)
     .otherwise(F.year("ATTENDANCE_DATE"))
)

student_yearly = (
    fact_attendance
    .groupBy("STUDENT_KEY", "school_year")
    .agg(
        F.sum("ABSENT_VALUE").alias("absent_value_sum"),
        F.sum("ATTENDANCE_VALUE").alias("attendance_value_sum"),
        F.sum(F.col("EXCUSED_INDICATOR").cast("int")).alias("excused_absences"),
        F.sum(F.col("TARDY_INDICATOR").cast("int")).alias("tardy_total"),
    )
    .withColumn(
        "attendance_rate",
        F.when(
            (F.col("absent_value_sum") + F.col("attendance_value_sum")) > 0,
            1.0 - (F.col("absent_value_sum") / (F.col("absent_value_sum") + F.col("attendance_value_sum")))
        ).otherwise(F.lit(1.0))
    )
    .withColumn("covid_year", F.when(F.col("school_year") == 2021, 1).otherwise(0))
)

w_hist = Window.partitionBy("STUDENT_KEY").orderBy("school_year")

student_with_lags = (
    student_yearly
    .withColumn("attendance_rate_t_minus_1",  F.lag("attendance_rate", 1).over(w_hist))
    .withColumn("attendance_rate_t_minus_2",  F.lag("attendance_rate", 2).over(w_hist))
    .withColumn("tardy_total_t_minus_1",      F.lag("tardy_total",     1).over(w_hist))
    .withColumn("excused_absences_t_minus_1", F.lag("excused_absences", 1).over(w_hist))
)

# Only keep TARGET_YEAR rows for students in scope
spine = (
    student_with_lags
    .filter(F.col("school_year") == TARGET_YEAR)
    .join(students_in_scope, on="STUDENT_KEY", how="inner")
    .select(
        "STUDENT_KEY", "school_year", "covid_year", "SCHOOL_NAME", "GOVERNANCE",
        "attendance_rate_t_minus_1", "attendance_rate_t_minus_2",
        "tardy_total_t_minus_1", "excused_absences_t_minus_1",
    )
    .withColumn("prior_school_year", F.col("school_year") - 1)
)
print(f"Spine rows (students with attendance history): {spine.count():,}")

# ── 5. DEMOGRAPHICS ───────────────────────────────────────────────────────────
student_demo = (
    dim_student
    .select(
        "STUDENT_KEY", "STUDENT_GENDER", "STUDENT_RACE", "STUDENT_ETHNICITY",
        "STUDENT_LANGUAGE", "STUDENT_CURRENT_GRADE_CODE",
        "STUDENT_SPECIAL_ED_INDICATOR", "STUDENT_HOMELESS_INDICATOR",
        "STUDENT_FOODSERVICE_INDICATOR", "STUDENT_504_INDICATOR",
        "ENROLLMENT_HISTORY_STATUS",
    )
    .dropDuplicates(["STUDENT_KEY"])
)
for c in ["STUDENT_SPECIAL_ED_INDICATOR", "STUDENT_HOMELESS_INDICATOR",
          "STUDENT_504_INDICATOR", "STUDENT_FOODSERVICE_INDICATOR"]:
    student_demo = student_demo.withColumn(
        c, F.when(F.col(c).cast("string").isin("Y", "Yes", "1"), 1).otherwise(0)
    )

# ── 6. HEALTH ─────────────────────────────────────────────────────────────────
medical_features = (
    fact_med.join(dim_med_cond, on="MEDICAL_CONDITION_KEY", how="left")
    .groupBy("STUDENT_KEY")
    .agg(
        F.count("MEDICAL_CONDITION_KEY").alias("chronic_condition_count"),
        F.max(F.when(F.col("MEDICAL_CONDITION_NAME").rlike("(?i)asthma"),         1).otherwise(0)).alias("has_asthma"),
        F.max(F.when(F.col("MEDICAL_CONDITION_NAME").rlike("(?i)diabet"),         1).otherwise(0)).alias("has_diabetes"),
        F.max(F.when(F.col("MEDICAL_CONDITION_NAME").rlike("(?i)allerg"),         1).otherwise(0)).alias("has_allergy"),
        F.max(F.when(F.col("MEDICAL_CONDITION_NAME").rlike("(?i)seizure|epilep"), 1).otherwise(0)).alias("has_seizure"),
    )
    .withColumn("has_chronic_condition", F.when(F.col("chronic_condition_count") > 0, 1).otherwise(0))
)

health_compliance = (
    health_ind
    .select(
        "STUDENT_KEY",
        F.when(F.col("IMMUNIZATIONS_COMPLIANT_INDICATOR").isin("Y","Yes",1), 1).otherwise(0).alias("immunizations_compliant"),
        F.when(F.col("HEALTH_EXAMS_COMPLIANT_INDICATOR").isin("Y","Yes",1),  1).otherwise(0).alias("health_exams_compliant"),
        F.when(F.col("DENTAL_COMPLIANT_INDICATOR").isin("Y","Yes",1),        1).otherwise(0).alias("dental_compliant"),
        F.when(F.col("VISION_SCREENING_COMPLIANT_INDICATOR").isin("Y","Yes",1), 1).otherwise(0).alias("vision_compliant"),
    )
    .dropDuplicates(["STUDENT_KEY"])
)

# ── 7. MEDICAID ───────────────────────────────────────────────────────────────
medicaid_features = (
    dim_medicaid
    .filter(F.col("CURRENT_IND") == True)
    .select("STUDENT_KEY").distinct()
    .withColumn("medicaid_indicator", F.lit(1))
)

# ── 8. TRANSFERS ──────────────────────────────────────────────────────────────
transfer_yearly = (
    fact_enroll
    .groupBy("STUDENT_KEY", "SCHOOL_YEAR")
    .agg(
        (F.countDistinct("SCHOOL_KEY") - 1).alias("school_transfers_this_year"),
        F.countDistinct("SCHOOL_KEY").alias("schools_this_year"),
    )
    .withColumn("transferred_this_year", F.when(F.col("school_transfers_this_year") > 0, 1).otherwise(0))
)

w_cum = Window.partitionBy("STUDENT_KEY").orderBy("SCHOOL_YEAR").rowsBetween(Window.unboundedPreceding, -1)
lifetime_transfers = (
    transfer_yearly
    .withColumn("cumulative_schools", F.sum("schools_this_year").over(w_cum))
    .withColumn("lifetime_school_transfers", F.coalesce(F.col("cumulative_schools") - 1, F.lit(0)))
    .select("STUDENT_KEY", "SCHOOL_YEAR", "lifetime_school_transfers")
)
trn_agg = (
    transfer_yearly
    .groupBy("STUDENT_KEY", "SCHOOL_YEAR")
    .agg(
        F.max("school_transfers_this_year").alias("school_transfers_this_year"),
        F.max("transferred_this_year").alias("transferred_this_year"),
    )
)

# ── 9. JOIN ALL FEATURES ──────────────────────────────────────────────────────
final_df = (
    spine
    .join(student_demo,      on="STUDENT_KEY", how="left")
    .join(medical_features,  on="STUDENT_KEY", how="left")
    .join(health_compliance, on="STUDENT_KEY", how="left")
    .join(medicaid_features, on="STUDENT_KEY", how="left")
    .join(trn_agg.alias("trn"),
          (spine["STUDENT_KEY"] == F.col("trn.STUDENT_KEY")) & (spine["school_year"] == F.col("trn.SCHOOL_YEAR")),
          how="left")
    .join(lifetime_transfers.alias("ltrn"),
          (spine["STUDENT_KEY"] == F.col("ltrn.STUDENT_KEY")) & (spine["prior_school_year"] == F.col("ltrn.SCHOOL_YEAR")),
          how="left")
    .select(
        spine["STUDENT_KEY"], spine["school_year"], spine["SCHOOL_NAME"], spine["GOVERNANCE"],
        "covid_year",
        "attendance_rate_t_minus_1", "attendance_rate_t_minus_2",
        "tardy_total_t_minus_1", "excused_absences_t_minus_1",
        "STUDENT_GENDER", "STUDENT_RACE", "STUDENT_ETHNICITY", "STUDENT_LANGUAGE",
        "STUDENT_CURRENT_GRADE_CODE", "ENROLLMENT_HISTORY_STATUS",
        "STUDENT_SPECIAL_ED_INDICATOR", "STUDENT_HOMELESS_INDICATOR",
        "STUDENT_FOODSERVICE_INDICATOR", "STUDENT_504_INDICATOR",
        "chronic_condition_count", "has_asthma", "has_diabetes",
        "has_allergy", "has_seizure", "has_chronic_condition",
        "immunizations_compliant", "health_exams_compliant",
        "dental_compliant", "vision_compliant", "medicaid_indicator",
        "school_transfers_this_year", "transferred_this_year", "lifetime_school_transfers",
    )
    .fillna({"attendance_rate_t_minus_1": 0.95, "attendance_rate_t_minus_2": 0.95})
    .fillna(0)
    .fillna("Unknown")
    .withColumn("was_chronic_last_year",
                F.when(F.col("attendance_rate_t_minus_1") < 0.90, 1).otherwise(0))
)

print(f"Final feature df rows: {final_df.count():,}")

# ── 10. SCORE WITH XGBOOST MODEL (mapInPandas — runs on workers, no full toPandas) ──
ARTIFACT_PATH = os.path.join(notebookutils.nbResPath, "builtin", "chronic_absenteeism_xgb.pkl")
artifact         = joblib.load(ARTIFACT_PATH)
all_feature_cols = artifact["all_feature_cols"]
THRESHOLD        = artifact["best_threshold"]

# Output schema: all existing columns + risk_score
output_schema = StructType(
    final_df.schema.fields + [StructField("risk_score", DoubleType(), True)]
)

def score_partition(iterator):
    import joblib, pandas as pd
    art  = joblib.load(ARTIFACT_PATH)
    mdl  = art["model"]
    les  = art["label_encoders"]
    feat = art["all_feature_cols"]
    cats = ["STUDENT_GENDER", "STUDENT_RACE", "STUDENT_ETHNICITY",
            "STUDENT_LANGUAGE", "STUDENT_CURRENT_GRADE_CODE", "ENROLLMENT_HISTORY_STATUS"]

    for batch in iterator:
        X = batch[feat].copy()
        for c in cats:
            le = les[c]
            X[c] = X[c].apply(
                lambda v: le.transform([v])[0] if v in le.classes_
                          else le.transform(["Unknown"])[0]
            )
        batch["risk_score"] = mdl.predict_proba(X)[:, 1].round(4)
        yield batch

print("Scoring via mapInPandas (distributed across workers)...")
scored_df = final_df.mapInPandas(score_partition, schema=output_schema)

# Assign risk tier in Spark — no pandas needed
scored_df = scored_df.withColumn(
    "risk_tier",
    F.when(F.col("risk_score") <= THRESHOLD, "Low Risk")
     .when(F.col("risk_score") <= 0.60,      "Elevated Risk")
     .otherwise("High Risk")
)

print("Scoring complete. Risk tier distribution:")
scored_df.groupBy("risk_tier").count().orderBy("count", ascending=False).show()

# ── 11. SELECT ONLY DASHBOARD COLUMNS IN SPARK BEFORE PULLING TO DRIVER ──────
output_df = (
    scored_df
    .select(
        "STUDENT_KEY", "GOVERNANCE", "SCHOOL_NAME",
        F.col("STUDENT_CURRENT_GRADE_CODE").alias("grade"),
        F.col("STUDENT_GENDER").alias("gender"),
        F.col("STUDENT_RACE").alias("race"),
        "risk_score", "risk_tier",
        F.col("attendance_rate_t_minus_1").alias("prior_yr_attendance"),
        F.col("STUDENT_HOMELESS_INDICATOR").alias("homeless"),
        F.col("medicaid_indicator").alias("medicaid"),
        F.col("STUDENT_SPECIAL_ED_INDICATOR").alias("sped"),
        "chronic_condition_count",
    )
    .orderBy("GOVERNANCE", "SCHOOL_NAME", F.col("risk_score").desc())
)

print(f"Output rows: {output_df.count():,}")
print("Governance breakdown:")
output_df.groupBy("GOVERNANCE").count().orderBy("count", ascending=False).show(20, truncate=False)

# ── 12. PULL SLIM OUTPUT TO DRIVER, ANONYMIZE, SAVE ──────────────────────────
# Only 12 columns pulled vs full feature set — much faster toPandas()
output = output_df.toPandas()
output["student_id"] = ["STU-" + str(i+1).zfill(5) for i in range(len(output))]
output = output.drop(columns=["STUDENT_KEY"])

builtin_dir = os.path.join(notebookutils.nbResPath, "builtin")
os.makedirs(builtin_dir, exist_ok=True)

save_path = os.path.join(builtin_dir, "predictions_by_network.csv")
output.to_csv(save_path, index=False)

print(f"\nSaved to notebook resources: {save_path}")
print("Download from: Notebook Resources → builtin → predictions_by_network.csv")
print(f"Total rows: {len(output):,}")
