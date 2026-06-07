"""
Fabric notebook cell — generates chronic_absenteeism_model_input.csv
Run this in Microsoft Fabric (PySpark kernel) to export a CSV with all
features the XGBoost model needs for scoring or retraining.

Paste each section as a separate notebook cell, or run as one block.
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os, shutil

# ── 1. LOAD TABLES ─────────────────────────────────────────────────────────────
fact_attendance = spark.read.table("MDMF_COPY_RN.dbo.fact_attendance")
dim_student     = spark.read.table("MDMF_COPY_RN.dbo.dim_student")
fact_med        = spark.read.table("MDMF_COPY_RN.dbo.fact_student_medical_condition")
dim_med_cond    = spark.read.table("MDMF_COPY_RN.dbo.dim_medical_condition")
health_ind      = spark.read.table("MDMF_COPY_RN.dbo.dim_student_health_indicator")
dim_medicaid    = spark.read.table("MDMF_COPY_RN.dbo.dim_medicaid")
fact_enroll     = spark.read.table("MDMF_COPY_RN.dbo.fact_enrollment_annualized")

print("All tables loaded.")

# ── 2. ATTENDANCE SPINE + LAG FEATURES ─────────────────────────────────────────
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
    .withColumn("is_chronic_current_year", F.when(F.col("attendance_rate") < 0.90, 1).otherwise(0))
)

# Lag features (t-1 and t-2)
w_hist = Window.partitionBy("STUDENT_KEY").orderBy("school_year")

student_with_lags = (
    student_yearly
    .withColumn("attendance_rate_t_minus_1", F.lag("attendance_rate", 1).over(w_hist))
    .withColumn("attendance_rate_t_minus_2", F.lag("attendance_rate", 2).over(w_hist))
    .withColumn("tardy_total_t_minus_1",     F.lag("tardy_total",     1).over(w_hist))
    .withColumn("excused_absences_t_minus_1",F.lag("excused_absences",1).over(w_hist))
)

# Spine = one row per (student, year) where we can form a prediction
spine = (
    student_with_lags
    .filter(F.col("attendance_rate_t_minus_1").isNotNull())   # need at least 1 prior year
    .select(
        "STUDENT_KEY", "school_year", "covid_year",
        "attendance_rate_t_minus_1", "attendance_rate_t_minus_2",
        "tardy_total_t_minus_1", "excused_absences_t_minus_1",
        "is_chronic_current_year",
    )
    .withColumn("prior_school_year", F.col("school_year") - 1)
)

print(f"Spine rows: {spine.count():,}")

# ── 3. DEMOGRAPHICS ────────────────────────────────────────────────────────────
student_demo = (
    dim_student
    .select(
        "STUDENT_KEY",
        "STUDENT_GENDER", "STUDENT_RACE", "STUDENT_ETHNICITY",
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

# ── 4. HEALTH CONDITIONS ───────────────────────────────────────────────────────
med_with_names = fact_med.join(dim_med_cond, on="MEDICAL_CONDITION_KEY", how="left")

medical_features = (
    med_with_names
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

# ── 5. MEDICAID ────────────────────────────────────────────────────────────────
medicaid_features = (
    dim_medicaid
    .filter(F.col("CURRENT_IND") == True)
    .select("STUDENT_KEY")
    .distinct()
    .withColumn("medicaid_indicator", F.lit(1))
)

# ── 6. SCHOOL TRANSFERS ────────────────────────────────────────────────────────
transfer_yearly = (
    fact_enroll
    .groupBy("STUDENT_KEY", "SCHOOL_YEAR")
    .agg(
        (F.countDistinct("SCHOOL_KEY") - 1).alias("school_transfers_this_year"),
        F.countDistinct("SCHOOL_KEY").alias("schools_this_year"),
    )
    .withColumn("transferred_this_year", F.when(F.col("school_transfers_this_year") > 0, 1).otherwise(0))
)

w_cum = (
    Window.partitionBy("STUDENT_KEY")
          .orderBy("SCHOOL_YEAR")
          .rowsBetween(Window.unboundedPreceding, -1)
)

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

# ── 7. JOIN EVERYTHING ─────────────────────────────────────────────────────────
final_df = (
    spine
    .join(student_demo,      on="STUDENT_KEY", how="left")
    .join(medical_features,  on="STUDENT_KEY", how="left")
    .join(health_compliance, on="STUDENT_KEY", how="left")
    .join(medicaid_features, on="STUDENT_KEY", how="left")
    .join(
        trn_agg.alias("trn"),
        (spine["STUDENT_KEY"] == F.col("trn.STUDENT_KEY")) &
        (spine["school_year"] == F.col("trn.SCHOOL_YEAR")),
        how="left"
    )
    .join(
        lifetime_transfers.alias("ltrn"),
        (spine["STUDENT_KEY"]       == F.col("ltrn.STUDENT_KEY")) &
        (spine["prior_school_year"] == F.col("ltrn.SCHOOL_YEAR")),
        how="left"
    )
    .select(
        spine["STUDENT_KEY"],
        spine["school_year"],
        "covid_year",
        "attendance_rate_t_minus_1",
        "attendance_rate_t_minus_2",
        "tardy_total_t_minus_1",
        "excused_absences_t_minus_1",
        "is_chronic_current_year",          # label — keep for evaluation / retraining
        "STUDENT_GENDER", "STUDENT_RACE", "STUDENT_ETHNICITY",
        "STUDENT_LANGUAGE", "STUDENT_CURRENT_GRADE_CODE", "ENROLLMENT_HISTORY_STATUS",
        "STUDENT_SPECIAL_ED_INDICATOR", "STUDENT_HOMELESS_INDICATOR",
        "STUDENT_FOODSERVICE_INDICATOR", "STUDENT_504_INDICATOR",
        "chronic_condition_count", "has_asthma", "has_diabetes",
        "has_allergy", "has_seizure", "has_chronic_condition",
        "immunizations_compliant", "health_exams_compliant",
        "dental_compliant", "vision_compliant",
        "medicaid_indicator",
        "school_transfers_this_year", "transferred_this_year",
        "lifetime_school_transfers",
    )
)

# ── 8. FILL NULLS + DERIVED FIELDS ────────────────────────────────────────────
final_df = (
    final_df
    .fillna({"attendance_rate_t_minus_1": 0.95, "attendance_rate_t_minus_2": 0.95})
    .fillna(0)
    .fillna("Unknown")
    .withColumn(
        "was_chronic_last_year",
        F.when(F.col("attendance_rate_t_minus_1") < 0.90, 1).otherwise(0)
    )
)

print(f"Final DF rows: {final_df.count():,}")

# ── 9. ATTACH SCHOOL / NETWORK ─────────────────────────────────────────────────
# fact_enrollment_annualized has SCHOOL_KEY; join to get SCHOOL_CODE + NETWORK.
# We only need the most recent school year per student for the demo CSV.

# Get SCHOOL_CODE for each student in the most recent year they appear
dim_schools_spark = spark.read.table("MDMF_COPY_RN.dbo.dim_school")   # adjust table name if different
# Alternatively if dim_school is not available, derive from fact_enrollment_annualized:
# pick the school with the latest entry per student in the most recent year

most_recent_year = final_df.agg(F.max("school_year")).collect()[0][0]
print(f"Most recent school year in spine: {most_recent_year}")

# Get one SCHOOL_KEY per student for the most recent year
school_per_student = (
    fact_enroll
    .filter(F.col("SCHOOL_YEAR") == most_recent_year)
    .groupBy("STUDENT_KEY")
    .agg(F.first("SCHOOL_KEY", ignorenulls=True).alias("SCHOOL_KEY"))
)

# Join dim_school to get SCHOOL_CODE
school_lookup = (
    dim_schools_spark
    .select("SCHOOL_KEY", "SCHOOL_CODE")
    .dropDuplicates(["SCHOOL_KEY"])
)

student_school = school_per_student.join(school_lookup, on="SCHOOL_KEY", how="left").drop("SCHOOL_KEY")

# Attach SCHOOL_CODE to final_df (most recent year only for the sample)
final_with_school = (
    final_df
    .filter(F.col("school_year") == most_recent_year)
    .join(student_school, on="STUDENT_KEY", how="left")
)

print(f"Rows for most recent year ({most_recent_year}): {final_with_school.count():,}")

# ── 10. SAMPLE ~50 STUDENTS PER SCHOOL ────────────────────────────────────────
# This keeps the file small (~500 schools × 50 = ~25k rows max) while
# ensuring every school and network is represented for the demo.

w_sample = Window.partitionBy("SCHOOL_CODE").orderBy(F.rand(seed=42))

sampled_df = (
    final_with_school
    .withColumn("_rn", F.row_number().over(w_sample))
    .filter(F.col("_rn") <= 50)
    .drop("_rn")
)

print(f"Sampled rows: {sampled_df.count():,}")

# ── 11. WRITE SMALL CSV + DOWNLOAD ────────────────────────────────────────────
csv_lakehouse_path = "Files/exports/chronic_absenteeism_sample"

(
    sampled_df
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(csv_lakehouse_path)
)

print(f"CSV written to: {csv_lakehouse_path}")

local_export = f"/lakehouse/default/{csv_lakehouse_path}"
part_file = next(f for f in os.listdir(local_export) if f.startswith("part-") and f.endswith(".csv"))

builtin_dir = os.path.join(notebookutils.nbResPath, "builtin")
os.makedirs(builtin_dir, exist_ok=True)

dest = os.path.join(builtin_dir, "chronic_absenteeism_sample.csv")
shutil.copy(os.path.join(local_export, part_file), dest)

print(f"\nDownload from: Notebook Resources → builtin → chronic_absenteeism_sample.csv")
print(f"Schools covered: {sampled_df.select('SCHOOL_CODE').distinct().count():,}")
print(f"Total rows: {sampled_df.count():,}")
