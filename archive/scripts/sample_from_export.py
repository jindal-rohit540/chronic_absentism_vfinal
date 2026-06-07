import os
import pandas as pd

# ── 1. Read the already-written full CSV from Lakehouse ───────────────────────
local_export = "/lakehouse/default/Files/exports/chronic_absenteeism_model_input"
part_file = next(f for f in os.listdir(local_export) if f.startswith("part-") and f.endswith(".csv"))

df = pd.read_csv(os.path.join(local_export, part_file))
print(f"Full dataset: {len(df):,} rows, columns: {list(df.columns)}")

# ── 2. Pull school assignment for each student (most recent year only) ─────────
most_recent_year = int(df["school_year"].max())
print(f"Most recent year: {most_recent_year}")

enroll_pdf = (
    spark.read.table("MDMF_COPY_RN.dbo.fact_enrollment_annualized")
    .filter(f"SCHOOL_YEAR = {most_recent_year}")
    .select("STUDENT_KEY", "SCHOOL_KEY")
    .dropDuplicates(["STUDENT_KEY"])
    .toPandas()
)
print(f"Enrollment rows for {most_recent_year}: {len(enroll_pdf):,}")

# dim_school Spark table — maps SCHOOL_KEY → SCHOOL_CODE
dim_school_pdf = (
    spark.read.table("MDMF_COPY_RN.dbo.dim_school")
    .select("SCHOOL_KEY", "SCHOOL_CODE")
    .dropDuplicates(["SCHOOL_KEY"])
    .toPandas()
)

# Merge to get SCHOOL_CODE per student
student_school = enroll_pdf.merge(dim_school_pdf, on="SCHOOL_KEY", how="left").drop(columns=["SCHOOL_KEY"])

# ── 3. Load dim_schools.csv → SCHOOL_CODE to SCHOOL_NAME + NETWORK ────────────
dim_schools = pd.read_csv("/lakehouse/default/Files/Built-in/dim_schools.csv")
dim_schools["SCHOOL_CODE"] = dim_schools["SCHOOL_CODE"].astype(str)
student_school["SCHOOL_CODE"] = student_school["SCHOOL_CODE"].astype(str)

student_school = student_school.merge(dim_schools[["SCHOOL_CODE","SCHOOL_NAME","NETWORK"]], on="SCHOOL_CODE", how="left")

# ── 4. Join school/network onto the feature df ─────────────────────────────────
df["STUDENT_KEY"] = df["STUDENT_KEY"].astype(str)
student_school["STUDENT_KEY"] = student_school["STUDENT_KEY"].astype(str)

df = df[df["school_year"] == most_recent_year].merge(student_school, on="STUDENT_KEY", how="left")
print(f"After join — rows: {len(df):,}, networks: {df['NETWORK'].nunique()}")
print(df["NETWORK"].value_counts().head(10))

# ── 5. Sample 100 students from each of the 2 biggest networks ────────────────
top2 = df["NETWORK"].value_counts().head(2).index.tolist()
print(f"\nSampling from networks: {top2}")

sample = (
    df[df["NETWORK"].isin(top2)]
    .groupby("NETWORK", group_keys=False)
    .apply(lambda g: g.sample(min(100, len(g)), random_state=42))
    .reset_index(drop=True)
)
print(f"Sample rows: {len(sample)}")
print(sample.groupby("NETWORK")["SCHOOL_NAME"].nunique())

# ── 6. Download ────────────────────────────────────────────────────────────────
builtin_dir = os.path.join(notebookutils.nbResPath, "builtin")
os.makedirs(builtin_dir, exist_ok=True)

dest = os.path.join(builtin_dir, "chronic_absenteeism_sample.csv")
sample.to_csv(dest, index=False)
print(f"\nDownload: Notebook Resources → builtin → chronic_absenteeism_sample.csv")
