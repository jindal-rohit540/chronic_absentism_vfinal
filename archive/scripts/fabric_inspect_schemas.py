# ── CELL 1: Inspect dim_school ────────────────────────────────────────────────
# Run this first and paste the output back so we know the column names

dim_school = spark.read.table("MDMF_COPY_RN.dbo.dim_school")
print("=== dim_school columns ===")
print(dim_school.columns)
print(f"Row count: {dim_school.count():,}")
dim_school.show(3, truncate=False)

# ── CELL 2: Inspect fact_enrollment_annualized ─────────────────────────────────
fact_enroll = spark.read.table("MDMF_COPY_RN.dbo.fact_enrollment_annualized")
print("=== fact_enrollment_annualized columns ===")
print(fact_enroll.columns)
fact_enroll.show(3, truncate=False)

# ── CELL 3: Check if SCHOOL_CODE exists directly in fact_enrollment_annualized ──
print("Does SCHOOL_CODE exist in fact_enrollment_annualized?", "SCHOOL_CODE" in fact_enroll.columns)
print("Does SCHOOL_KEY exist?",  "SCHOOL_KEY"  in fact_enroll.columns)
print("Does NETWORK exist?",     "NETWORK"     in fact_enroll.columns)

# ── CELL 4: Check dim_school for NETWORK / SCHOOL_NAME / SCHOOL_CODE ──────────
for col in ["SCHOOL_CODE", "SCHOOL_NAME", "NETWORK", "SCHOOL_SHORT_NAME"]:
    print(f"  {col} in dim_school: {col in dim_school.columns}")

# ── CELL 5: Sample distinct networks from dim_school ──────────────────────────
for col in dim_school.columns:
    if any(kw in col.upper() for kw in ["NETWORK", "SCHOOL_NAME", "SCHOOL_CODE", "SCHOOL_KEY"]):
        print(f"\nDistinct values of {col} (top 10):")
        dim_school.select(col).distinct().show(10, truncate=False)
