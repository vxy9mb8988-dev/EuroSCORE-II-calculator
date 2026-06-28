"""
EuroSCORE II Risk Calculator

Research-grade implementation of the EuroSCORE II logistic regression model for
predicting operative mortality risk in cardiac surgery patients.

Reference:
    Nashef SAM, et al. EuroSCORE II. Eur J Cardiothorac Surg. 2012;41(4):734-744.
    https://www.euroscore.org/

Usage:
    1. Prepare input CSV: EuroSCORE_DATA.csv
    2. Run: python EuroSCORE.py
    3. Output: EuroSCORE_DATA_with_euroscore.csv
"""

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION & FILE PATHS
# ============================================================================

INPUT_FILE = 'EuroSCORE_DATA.csv'
OUTPUT_FILE = 'EuroSCORE_DATA_with_euroscore.csv'

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

# All required columns that must be present in input data
REQUIRED_COLUMNS = {
    "record_id",           # Unique patient identifier
    "age",                 # Patient age in years
    "sex",                 # Gender (0=male, 1=female)
    "lung_disease",        # Chronic Pulmonary Disease (0=no, 1=yes)
    "eca",                 # Extra-cardiac Arteriopathy (0=no, 1=yes)
    "poor_mobil",          # Poor Mobility (0=no, 1=yes)
    "redo_proc",           # Previous Cardiac Surgery (0=no, 1=yes)
    "endocard",            # Active Endocarditis (0=no, 1=yes)
    "critical_preop",      # Critical Preoperative State (0=no, 1=yes)
    "renal_impairment",    # Renal Impairment classification (0-3)
    "iddm",                # Insulin-Dependent Diabetes (0=no, 1=yes)
    "ccs_angina",          # CCS Class 4 Angina (0=no, 1=yes)
    "lv_function",         # Left Ventricular Function (0-3)
    "recentmi",            # Recent Myocardial Infarction (0=no, 1=yes)
    "pulm_ht",             # Pulmonary Hypertension (0-2)
    "nyha",                # NYHA Functional Class (0-3)
    "surg_thoracicao",     # Thoracic Aorta Surgery (0=no, 1=yes)
    "urgency",            # Surgery Urgency (0-3)
    "weight_of_op",        # Weight/Type of Procedure (0-3)
    "euroscore_ii",        # Output: calculated risk score
}

# Input columns used for validation and missing data removal
# (excludes the output column)
input_columns = [
    "record_id",
    "age",
    "sex",
    "lung_disease",
    "eca",
    "poor_mobil",
    "redo_proc",
    "endocard",
    "critical_preop",
    "renal_impairment",
    "iddm",
    "ccs_angina",
    "lv_function",
    "recentmi",
    "pulm_ht",
    "nyha",
    "surg_thoracicao",
    "urgency",
    "weight_of_op",
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_headers(df):
    """
    Validate that all required columns exist in the input dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If any required columns are missing
    """
    actual_columns = set(df.columns)
    missing_columns = REQUIRED_COLUMNS - actual_columns

    if missing_columns:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    return True

# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

print("Validating column headers...")
validate_headers(df)

# Remove rows with missing values in required input columns
removed_count = df[input_columns].isna().any(axis=1).sum()
print(f"Removing {removed_count} rows with missing values...")
df = df.dropna(subset=input_columns)
print(f"Processing {len(df)} patient records...\n")

# ============================================================================
# EUROSCORE II COEFFICIENTS
# ============================================================================

# Validated coefficients from the EuroSCORE II model
# Reference: Nashef SAM, et al. EuroSCORE II. 2012
coefficients = {
    # NYHA Functional Class coefficients
    "NYHA_I": 0,               # NYHA Class I (asymptomatic)
    "NYHA_II": 0.1070545,      # NYHA Class II (mild symptoms)
    "NYHA_III": 0.2958358,     # NYHA Class III (marked symptoms)
    "NYHA_IV": 0.5597929,      # NYHA Class IV (severe symptoms)
    
    # Cardiac risk factors
    "CCS4": 0.2226147,         # CCS Class 4 Angina
    "IDDM": 0.3542749,         # Insulin-Dependent Diabetes Mellitus
    "age": 0.0285181,          # Age adjustment coefficient
    "female": 0.2196434,       # Female gender
    "ECA": 0.5360268,          # Extra-cardiac Arteriopathy
    "CPD": 0.1886564,          # Chronic Pulmonary Disease
    "nm_mobility": 0.2407181,  # Non-mobile (poor mobility)
    "redo": 1.118599,          # Redo/Previous cardiac surgery
    
    # Renal dysfunction coefficients
    "ccabove85": 0,            # Creatinine >85 (normal)
    "On_dialysis": 0.6421508,  # On dialysis
    "ccbelow50": 0.8592256,    # Creatinine <50 (severe impairment)
    "cc5085": 0.303553,        # Creatinine 50-85 (mild-moderate impairment)
    "endocarditis": 0.6194522, # Active endocarditis
    "critical_state": 1.086517,# Critical preoperative state
    
    # Left Ventricular Function coefficients
    "lvef_good": 0,            # LVEF >50% (good)
    "lvef_moderate": 0.3150652,# LVEF 31-50% (moderate)
    "lvef_poor": 0.8084096,    # LVEF 21-30% (poor)
    "lvef_very_poor": 0.9346919, # LVEF ≤20% (very poor)
    
    # Recent MI coefficient
    "recent_mi": 0.1528943,    # Recent MI (<90 days)
    
    # Pulmonary Artery Systolic Pressure coefficients
    "normal": 0,               # PA systolic pressure normal
    "moderate": 0.1788899,     # PA systolic pressure moderate
    "severe": 0.3491475,       # PA systolic pressure severe
    
    # Surgical urgency coefficients
    "elective": 0,             # Elective surgery
    "urgent": 0.3174673,       # Urgent surgery
    "emergency": 0.7039121,    # Emergency surgery
    "salvage": 1.362947,       # Salvage/Life-saving surgery
    
    # Weight/Type of procedure coefficients
    "singleCABG": 0,           # Single CABG
    "singlenonCABG": 0.0062118, # Single non-CABG
    "twoprocedure": 0.5521478, # Two procedures
    "threeprocedure": 0.9724533, # Three or more procedures
    "thoracic_aorta": 0.6527205, # Thoracic aorta surgery
    
    # Model constant
    "constant": -5.324537
}

# ============================================================================
# LOGIT SCORE CALCULATION
# ============================================================================

print("Calculating logit scores...")

# Initialize logit with model constant
logit = coefficients["constant"]

# Age adjustment: Special handling for patients ≤60 vs >60 years
# Ages ≤60: use 1, Ages >60: use (age - 60) + 1
logit += np.where(df["age"] <= 60, 1, (df["age"] - 60) + 1) * coefficients["age"]

# ============================================================================
# BINARY RISK FACTORS (0 or 1 indicators)
# ============================================================================

# Add contributions from binary clinical risk factors
logit += df["sex"] * coefficients["female"]
logit += df["iddm"] * coefficients["IDDM"]
logit += df["lung_disease"] * coefficients["CPD"]
logit += df["eca"] * coefficients["ECA"]
logit += df["poor_mobil"] * coefficients["nm_mobility"]
logit += df["redo_proc"] * coefficients["redo"]
logit += df["endocard"] * coefficients["endocarditis"]
logit += df["critical_preop"] * coefficients["critical_state"]
logit += df["ccs_angina"] * coefficients["CCS4"]
logit += df["recentmi"] * coefficients["recent_mi"]
logit += df["surg_thoracicao"] * coefficients["thoracic_aorta"]

# ============================================================================
# CATEGORICAL RISK FACTORS (mapped via lookup dictionaries)
# ============================================================================

# NYHA Functional Class mapping (0=I, 1=II, 2=III, 3=IV)
nyha_map = {
    0: coefficients["NYHA_I"],
    1: coefficients["NYHA_II"],
    2: coefficients["NYHA_III"],
    3: coefficients["NYHA_IV"]
}
logit += df["nyha"].map(nyha_map)

# Renal Impairment based on serum creatinine
# 0: Creatinine >85, 1: Creatinine 50-85, 2: Creatinine <50, 3: On dialysis
renal_map = {
    0: coefficients["ccabove85"],
    1: coefficients["cc5085"],
    2: coefficients["ccbelow50"],
    3: coefficients["On_dialysis"]
}
logit += df["renal_impairment"].map(renal_map)

# Left Ventricular Ejection Fraction (LVEF)
# 0: Good (>50%), 1: Moderate (31-50%), 2: Poor (21-30%), 3: Very Poor (≤20%)
lvef_map = {
    0: coefficients["lvef_good"],
    1: coefficients["lvef_moderate"],
    2: coefficients["lvef_poor"],
    3: coefficients["lvef_very_poor"]
}
logit += df["lv_function"].map(lvef_map)

# Pulmonary Artery Systolic Pressure
# 0: Normal, 1: Moderate, 2: Severe
pah_map = {
    0: coefficients["normal"],
    1: coefficients["moderate"],
    2: coefficients["severe"],
}
logit += df["pulm_ht"].map(pah_map)

# Surgical Urgency classification
# 0: Elective, 1: Urgent, 2: Emergency, 3: Salvage
urgency_map = {
    0: coefficients["elective"],
    1: coefficients["urgent"],
    2: coefficients["emergency"],
    3: coefficients["salvage"]
}
logit += df["urgency"].map(urgency_map)

# Weight/Type of Procedure
# 0: Single CABG, 1: Single non-CABG, 2: Two, 3: Three or more
wop_map = {
    0: coefficients["singleCABG"],
    1: coefficients["singlenonCABG"],
    2: coefficients["twoprocedure"],
    3: coefficients["threeprocedure"]
}
logit += df["weight_of_op"].map(wop_map)

# ============================================================================
# LOGISTIC TRANSFORMATION
# ============================================================================

# Convert logit to probability using logistic function
# P = 1 / (1 + e^(-logit))
print("Converting logit to probability...")
p = 1 / (1 + np.exp(-logit))

# Convert probability to percentage risk (0-100%)
euroscoreii = p * 100

# Display sample results
print("\nEuroSCORE II Results Summary:")
print(euroscoreii.describe())

# ============================================================================
# OUTPUT
# ============================================================================

# Add calculated risk scores to dataframe
df["euroscore_ii"] = euroscoreii

# Save results to CSV
print(f"\nSaving results to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)
print("✓ Calculation complete!")