# EuroSCORE II Calculator

A Python script for **research purposes only** that calculates EuroSCORE II from CSV data and helps simplify REDCap data collection and preprocessing. The script reads patient-level data, validates required fields, applies the EuroSCORE II coefficients and logistic formula, and writes the calculated score to a new CSV file.

## Overview

This script is intended to support research workflows, not clinical decision-making. It can be used to standardize EuroSCORE II data entry, reduce manual calculation steps, and prepare datasets for REDCap-based research projects.

The details of the EuroSCORE II model, including the variables and coefficients used in the calculation, can be found on the official EuroSCORE website or in the original publication by Nashef et al.

## Requirements

- Python 3.x
- pandas
- numpy

Install dependencies using:
```bash
pip install pandas numpy
```

## Input Data

**File:** `EuroSCORE_DATA.csv`

The input CSV must contain the following columns:

### Patient Demographics & IDs
- `record_id` - Unique patient identifier
- `age` - Patient age in years
- `sex` - Gender (0 = male, 1 = female)

### Clinical Risk Factors
- `lung_disease` - Chronic Pulmonary Disease (0 = no, 1 = yes)
- `eca` - Extra-cardiac Arteriopathy (0 = no, 1 = yes)
- `poor_mobil` - Poor Mobility (0 = no, 1 = yes)
- `redo_proc` - Previous Cardiac Surgery (0 = no, 1 = yes)
- `endocard` - Active Endocarditis (0 = no, 1 = yes)
- `critical_preop` - Critical Preoperative State (0 = no, 1 = yes)
- `iddm` - Insulin-Dependent Diabetes Mellitus (0 = no, 1 = yes)
- `ccs_angina` - CCS Class 4 Angina (0 = no, 1 = yes)
- `recentmi` - Recent Myocardial Infarction (0 = no, 1 = yes)
- `surg_thoracicao` - Thoracic Aorta Surgery (0 = no, 1 = yes)

### Functional & Hemodynamic Status
- `lv_function` - Left Ventricular Function (0 = good, 1 = moderate, 2 = poor, 3 = very poor)
- `nyha` - NYHA Functional Class (0 = I, 1 = II, 2 = III, 3 = IV)
- `pulm_ht` - Pulmonary Hypertension (0 = normal, 1 = moderate, 2 = severe)

### Operative Factors
- `urgency` - Urgency of Surgery (0 = elective, 1 = urgent, 2 = emergency, 3 = salvage)
- `weight_of_op` - Weight/Type of Procedure (0 = single CABG, 1 = single non-CABG, 2 = two procedures, 3 = three or more procedures)

### Renal Function
- `renal_impairment` - Renal Impairment (0 = creatinine >85, 1 = creatinine 50-85, 2 = creatinine <50, 3 = on dialysis)

### Output Column
- `euroscore_ii` - Will be populated with calculated risk scores (%) after running the script

## Usage

1. Prepare your input data in CSV format with all required columns listed above
2. Save it as `EuroSCORE_DATA.csv` in the same directory as the script
3. Run the script:
   ```bash
   python EuroSCORE.py
   ```
4. The script will output a new file: `EuroSCORE_DATA_with_euroscore.csv` containing the original data with an additional `euroscore_ii` column

## Data Processing

- Rows with missing values in required input fields are removed before calculation.
- The script validates that all required columns are present.
- EuroSCORE II is calculated using the published logistic regression formula.

## Algorithm Notes

The EuroSCORE II calculation involves:
1. Computing a logit score from patient-specific coefficients.
2. Applying the age adjustment.
3. Converting the logit to probability.
4. Converting the probability to percentage.

## Output Example

| record_id | age | sex | ... | euroscore_ii |
|-----------|-----|-----|-----|--------------|
| 001       | 65  | 0   | ... | 2.45         |
| 002       | 72  | 1   | ... | 8.32         |
| 003       | 58  | 0   | ... | 1.89         |

## References

- Nashef SAM, et al. EuroSCORE II. Eur J Cardiothorac Surg. 2012;41(4):734-744.
- https://www.euroscore.org/

## License

MIT License

## Author

Bence Jonas

## Support

For questions or issues, please contact.
