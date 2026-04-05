# Statement Parser Compliance Rules

This document outlines the strict schema and data integrity constraints that any upstream statement parser (CSV, JSON, PDF extractor) **must** comply with before passing transaction data into the Insight Engine pipeline. 

Failure to adhere to these rules will result in pipeline failure or degraded ML model inference.

## 1. Schema Contract
The parser must output a structured format (e.g., Pandas DataFrame or JSON array of objects) containing **exactly** these four column names (case-sensitive):

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `date` | String / Datetime | Transaction Date. Must be parsable. | `24-03-2024` |
| `amount` | Float | Absolute transaction value. | `1250.75` |
| `amount_flag` | String | Indicator for Debit or Credit. | `DR` |
| `remarks` | String | Raw transaction narrative / Merchant. | `SWIGGY ORDER 123` |

> [!WARNING]
> No extra columns should be passed, and no column can be omitted. **The engine uses exactly `schema.Col.raw_input()`** for validation. Do not pass `signed_amount`, `balance`, or other fields.

## 2. Formatting & Parsing Rules

### 2.1 Timeline / Date Mapping (`date`)
- **Format:** The engine's preprocessor explicitly uses `pd.to_datetime(dayfirst=True)`. Thus, ambiguous dates like `03/04/2024` will be interpreted as **April 3rd**, not March 4th. Ensure your parser standardizes dates unambiguously (ISO-8601 `YYYY-MM-DD` is safest, or strict `DD-MM-YYYY`).

### 2.2 Amount Integrity (`amount`)
- **Absolute Value:** The `amount` column **must** only contain positive numbers (absolute value). The engine natively assigns the mathematical sign based on the `amount_flag`. Passing negative numbers here will cause undefined behavior.
- **Zero Amounts:** Any transaction parsed with an amount of `0.0` will be aggressively dropped by the engine (`_drop_zero_amount`). Do not pass placeholder rows.

### 2.3 Debit/Credit Indicators (`amount_flag`)
- **Allowed Values:** Strictly `"DR"` (Debit) or `"CR"` (Credit).
- The pipeline applies `.strip().upper()`, so `" dr "` or `"Cr "` are acceptable.
- **Failure Condition:** If the parser yields an empty, null, or unrecognized flag, the Engine will forcibly default the row to a Debit (`"DR"`). To prevent silent data corruption, the parser should validate flags strictly instead of relying on this fallback.

### 2.4 Textual Data (`remarks`)
- **Richness:** The categorization and tip-generation ML models heavily rely on keywords in the `remarks`. Do not aggressively truncate this field in the parser.
- **PII:** The Engine's preprocessor naturally sweeps out runs of >=4 digits (UPI references, Phone numbers) and email addresses via `clean_remark()`. The parser **does not** need to scrub these manually, but it should ensure the merchant name text remains heavily intact.

## 3. Row-Level Rules
- **Deduplication:** The Engine automatically drops rows that are exact matches across `(date, amount, remarks, amount_flag)`. Ensure the parser isn't combining separate leg-transactions into duplicates unless they are truly the same swipe.
- **Minimum History:** While passing single transactions is allowed for inference, the engine calculates sliding 30-day volatility scores (`rolling_7d_std`, `amount_zscore`). A parser should ideally supply at least 30 historical transactions for a user to initialize their specific spending baselines before alerting logic activates properly.
