"""
generate_sample_data.py
------------------------
Generates two sample CSV files used to demo the AI Finance Controller:
  1. ledger.csv          -> company's internal accounting ledger
  2. bank_statement.csv  -> bank statement for the same period

Run this once to (re)create fresh demo data:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

VENDORS = [
    "Amazon Web Services", "Reliance Office Supplies", "Tata Power",
    "Zomato Corporate", "Ola Fleet", "Airtel Business", "Microsoft Azure",
    "Swiggy Genie", "Godrej Interio", "IndiGo Airlines", "WeWork India",
    "Google Ads", "LinkedIn Talent", "Adobe Creative Cloud", "Uber for Business"
]

CATEGORIES = {
    "Amazon Web Services": "Cloud & Software",
    "Microsoft Azure": "Cloud & Software",
    "Google Ads": "Marketing",
    "LinkedIn Talent": "HR & Recruiting",
    "Adobe Creative Cloud": "Cloud & Software",
    "Reliance Office Supplies": "Office Supplies",
    "Tata Power": "Utilities",
    "Zomato Corporate": "Meals & Entertainment",
    "Ola Fleet": "Travel",
    "Uber for Business": "Travel",
    "Airtel Business": "Utilities",
    "Swiggy Genie": "Logistics",
    "Godrej Interio": "Office Supplies",
    "IndiGo Airlines": "Travel",
    "WeWork India": "Rent & Facilities",
}

def random_dates(n, start, end):
    delta = (end - start).days
    return [start + timedelta(days=np.random.randint(0, delta)) for _ in range(n)]

def generate_ledger(n=120, start=datetime(2026, 6, 1), end=datetime(2026, 8, 25)):
    vendors = np.random.choice(VENDORS, n)
    amounts = np.round(np.random.gamma(shape=2.2, scale=6500, size=n), 2)
    dates = sorted(random_dates(n, start, end))
    df = pd.DataFrame({
        "txn_id": [f"LGR-{1000+i}" for i in range(n)],
        "date": dates,
        "vendor": vendors,
        "category": [CATEGORIES[v] for v in vendors],
        "amount": amounts,
        "type": "Debit",
        "recorded_by": np.random.choice(["Aditya", "Priya", "System-Auto"], n, p=[0.4, 0.3, 0.3]),
    })
    return df

def generate_bank_statement(ledger_df, missing_rate=0.08, extra_rate=0.05, tamper_rate=0.06):
    """
    Simulate a real bank statement derived from the ledger, but with realistic
    discrepancies an AI finance controller should catch:
      - some ledger entries missing from the bank feed (in-transit / unrecorded)
      - a few extra bank charges not in the ledger (bank fees, unexplained debits)
      - a few amounts slightly tampered (typo / duplicate / partial payment)
    """
    df = ledger_df.copy()

    # Drop some rows entirely (simulate timing differences / omissions)
    drop_mask = np.random.rand(len(df)) < missing_rate
    df = df[~drop_mask].reset_index(drop=True)

    # Tamper a few amounts (small discrepancies AI should flag)
    tamper_idx = df.sample(frac=tamper_rate, random_state=1).index
    df.loc[tamper_idx, "amount"] = (df.loc[tamper_idx, "amount"] * np.random.uniform(0.85, 1.25, len(tamper_idx))).round(2)

    bank_df = pd.DataFrame({
        "bank_ref": [f"BNK-{5000+i}" for i in range(len(df))],
        "value_date": df["date"],
        "narration": df["vendor"] + " - PAYMENT",
        "debit_amount": df["amount"],
    })

    # Add extra bank-only charges (fees, unexplained debits)
    n_extra = int(len(df) * extra_rate)
    extra_dates = random_dates(n_extra, df["date"].min(), df["date"].max())
    extra = pd.DataFrame({
        "bank_ref": [f"BNK-EXTRA-{i}" for i in range(n_extra)],
        "value_date": extra_dates,
        "narration": np.random.choice(
            ["BANK SERVICE CHARGE", "NEFT PROCESSING FEE", "UNIDENTIFIED DEBIT", "SMS ALERT CHARGE"],
            n_extra,
        ),
        "debit_amount": np.round(np.random.uniform(50, 900, n_extra), 2),
    })

    bank_df = pd.concat([bank_df, extra], ignore_index=True).sort_values("value_date").reset_index(drop=True)
    return bank_df

if __name__ == "__main__":
    ledger = generate_ledger()
    bank = generate_bank_statement(ledger)

    ledger.to_csv("ledger.csv", index=False)
    bank.to_csv("bank_statement.csv", index=False)

    print(f"Generated ledger.csv with {len(ledger)} rows")
    print(f"Generated bank_statement.csv with {len(bank)} rows")
