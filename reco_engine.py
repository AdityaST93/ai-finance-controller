"""
reco_engine.py
---------------
Core logic for the AI Finance Controller:
  1. Reconciliation: matches ledger entries against bank statement entries
     using vendor-name similarity + date proximity + amount tolerance.
  2. Anomaly detection: flags unusually large transactions, duplicate
     payments, and vendor spend spikes.
  3. Recommendations: simple rule-based "AI insights" (can be swapped for
     an LLM call later) that summarize findings in plain language.
"""

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(s: str) -> str:
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _clean_text(a), _clean_text(b)).ratio()


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile(ledger: pd.DataFrame, bank: pd.DataFrame,
              date_tolerance_days: int = 3,
              amount_tolerance_pct: float = 0.02,
              text_similarity_threshold: float = 0.35) -> pd.DataFrame:
    """
    Attempts to match every ledger row to a bank_statement row.

    Returns a DataFrame with one row per ledger entry plus match status:
      - "Matched"        : found a confident bank match
      - "Amount Mismatch": vendor/date matched but amount differs beyond tolerance
      - "Unmatched"      : no plausible bank entry found (possible unrecorded/fraud risk)
    """
    ledger = ledger.copy()
    bank = bank.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
    bank["value_date"] = pd.to_datetime(bank["value_date"])

    bank_used = set()
    results = []

    for _, row in ledger.iterrows():
        best_idx, best_score = None, -1.0
        for b_idx, b_row in bank.iterrows():
            if b_idx in bank_used:
                continue
            day_diff = abs((row["date"] - b_row["value_date"]).days)
            if day_diff > date_tolerance_days:
                continue
            text_sim = _similarity(row["vendor"], b_row["narration"])
            if text_sim < text_similarity_threshold:
                continue
            score = text_sim - (day_diff * 0.02)
            if score > best_score:
                best_score, best_idx = score, b_idx

        if best_idx is not None:
            bank_row = bank.loc[best_idx]
            bank_used.add(best_idx)
            pct_diff = abs(bank_row["debit_amount"] - row["amount"]) / max(row["amount"], 1e-6)
            status = "Matched" if pct_diff <= amount_tolerance_pct else "Amount Mismatch"
            results.append({
                **row.to_dict(),
                "bank_ref": bank_row["bank_ref"],
                "bank_amount": bank_row["debit_amount"],
                "amount_diff": round(bank_row["debit_amount"] - row["amount"], 2),
                "status": status,
            })
        else:
            results.append({
                **row.to_dict(),
                "bank_ref": None,
                "bank_amount": None,
                "amount_diff": None,
                "status": "Unmatched",
            })

    matched_df = pd.DataFrame(results)

    unexplained_bank = bank[~bank.index.isin(bank_used)].copy()
    unexplained_bank["status"] = "Bank-Only (Not in Ledger)"

    return matched_df, unexplained_bank


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(ledger: pd.DataFrame, z_threshold: float = 2.2) -> pd.DataFrame:
    df = ledger.copy()
    df["date"] = pd.to_datetime(df["date"])
    mean, std = df["amount"].mean(), df["amount"].std()
    df["z_score"] = (df["amount"] - mean) / (std if std > 0 else 1)
    df["is_outlier"] = df["z_score"].abs() > z_threshold

    dup_cols = ["vendor", "amount"]
    df["is_possible_duplicate"] = df.duplicated(subset=dup_cols, keep=False)

    return df


def vendor_spend_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ledger.groupby("vendor")["amount"]
        .agg(total_spend="sum", txn_count="count", avg_txn="mean")
        .sort_values("total_spend", ascending=False)
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Rule-based "AI" recommendations
# ---------------------------------------------------------------------------

def generate_recommendations(matched_df: pd.DataFrame, unexplained_bank: pd.DataFrame,
                              anomalies_df: pd.DataFrame) -> list[str]:
    tips = []

    unmatched = matched_df[matched_df["status"] == "Unmatched"]
    mismatched = matched_df[matched_df["status"] == "Amount Mismatch"]

    if len(unmatched) > 0:
        total = unmatched["amount"].sum()
        tips.append(
            f"⚠️ {len(unmatched)} ledger entries (₹{total:,.0f} total) have no matching bank "
            f"transaction. Verify these aren't unrecorded payments or duplicate journal entries."
        )

    if len(mismatched) > 0:
        total_diff = mismatched["amount_diff"].abs().sum()
        tips.append(
            f"🔍 {len(mismatched)} transactions show amount mismatches between ledger and bank "
            f"(₹{total_diff:,.0f} combined difference). Check for partial payments, bank fees, "
            f"or data-entry typos."
        )

    if len(unexplained_bank) > 0:
        total = unexplained_bank["debit_amount"].sum()
        tips.append(
            f"🏦 {len(unexplained_bank)} bank debits (₹{total:,.0f} total) don't appear in your "
            f"ledger at all — likely bank charges or fees that should be recorded as expenses."
        )

    outliers = anomalies_df[anomalies_df["is_outlier"]]
    if len(outliers) > 0:
        top = outliers.sort_values("amount", ascending=False).iloc[0]
        tips.append(
            f"📈 Unusually large transaction detected: ₹{top['amount']:,.0f} to "
            f"'{top['vendor']}' on {pd.to_datetime(top['date']).strftime('%d %b %Y')}. "
            f"This is a statistical outlier vs. typical spend — worth a second look."
        )

    dupes = anomalies_df[anomalies_df["is_possible_duplicate"]]
    if len(dupes) > 0:
        tips.append(
            f"🧾 {len(dupes)} transactions look like possible duplicate payments (same vendor, "
            f"same amount). Review to avoid double-paying a vendor."
        )

    if not tips:
        tips.append("✅ No major issues detected. Ledger and bank statement are well reconciled.")

    return tips
