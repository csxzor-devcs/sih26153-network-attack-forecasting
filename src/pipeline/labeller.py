"""
labeller.py -- Attack stage and MITRE ATT&CK label mapping.

Maps CIC-IDS2017 dataset labels to attack stages,
and then to MITRE ATT&CK tactic identifiers.
"""

import pandas as pd
from typing import Dict, Tuple


def apply_stage_labels(df: pd.DataFrame, stage_map: Dict[str, str]) -> pd.DataFrame:
    """
    Apply attack stage labels to DataFrame based on the 'Label' column.

    Args:
        df: DataFrame with a 'Label' column containing CIC-IDS2017 labels.
        stage_map: Dict mapping CIC-IDS label strings to stage names.

    Returns:
        DataFrame with a new 'stage' column added.

    Notes:
        Rows with labels not found in stage_map get stage='Unknown'.
    """
    df = df.copy()
    df["stage"] = df["Label"].map(stage_map).fillna("Unknown")
    n_unknown = (df["stage"] == "Unknown").sum()
    if n_unknown > 0:
        print(f"[labeller] {n_unknown} rows mapped to 'Unknown' stage")

    print(f"[labeller] Stage distribution:\n{df['stage'].value_counts()}")
    return df


def apply_mitre_labels(df: pd.DataFrame,
                       mitre_map: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """
    Apply MITRE ATT&CK tactic IDs and names based on the 'stage' column.

    Args:
        df: DataFrame with a 'stage' column.
        mitre_map: Dict mapping stage names to (mitre_id, mitre_tactic_name).

    Returns:
        DataFrame with 'mitre_id' and 'mitre_tactic' columns added.
    """
    df = df.copy()
    df["mitre_id"] = df["stage"].map(lambda s: mitre_map.get(s, ("", ""))[0])
    df["mitre_tactic"] = df["stage"].map(lambda s: mitre_map.get(s, ("", ""))[1])

    print(f"[labeller] Applied MITRE labels to {len(df)} rows")
    return df


def get_stage_distribution(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate stage counts and percentages.

    Args:
        df: DataFrame with a 'stage' column.

    Returns:
        Dict mapping stage names to {'count': int, 'percentage': float}.
    """
    total = len(df)
    counts = df["stage"].value_counts()
    distribution = {}
    for stage, count in counts.items():
        distribution[stage] = {
            "count": int(count),
            "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
        }
    return distribution


def print_label_mapping_table(stage_map: Dict[str, str],
                               mitre_map: Dict[str, Tuple[str, str]]) -> None:
    """
    Print a table showing: original label → stage → MITRE tactic.

    Args:
        stage_map: Dict mapping CIC-IDS labels to stage names.
        mitre_map: Dict mapping stage names to (mitre_id, mitre_tactic_name).
    """
    print("\n" + "=" * 90)
    print(f"{'CIC-IDS Label':<30} | {'Stage':<18} | {'MITRE ID':<10} | {'MITRE Tactic'}")
    print("=" * 90)
    for label, stage in sorted(stage_map.items()):
        mitre_id, mitre_name = mitre_map.get(stage, ("", ""))
        print(f"{label:<30} | {stage:<18} | {mitre_id:<10} | {mitre_name}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    from src.config import STAGE_MAP, MITRE_MAP
    print("=== labeller.py self-test ===")
    print_label_mapping_table(STAGE_MAP, MITRE_MAP)
