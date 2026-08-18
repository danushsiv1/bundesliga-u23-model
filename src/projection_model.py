import csv
import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np

LEAGUE_TIERS = {
    "Premier League": 1.20,
    "La Liga": 1.15,
    "Ligue 1": 1.05,
    "Serie A": 1.05,
    "Bundesliga": 1.05,
    "Liga Portugal": 0.95,
    "Eredivisie": 0.85,
    "Swiss Super League": 0.85,
    "Aut Bundesliga": 0.85,
    "BEL First Division": 0.85,
    "2 Lg": 0.70,
    "Ligue 2": 0.70,
    "Segunda Liga (ESP)": 0.70,
    "BRA Serie A": 0.70,
    "Allsvenskan": 0.65,
    "Eerste Divisie": 0.60,
    "A-League": 0.60,
    "3 Lg": 0.50,
    "ECU Serie A": 0.50,
    "Slovakia 1 liga": 0.50,
}
TIER_DEFAULT = 0.50

def age_multiplier(row: pd.Series) -> float:
    age = row["Age"]
    pos = row["PosGroup"]
    
    if pos == "CB":
        if age <= 18: return 1.30
        if age == 19: return 1.15
        if age == 20: return 1.05
        if age == 21: return 1.00
        if age == 22: return 0.95
        return 0.90
    elif pos == "FW":
        if age <= 17: return 1.30
        if age <= 19: return 1.15
        if age == 20: return 1.05
        if age == 21: return 1.00
        if age == 22: return 0.95
        return 0.90
    else: # Midfielders
        if age <= 17: return 1.30
        if age == 18: return 1.225
        if age == 19: return 1.15
        if age == 20: return 1.05
        if age == 21: return 1.025
        if age == 22: return 0.975
        return 0.925


def league_multiplier(league: str) -> float:
    if pd.isna(league):
        return TIER_DEFAULT
    return LEAGUE_TIERS.get(str(league).strip(), TIER_DEFAULT)


NUMERIC_COLS = [
    "Age", "Born", "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A",
    "G-PK", "PK", "PKatt", "CrdY", "CrdR", "Fls", "Fld", "Off", "Crs",
    "Int", "TklW", "Accurate passes per 90", "Chances Created",
    "Possession won final 3rd per 90",
]


def load_and_normalize_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    col_mapping = {
        "Player Name": "Player",
        "Squad Name": "Squad",
        "Comp": "25/26 Lg",
        "Minutes": "Min",
        "Matches Played": "MP"
    }
    df = df.rename(columns=col_mapping)

    if "Pos" in df.columns:
        def map_pos(p):
            p_str = str(p).upper()
            if "," in p_str:
                p_str = p_str.split(",")[0].strip()
            
            # Value all defenders and full-backs as Center-Backs (ignore FBs)
            if "CB" in p_str or "DF" in p_str or "FB" in p_str or "LB" in p_str or "RB" in p_str or "WB" in p_str or "CENTER" in p_str:
                return "CB"
            if "MF" in p_str or "MID" in p_str:
                return "MF"
            return "FW"
        df["PosGroup"] = df["Pos"].apply(map_pos)
    elif "PosGroup" not in df.columns:
        df["PosGroup"] = "MF"
        df["Pos"] = "MF"

    if "25/26 Lg" not in df.columns:
        df["25/26 Lg"] = "Bundesliga"

    if "Squad" not in df.columns:
        df["Squad"] = "Unknown Squad"

    if "Player" not in df.columns:
        df["Player"] = [f"Player_{i}" for i in range(len(df))]

    if "Age" not in df.columns:
        df["Age"] = 20

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"-": np.nan, "": np.nan, "nan": np.nan})
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df["90s"] = df["90s"].fillna(df["Min"] / 90.0 if "Min" in df.columns else 1.0).clip(lower=0.3)
    df["Min"] = df["Min"].fillna(df["90s"] * 90.0).clip(lower=27.0)
    df["MP"] = df["MP"].fillna(df["90s"]).clip(lower=1.0)

    for col in NUMERIC_COLS:
        if col in df.columns:
            group_median = df.groupby("PosGroup")[col].transform("median")
            df[col] = df[col].fillna(group_median).fillna(0.0)

    return df


def build_rate(df: pd.DataFrame, total_col: str, already_per90: bool = False, shrink_k: float = 6.0) -> pd.Series:
    if already_per90:
        raw = df[total_col]
        weight = df["90s"] / (df["90s"] + shrink_k)
        group_mean = raw.mean(skipna=True)
        raw = raw.fillna(group_mean)
        return weight * raw + (1 - weight) * group_mean
    else:
        totals = df[total_col].fillna(0)
        raw_rate = totals / df["90s"]
        group_mean = raw_rate.replace([np.inf, -np.inf], np.nan).mean(skipna=True)
        if math.isnan(group_mean):
            group_mean = 0.0
        shrunk = (totals + shrink_k * group_mean) / (df["90s"] + shrink_k)
        return shrunk


def build_rate_per_min(df: pd.DataFrame, total_col: str, shrink_k: float = 6.0) -> pd.Series:
    totals = df[total_col].fillna(0)
    mins = df["Min"]
    raw_rate = totals / mins
    group_mean = raw_rate.replace([np.inf, -np.inf], np.nan).mean(skipna=True)
    if math.isnan(group_mean):
        group_mean = 0.0
    shrink_k_mins = shrink_k * 90.0
    shrunk = (totals + shrink_k_mins * group_mean) / (mins + shrink_k_mins)
    return shrunk


def minmax_0_100(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9 or math.isnan(lo) or math.isnan(hi):
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def score_cb(df: pd.DataFrame) -> pd.Series:
    DEF_SHRINK = 3.0
    tklw = build_rate(df, "TklW", shrink_k=DEF_SHRINK)
    intc = build_rate(df, "Int", shrink_k=DEF_SHRINK)
    passes = build_rate(df, "Accurate passes per 90", already_per90=True, shrink_k=DEF_SHRINK)
    ga = build_rate(df, "G+A", shrink_k=DEF_SHRINK)
    fls = build_rate(df, "Fls", shrink_k=DEF_SHRINK)
    crdy = build_rate(df, "CrdY", shrink_k=DEF_SHRINK)
    crdr = build_rate(df, "CrdR", shrink_k=DEF_SHRINK)

    fls_n = minmax_0_100(fls)
    crdy_n = minmax_0_100(crdy)
    discipline = 100 - (0.70 * fls_n + 0.30 * crdy_n)
    discipline = discipline - (crdr * 25)
    discipline = discipline.clip(lower=0, upper=100)

    return (
        0.25 * minmax_0_100(tklw)
        + 0.25 * minmax_0_100(intc)
        + 0.35 * minmax_0_100(passes)
        + 0.10 * minmax_0_100(ga)
        + 0.05 * discipline
    )


def score_mf(df: pd.DataFrame) -> pd.Series:
    cc = build_rate_per_min(df, "Chances Created")
    passes = build_rate(df, "Accurate passes per 90", already_per90=True)
    ga = build_rate(df, "G+A")
    intc = build_rate(df, "Int")
    tklw = build_rate(df, "TklW")
    poss_won = build_rate(df, "Possession won final 3rd per 90", already_per90=True)

    return (
        0.30 * minmax_0_100(cc)
        + 0.30 * minmax_0_100(passes)
        + 0.15 * minmax_0_100(ga)
        + 0.10 * minmax_0_100(intc)
        + 0.10 * minmax_0_100(tklw)
        + 0.05 * minmax_0_100(poss_won)
    )


def score_fw(df: pd.DataFrame) -> pd.Series:
    cc = build_rate_per_min(df, "Chances Created")
    df["Raw_NPG_A"] = df["G-PK"].fillna(0) + df["Ast"].fillna(0)
    npg_a_rate = build_rate_per_min(df, "Raw_NPG_A")
    npg_a_score = 0.50 * minmax_0_100(npg_a_rate) + 0.50 * minmax_0_100(df["Raw_NPG_A"])

    passes = build_rate(df, "Accurate passes per 90", already_per90=True)
    fld = build_rate(df, "Fld")
    tklw = build_rate(df, "TklW")
    intc = build_rate(df, "Int")

    pk = df["PK"].fillna(0)
    pkatt = df["PKatt"].fillna(0)
    pk_rate = np.where(pkatt > 0, pk / pkatt, np.nan)
    pk_rate = pd.Series(pk_rate, index=df.index)
    league_pk_mean = pk_rate.mean(skipna=True)
    if math.isnan(league_pk_mean):
        league_pk_mean = 0.78
    pk_rate_filled = pk_rate.fillna(league_pk_mean)
    pk_score = minmax_0_100(pk_rate_filled)

    return (
        0.33 * minmax_0_100(cc)
        + 0.27 * npg_a_score
        + 0.09 * minmax_0_100(passes)
        + 0.20 * minmax_0_100(fld)
        + 0.07 * minmax_0_100(tklw + intc)
        + 0.04 * pk_score
    )


SCORERS = {"CB": score_cb, "MF": score_mf, "FW": score_fw}


def build_leaderboard(target_csv: str = "all_players_cleaned.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    target_path = Path(target_csv)
    if not target_path.exists():
        raise FileNotFoundError(f"Could not find target file: {target_path}")
        
    raw_df = pd.read_csv(target_path)
    df = load_and_normalize_data(target_path)
    
    composite_scores = pd.Series(0.0, index=df.index)
    for pos_group, group_indices in df.groupby("PosGroup").groups.items():
        scorer_func = SCORERS.get(pos_group, score_mf)
        composite_scores.loc[group_indices] = scorer_func(df.loc[group_indices])
    df["PositionComposite"] = composite_scores.round(1)

    all_df = df
    all_df["AgeMultiplier"] = all_df.apply(age_multiplier, axis=1)
    all_df["LeagueMultiplier"] = all_df["25/26 Lg"].apply(league_multiplier)

    # Make Matches Played (MP) a stronger factor in the final score calculation
    max_mp = all_df["MP"].max() if "MP" in all_df.columns and all_df["MP"].max() > 0 else 34.0
    all_df["MP_Factor"] = (all_df["MP"].fillna(1) / max_mp).clip(lower=0.2, upper=1.0)

    all_df["RawFinalScore"] = (
        all_df["PositionComposite"] * all_df["AgeMultiplier"] * all_df["LeagueMultiplier"] * (0.5 + 0.5 * all_df["MP_Factor"])
    )

    all_df["Projection Score"] = minmax_0_100(all_df["RawFinalScore"]).round(1)

    all_df = all_df.sort_values("Projection Score", ascending=False).reset_index(drop=True)
    all_df.insert(0, "Rank", all_df.index + 1)

    for col in raw_df.columns:
        if col not in all_df.columns:
            all_df[col] = raw_df[col]

    model_cols = ["Rank", "Projection Score", "PositionComposite", "AgeMultiplier", "LeagueMultiplier"]
    original_cols = [c for c in raw_df.columns if c in all_df.columns and c not in model_cols]
    
    final_cols = model_cols + original_cols

    return all_df[final_cols], all_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = input("Enter the path to your CSV file [default: all_players_cleaned.csv]: ").strip()
        if not input_csv:
            input_csv = "all_players_cleaned.csv"
        
    print("\n" + "="*80)
    print(f" 🚀 RUNNING U23 PROJECTION MODEL ON: {input_csv} ")
    print("="*80)
    
    leaderboard, full = build_leaderboard(input_csv)
    
    Path("output").mkdir(parents=True, exist_ok=True)
    output_path = "output/u23_projection_leaderboard.csv"
    leaderboard.to_csv(output_path, index=False)
    
    limit = min(20, len(leaderboard))
    print(f"\n📊 TOP {limit} PLAYERS RANKED:")
    
    display_cols = ["Rank", "Projection Score", "Player", "Pos", "Squad", "25/26 Lg", "Age", "PositionComposite"]
    sub_df = leaderboard[display_cols].head(limit).copy()
    for col in display_cols:
        sub_df[col] = sub_df[col].astype(str)
        
    col_widths = {col: max(len(col), sub_df[col].str.len().max()) for col in display_cols}
    
    header = " | ".join([f"{col:<{col_widths[col]}}" for col in display_cols])
    divider = "-" * len(header)
    
    print(divider)
    print(header)
    print(divider)
    for _, row in sub_df.iterrows():
        row_str = " | ".join([f"{row[col]:<{col_widths[col]}}" for col in display_cols])
        print(row_str)
    print(divider)
    
    print(f"✨ Total players processed & ranked: {len(leaderboard)}")
    print(f"📁 Full dataset saved cleanly to: {output_path}")
    print("="*80 + "\n")