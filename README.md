# ⚽ U23 Cross-Position Football Projection Model

*An End-to-End Data Science Pipeline Decoding Scouting Metrics, Empirical-Bayes Shrinkage, Age Curves, and League Strength to Project Young Football Talent.*

---

## 📊 Data Source & Acknowledgments

The underlying quantitative analysis ingests standardized scouting datasets split across four positional groups: **Center-Backs, Full-Backs, Midfielders, and Forwards**. These datasets provide the per-minute and per-90 performance metrics required to evaluate young players across different positions, ages, and league environments.

The model processes these datasets through a unified analytical framework designed to produce a single **0–100 Projection Score** for every eligible player.

---

## 🚀 Executive Summary & Project Goal

In modern football scouting, comparing young prospects across different positions and leagues is difficult because raw statistics do not exist in a vacuum.

A forward's attacking production cannot be directly compared to a center-back's defensive output, while statistics produced in the Premier League should not necessarily carry the same weight as identical numbers produced in a lower-tier league.

This project moves beyond raw statistical comparisons by combining **position-specific skill composites, Empirical-Bayes shrinkage, position-aware age multipliers, and league-strength adjustments**.

Designed as a strategic decision-making engine for **scouts, analysts, and sporting departments**, the model produces a unified **0–100 Projection Leaderboard** that contextualizes player production across positions, ages, and competitive environments.

---

## 🔍 Analytical Framework: The 4-Layer Architecture

The projection model is built around four analytical layers that progressively transform raw scouting data into a standardized cross-position projection score:

### 1. ⚽ Position-Specific Skill Composites (0–100 Scaled)

Football cannot be judged by a single metric. Each position group utilizes a custom-weighted matrix of per-minute and per-90 metrics designed around the responsibilities of that position:

* **Forwards (FW):** Heavily favors Chance Created (33%), Non-Penalty Goals + Assists blend (27%), Fouls Drawn (20%), Accurate Passes (9%), Defensive Tracking/Tackles (7%), and Penalty Conversion (4%).
* **Midfielders (MF):** Balanced heavily toward Chance Created (30%) and Accurate Passes (30%), supported by G+A (15%), Interceptions (10%), Tackles Won (10%), and Final-Third Possession Won (5%).
* **Full-Backs (FB):** Built to mirror modern wide roles—emphasizing Chances Created (22%), Interceptions (18%), Tackles Won (18%), Assists (15%), Accurate Passes (12%), Final-Third Possession Won (10%), and Goals (5%).
* **Center-Backs (CB):** Prioritizes distribution and defensive bedrock—Accurate Passes (35%), Tackles Won (25%), Interceptions (25%), Goals+Assists (10%), and a heavily penalized Discipline metric (5%) tracking foul volume and red cards.

### 2. 📐 Empirical-Bayes Shrinkage (Sample Size Control)

Young players often accumulate noisy data over small sample sizes. A 17-year-old who plays only 150 brilliant minutes, for example, can appear dramatically better on a per-90 basis than a player with a full season of production.

To prevent small-sample bias, raw metrics are **shrunk toward their positional group average** using an Empirical-Bayes prior.

* **Attackers/Midfielders:** Standard shrinkage weight of $K = 6.0$ matches.
* **Defenders (CB/FB):** Lower shrinkage weight of $K = 3.0$ matches, allowing young starting defenders to stabilize more quickly in match-to-match actions.

This creates a more reliable representation of player ability when sample sizes are limited.

### 3. 📈 Position-Aware Age Leverage Multipliers

Player development does not occur at the same rate across every position. Wingers and attackers can break out at 17 or 18, while center-backs typically require additional physical and tactical maturity.

The model therefore applies distinct age-leverage curves:

* **Forwards:** Peak youth leverage starts at age 17 with a maximum $1.30\times$ multiplier and steps down gradually.
* **Defenders (CB/FB):** Given an extended development runway. An 18-year-old defender receives the maximum $1.30\times$ multiplier, while a 20-year-old starting center-back retains a strong $1.15\times$ ceiling boost.
* **Midfielders:** Positioned mathematically between the defensive and offensive age curves.

### 4. 🌍 League Hierarchy Multipliers

Raw production must be contextualized by the strength of the opposition. The model assigns league-strength multipliers so that identical statistics produced in stronger competitive environments receive greater weight.

* **Tier 1:** Premier League ($1.20\times$), La Liga ($1.15\times$)
* **Tier 2:** Bundesliga, Serie A, Ligue 1 ($1.05\times$)
* **Tier 3:** Liga Portugal ($0.95\times$); Eredivisie, Swiss Super League, Austrian Bundesliga, Belgian First Division ($0.85\times$); 2. Bundesliga, Ligue 2, Segunda División, Brazilian Série A ($0.70\times$)
* **Tier 4 & Lower:** Allsvenskan ($0.65\times$); Eerste Divisie, A-League ($0.60\times$); 3. Liga, Ecuadorian Série A, Slovak 1. Liga ($0.50\times$)

---

## 🧮 Final Calculation & Normalization

For every player, the model first calculates a raw projection score:

$$\text{Raw Score} = \text{Position Composite} \times \text{Age Multiplier} \times \text{League Multiplier}$$

This raw output is then **min-max normalized across the entire cross-position player pool** into a clean, readable **0–100 Projection Score**.

The resulting score allows players from different positions and competitive environments to be ranked on a single unified leaderboard.

---

## 💻 How to Use and Run the Project in the Terminal

You can execute, test, and run the complete player projection pipeline directly from your command line.

### 1. Install Dependencies

Ensure your Python environment is set up with the required packages:

```bash
pip install pandas numpy
```

### 2. Run the Interactive Top 20 Leaderboard

To run the model on your dataset and output a clean, formatted terminal view of the top prospects:

```bash
python3 projection_model.py
```

When prompted, enter or paste the path to your CSV file, such as:

```text
all_players_cleaned.csv
```

### 3. Generate the Full Leaderboard CSV

To process the entire player pool and export every ranked player into a clean output file:

```bash
python3 run_full_leaderboard.py
```

The resulting CSV contains the final ranked players and their projection metrics.

---

## 🛠️ Technical Architecture & Repository Structure

```text
bundesliga-u23-model/
│
├── all_players_cleaned.csv          # Master scouting dataset
├── 2016_17_u23_players.csv          # Historical validation dataset
│
├── projection_model.py              # Core 4-layer projection pipeline & terminal display engine
├── run_full_leaderboard.py          # Batch runner script for complete CSV export
│
├── output/
│   └── u23_projection_leaderboard.csv  # Final ranked export containing all metrics
│
└── README.md                        # Executive project documentation
```

---

*Built with Python (Pandas, NumPy) and a multi-layer football scouting projection framework.*
