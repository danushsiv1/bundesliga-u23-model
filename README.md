# U23 Cross-Position Football Projection Model

## Overview
This model ingests four position-split scouting datasets (Center-Backs, Full-Backs, Midfielders, and Forwards) and processes them through a rigorous, multi-layered analytical pipeline. Its ultimate goal is to produce a single, unified **0-100 Projection Leaderboard** that fairly compares players across completely different positions, ages, and league competitive environments.

---

## The 4-Layer Architecture

### 1. Position-Specific Skill Composites (0–100 Scaled)
Football cannot be judged by a single metric. A center-back succeeds through defensive intervention and ball retention, while a forward succeeds through chance creation and dynamic goal threat. Each position group utilizes a custom-weighted matrix of per-minute and per-90 metrics:

* **Forwards (FW):** Heavily favors Chance Created (33%), Non-Penalty Goals + Assists blend (27%), Fouls Drawn (20%), Accurate Passes (9%), Defensive Tracking/Tackles (7%), and Penalty Conversion (4%).
* **Midfielders (MF):** Balanced heavily toward Chance Created (30%) and Accurate Passes (30%), supported by G+A (15%), Interceptions (10%), Tackles Won (10%), and Final-Third Possession Won (5%).
* **Full-Backs (FB):** Built to mirror modern wide roles—emphasizing Chances Created (22%), Interceptions (18%), Tackles Won (18%), Assists (15%), Accurate Passes (12%), Final-Third Possession Won (10%), and Goals (5%).
* **Center-Backs (CB):** Prioritizes distribution and defensive bedrock—Accurate Passes (35%), Tackles Won (25%), Interceptions (25%), Goals+Assists (10%), and a heavily penalized Discipline metric (5%) tracking foul volume and red cards.

### 2. Empirical-Bayes Shrinkage (Sample Size Control)
Young players often accumulate noisy data over small sample sizes (e.g., a 17-year-old who plays 150 brilliant minutes and looks like a world-beater on a per-90 basis). 
* To prevent small-sample bias, raw metrics are "shrunk" toward their positional group average using an Empirical-Bayes prior.
* **Attackers/Midfielders** use a standard shrinkage weight ($K = 6.0$ matches).
* **Defenders (CB/FB)** stabilize faster in match-to-match actions, so their shrinkage is lowered to $K = 3.0$ to ensure young starting defenders aren't unfairly penalized for minor sample size constraints.

### 3. Position-Aware Age Leverage Multipliers
Real-world scouting acknowledges that players mature at different rates depending on their position. Wingers and attackers frequently break out at 17 or 18, whereas center-backs require physical and tactical maturity. 
The model applies distinct age-leverage curves:
* **Forwards:** Peak youth leverage starts at age 17 ($1.30\times$) and steps down gradually.
* **Defenders (CB/FB):** Given an extended runway. An 18-year-old defender and a 16-year-old forward both receive the maximum $1.30\times$ multiplier, and a 20-year-old starting center-back retains a strong $1.15\times$ ceiling boost.
* **Midfielders:** Mapped precisely in the mathematical middle between the defensive and offensive curves.

### 4. League Hierarchy Multipliers
Raw production must be contextualized by the strength of the opposition. Leagues are tiered so that identical statistics compiled in elite environments are rewarded more heavily than in smaller domestic leagues:
* **Tier 1 (Premier League - $1.20\times$, La Liga - $1.15\times$)**
* **Tier 2 (Bundesliga, Serie A, Ligue 1 - $1.05\times$)**
* **Tier 3 (Liga Portugal - $0.95\times$; Eredivisie, Swiss Super League, Austrian Bundesliga, Belgian First Division - $0.85\times$; 2. Bundesliga, Ligue 2, Segunda División, Brazilian Série A - $0.70\times$)**
* **Tier 4 & Lower (Allsvenskan - $0.65\times$; Eerste Divisie, A-League - $0.60\times$; 3. Liga, Ecuadorian Série A, Slovak 1. Liga - $0.50\times$)**

---

## Final Calculation & Normalization
For every player, a raw projection score is calculated:
$$\text{Raw Score} = \text{Position Composite} \times \text{Age Multiplier} \times \text{League Multiplier}$$

This raw output is then min-max normalized across the entire cross-position player pool into a clean, readable **0-100 Projection Score**.

## Execution
Run the script from your terminal using:
```bash
python3 projection_model.py