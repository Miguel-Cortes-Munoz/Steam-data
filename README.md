# Predicting Steam Game Reception

A machine learning project that predicts whether a Steam game will be well received by players, using logistic regression trained on 40,000+ games. Built with Python, pandas, and scikit-learn.

---

## Overview

Can observable game attributes — price, genre, tags, platform support, playtime — predict whether a game will be well received by the Steam community? This project answers that question using real Steam data, with practical takeaways for indie developers deciding how to position a new title.

**Target variable:** Wilson Score ≥ 0.75 (binary classification)

The Wilson Score is a statistically robust measure of review quality that accounts for total review volume, giving a fairer estimate for games with fewer reviews than a simple positive-ratio would.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.67 |
| ROC-AUC | 0.7253 |
| Precision (well-received) | 0.65 |
| Recall (well-received) | 0.55 |
| F1 (well-received) | 0.60 |

The model correctly classifies 67% of games and achieves an AUC of 0.73 — meaningfully above random. It is better at identifying games that will *not* be well received (recall 0.77), which reflects how difficult it is to predict positive reception from metadata alone.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data wrangling and feature engineering |
| scikit-learn | Logistic regression, train/val/test split, AUC |
| matplotlib / seaborn | EDA visualizations |
| scipy | Chi-square test, Cramér's V |

---

## Dataset

- **Source:** Steam Games dataset
- **Raw size:** 122,610 games
- **After cleaning:** 40,194 games
- **Filter criteria:** 20+ reviews, no adult content

### Data Split

| Split | Size | Rows |
|---|---|---|
| Training | 60% | 24,116 |
| Validation | 20% | 8,039 |
| Test | 20% | 8,039 |

---

## Feature Engineering

Starting from raw Steam metadata, the following features were constructed:

- **Wilson Score** — calculated from positive/negative review counts, binarized at 0.75
- **`num_languages`** — count of supported languages
- **`estimated_owners_mid`** — midpoint of Steam's ownership range bands
- **One-hot encoding** — genres, categories, and common tags (tags appearing in 5,000+ games)
- **Tag count / genre count** — aggregate feature counts

**Total features: 507**

Dropped columns: developers, publishers, screenshots, movies, user score, score rank (strings with no analytical value or sparse/empty data).

---

## Exploratory Data Analysis

Key findings from EDA:

- **Wilson score distribution is left-skewed** — most games score between 0.6 and 0.9. Players leave positive reviews at a much higher rate: mean positive reviews = 2,694 vs. mean negative = 432 (6× ratio).
- **Top tags** — "Indie" and "Single Player" dominate, reflecting the large volume of small-studio games on the platform.
- **Highest correlations with well-received:** tag count, Linux support, Mac support. Tag count may be high because users only tag games they engage with. Mac/Linux support correlates with success possibly because developers only port to additional platforms when a game is already performing well.

---

## Statistical Analysis

**Question:** Is there an association between a game's primary Steam tag and whether it is well received?

**Test:** Chi-square test (categorical variables)

| Statistic | Value |
|---|---|
| Chi-square | 943.92 |
| Degrees of freedom | 19 |
| p-value | < 0.0001 |
| Cramér's V | 0.179 |

**Conclusion:** Statistically significant association exists between Steam tag and reception, but the effect size is weak-to-moderate. Tags alone are not a strong standalone predictor.

---

## Model Details

**Algorithm:** Logistic Regression (scikit-learn)

- `max_iter=1000` — set to ensure convergence across 507 features
- **Hyperparameter tuning:** best `C` selected using validation AUC; optimal `C=100` (val AUC = 0.7189)

### Top Coefficients

**Positive predictors (features associated with being well-received):**
- Recommendation, Replay Value — players who enjoy a game both recommend it and replay it

**Negative predictors (features associated with poor reception):**
- Generic, high-volume tags shared across both good and bad games — these add noise rather than signal

---

## Key Insight

For indie developers: focus on a niche genre and build something that stands out. Generic tags shared by thousands of games have little predictive power for reception.

**Limitations:**
- Several top features (estimated owners, recommendation count) are not controllable by a developer pre-launch
- No time-series component — trends in what players want shift over time
- Reception is influenced by many factors outside Steam metadata (marketing, streamers, timing, word of mouth)

---

## Future Improvements

- [ ] Time-series analysis — track what genres and tags trend positively over different release windows
- [ ] Remove post-launch features to build a true pre-release prediction tool
- [ ] Try tree-based models (XGBoost, Random Forest) to capture non-linear feature interactions
- [ ] Add review sentiment analysis from text if available

---

## Project Context

**Course:** Data Analysis / Machine Learning  
**Author:** Miguel Cortes  
**Institution:** University Of Lethbridge
