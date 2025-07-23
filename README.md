# Scaledux – Startup Evaluation Engine

**Scaledux** is a lightweight engine that scores startups on a scale from 0 to 100 (like a credit score) to assess their business health based on key operational and financial metrics.

---

## 1. Dataset Overview

The dataset contains details of **100 startups** with the following features:

* **Team Experience** – Skill level and background of the founding team.
* **Market Size** – Estimated total addressable market.
* **Monthly Active Users** – Number of active platform users.
* **Burn Rate** – Monthly expenditure.
* **Funds Raised** – Total external capital received.
* **Company Valuation** – Estimated market worth of the company.

---

## 2. Data Preprocessing

* Applied **Min-Max normalization** to scale all numeric features between 0 and 1.

### Rationale:

* Ensures fairness among features with varying scales (e.g., valuation vs. users).
* Prevents larger numerical features from dominating smaller ones in model training and scoring.

### Special Handling:

* **Positive Indicators** (higher is better):

  * Team Experience, Market Size, Users, Funds Raised, Valuation → kept as-is after normalization.
* **Negative Indicator** (lower is better):

  * Burn Rate → inverted using `1 - normalized_value` after scaling.

---

## 3. Scoring Logic

Each startup is assigned a score out of 100 using a **weighted average** of its normalized features.

| Feature              | Weight (%) | Rationale                                     |
| -------------------- | ---------- | --------------------------------------------- |
| Team Experience      | 15%        | Strong teams are more likely to execute well  |
| Market Size          | 15%        | Larger markets allow more growth opportunity  |
| Monthly Active Users | 20%        | Indicates product traction and demand         |
| Burn Rate (inverted) | 10%        | Lower burn implies better financial health    |
| Funds Raised         | 20%        | Shows investor confidence and runway          |
| Company Valuation    | 20%        | Reflects market perception and growth outlook |

---

## 4. Startup Ranking

* Startups are ranked based on their final scores.
* Identified:

  * **Top 10 Startups**: Strongest performers across all dimensions.
  * **Bottom 10 Startups**: Likely to be at risk or underperforming.

---

## 5. Score Interpretation

* **High-Scoring Example (\~95+)**:
  Startup with strong team, high user engagement, good funding, and efficient spending.

* **Low-Scoring Example (\~30)**:
  Startup with limited users, high burn, and little funding support.

---

## 6. Machine Learning Integration

* Trained a **regression model** to predict health scores of new or unseen startups using the same features.
* Used libraries like **scikit-learn** for implementation.

---

## 7. Visualization and Insights

Added visual components to support interpretation:

* **Bar Chart**: Displays scores of all startups.
* **Correlation Heatmap**: Highlights relationships between features.
* **Score Distribution Histogram**: Shows how startup scores are spread across the dataset.

---

## Tools & Libraries Used

* `Pandas`, `NumPy` – for data handling
* `scikit-learn` – for preprocessing and regression modeling
* `Matplotlib`, `Seaborn` – for visualizations
