import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\jasva\Downloads\Scaledux\Task 1 Dataset\Startup_Scoring_Dataset.csv')

def min_max(col):
    return (col - col.min()) / (col.max() - col.min())

features = [
    'team_experience',
    'market_size_million_usd',
    'monthly_active_users',
    'monthly_burn_rate_inr',
    'funds_raised_inr',
    'valuation_inr'
]
df_norm = df.copy()
for f in features:
    df_norm[f] = min_max(df[f])
df_norm['monthly_burn_rate_inr'] = 1 - df_norm['monthly_burn_rate_inr']

weights = {
    'team_experience': 0.15,
    'market_size_million_usd': 0.20,
    'monthly_active_users': 0.25,
    'monthly_burn_rate_inr': 0.10,
    'funds_raised_inr': 0.10,
    'valuation_inr': 0.20
}
df_norm['score'] = sum(df_norm[f] * w for f, w in weights.items()) * 100

df_norm['rank'] = df_norm['score'].rank(ascending=False, method='min')
df_norm_sorted = df_norm.sort_values('score', ascending=False)
top10 = df_norm_sorted.head(10)
bottom10 = df_norm_sorted.tail(10)

print("Top 10 Startups:")
print(top10[['startup_id', 'score']])
print("\nBottom 10 Startups:")
print(bottom10[['startup_id', 'score']])

def explain(row):
    reasons = []
    if row['team_experience'] > 0.8: reasons.append("very experienced team")
    if row['market_size_million_usd'] > 0.8: reasons.append("large market size")
    if row['monthly_active_users'] > 0.8: reasons.append("strong user traction")
    if row['monthly_burn_rate_inr'] > 0.8: reasons.append("very low burn rate")
    if row['funds_raised_inr'] > 0.8: reasons.append("well funded")
    if row['valuation_inr'] > 0.8: reasons.append("high valuation")
    return ", ".join(reasons)

print("\nWhy Top Startup Scored High:")
print(explain(df_norm_sorted.iloc[0]))
print("\nWhy Bottom Startup Scored Low:")
print(explain(df_norm_sorted.iloc[-1]))

os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(12, 4))
plt.bar(range(len(df_norm_sorted)), df_norm_sorted['score'])
plt.title('Startup Scores (Sorted)')
plt.xlabel('Startup Rank')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('outputs/scores_bar_chart.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png')
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(df_norm_sorted['score'], bins=20, color='skyblue')
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('outputs/score_histogram.png')
plt.close()

X = df_norm[features]
y = df_norm['score']
model = LinearRegression()  # ML MODEL
model.fit(X, y)
ml_pred = model.predict(X)
rmse = mean_squared_error(y, ml_pred, squared=False)
r2 = r2_score(y, ml_pred)
print(f"\nML Model RMSE vs manual score: {rmse:.2f}")
print(f"ML Model R^2 vs manual score: {r2:.3f}")

importances = pd.Series(model.coef_, index=features)
print("\nFeature importances (ML coefficients):")
print(importances.sort_values(ascending=False))

plt.figure(figsize=(8,4))
importances.sort_values().plot(kind='barh', color='teal')
plt.title('Feature Importances (Linear Regression)')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.savefig('outputs.png')
plt.close()
