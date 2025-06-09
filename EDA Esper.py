import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# 1) Create a SQLAlchemy engine
engine = create_engine("postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")

# 2) Inspect available tables
inspector = inspect(engine)
print(inspector.get_table_names())

# List all schemas and their tables
schemas = inspector.get_schema_names()
print("Schemas:", schemas)
for schema in schemas:
    tables = inspector.get_table_names(schema=schema)
    print(f"Tables in schema '{schema}': {tables}")


# 3) Load the 2024 data partition into a DataFrame
df = pd.read_sql(
    """
    SELECT id,
           title,
           score::INTEGER AS score,
           time::TIMESTAMP AS time
      FROM hacker_news.items_by_year_2024
      lIMIT 1000
    """,
    engine
)

# Convert types if needed
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['time']  = pd.to_datetime(df['time'], errors='coerce')

print("\nFirst 5 rows:")
print(df.head())

print("\nDataFrame info:")
df.info()  # info() already prints to stdout

print("\nDescriptive statistics (all columns):")
print(df.describe(include='all'))

# 5) (Optional) Visualizations
sns.histplot(df['score'], bins=50, kde=True)  # Replace 'score' with your numeric column of interest
plt.title("Distribution of Upvotes (score)")

# Optionally, count rows in the 2024 partition
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM hacker_news.items_by_year_2024")).scalar()
print("Total rows in items_by_year_2024:", count)

plt.show()

#Now that you’ve seen how heavily right-skewed the upvotes are 
#(a huge spike at 0–10 votes and a long, thin tail out to 500+), 
# your very next move is to decide how you’ll tame that skew before throwing it at a regression algorithm. Here’s a concise sequence:
# 1. Transform (or reframe) your target
#	•	Log‐transform: Replace y = raw_score with y′ = log1p(raw_score). That compresses the tail and makes errors on high‐vote posts commensurate with errors on low‐vote posts. Your histogram of log1p(score) ought to look much more Gaussian.
#	•	Count models: Alternatively, consider a Poisson or negative‐binomial regression, which natively models count data without transformation.

#2. Pick the right evaluation metric
#	•	On a skewed target, MSE will obsess over the rare 500-vote posts. Better choices are:
#	•	MAE on log‐scale: mean(|log1p(y_pred) – log1p(y_true)|).
#	•	Poisson deviance or mean absolute percentage error (MAPE) if you want relative errors.

#3. Do feature‐target EDA under your transform
#	•	Plot scatter of log1p(score) vs. each numeric feature (e.g. post length, number of comments) to see linear vs. nonlinear relationships.
#	•	Plot boxplots of log1p(score) by categorical features (e.g. domain) to discover strong signal groups.

#4. Baseline model under your transform
#	•	Fit a quick linear regression or random forest on log1p(score) with your initial feature set to get a baseline metric.
#	•	This tells you whether you need more advanced modeling (e.g. gradient boosting, neural nets).

#5. Handle the tail if needed
#	•	If you still see huge outliers (the log‐transform doesn’t fully normalize extremes), you can:
#	1.	Two-stage modeling: first classify “low vs. viral” (e.g. above 50 votes), then train a separate regressor for the “viral” subset.
#	2.	Robust losses: use Huber loss or quantile regression to downweight extreme residuals.

 # 1) Log‐transform the target
df['log_score'] = np.log1p(df['score'])

# Drop rows where log_score is NaN (i.e. original score was missing)
df = df.dropna(subset=['log_score']).reset_index(drop=True)

# 2) Visualize the new distribution
plt.figure()
sns.histplot(df['log_score'], bins=50, kde=True)
plt.title("Distribution of log1p(upvotes)")
plt.show()

# 3) Mean‑predictor baseline on log‑scale
y = df['log_score']
# split last 200 for validation
y_train = y.iloc[:-200]
y_val   = y.iloc[-200:]
mean_pred = y_train.mean()
baseline_preds = [mean_pred] * len(y_val)
mae_mean = mean_absolute_error(y_val, baseline_preds)
print(f"Baseline MAE (log‑scale, mean predictor): {mae_mean:.4f}")

# 4) Feature‑based Ridge regression baseline
# Example feature: title length
df['title_len'] = df['title'].fillna('').str.len()
X_feat       = df[['title_len']].iloc[:-200]
y_feat_train = df['log_score'].iloc[:-200]
X_feat_val   = df[['title_len']].iloc[-200:]
y_feat_val   = df['log_score'].iloc[-200:]

ridge = Ridge(alpha=1.0)
ridge.fit(X_feat, y_feat_train)
ridge_preds = ridge.predict(X_feat_val)
mae_ridge = mean_absolute_error(y_feat_val, ridge_preds)
print(f"Baseline MAE (log‑scale, ridge(title_len)): {mae_ridge:.4f}")