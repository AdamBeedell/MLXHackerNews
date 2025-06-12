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


# 3) Load full DataFrame (may be large) then sample a reproducible subset
full_df = pd.read_sql(
    """
    SELECT p.id, i.title, i.score::INTEGER AS score, p.time::TIMESTAMP AS time
      FROM hacker_news.items_by_year_2024 AS p
      JOIN hacker_news.items AS i ON p.id = i.id
      WHERE i.score IS NOT NULL
    """,
    engine
)
# Sample 10000 rows reproducibly for EDA and baseline
df = full_df.sample(n=100000, random_state=42).reset_index(drop=True)

# Drop rows where score is missing
df = df.dropna(subset=['score']).reset_index(drop=True)

# If you plan to use title-based features, you can fill missing titles with empty string:
df['title'] = df['title'].fillna('')

# Convert types if needed
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df = df.dropna(subset=['score']).reset_index(drop=True)
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

#  Check if enough rows for baseline split
if len(df) <= 200:
    raise ValueError(f"Not enough non-null rows for baseline split: only {len(df)} rows")

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
df['title_len'] = df['title'].str.len()
X_feat       = df[['title_len']].iloc[:-200]
y_feat_train = df['log_score'].iloc[:-200]
X_feat_val   = df[['title_len']].iloc[-200:]
y_feat_val   = df['log_score'].iloc[-200:]

ridge = Ridge(alpha=1.0)
ridge.fit(X_feat, y_feat_train)
ridge_preds = ridge.predict(X_feat_val)
mae_ridge = mean_absolute_error(y_feat_val, ridge_preds)

print(f"Baseline MAE (log‑scale, ridge(title_len)): {mae_ridge:.4f}")

# === Feature EDA on log-scale ===
# Assumes df already contains columns: 'log_score', 'title', 'time', and that you have joined 'items' to get 'descendants' and 'url' if available.

# 1) Numeric features EDA
# Example numeric features: title length (already in df), comment count (descendants), time-of-day, day-of-week
# If 'descendants' column exists in df (comment count):
if 'descendants' in df.columns:
    df['comment_count'] = pd.to_numeric(df['descendants'], errors='coerce').fillna(0).astype(int)
else:
    # If not joined earlier, try to fetch from items table
    try:
        # Attempt to load descendants via a join
        tmp = pd.read_sql(
            """
            SELECT p.id, i.descendants
              FROM hacker_news.items_by_year_2024 AS p
              JOIN hacker_news.items AS i ON p.id = i.id
              WHERE i.descendants IS NOT NULL
              LIMIT 100000
            """,
            engine
        )
        df = df.merge(tmp, on='id', how='left')
        df['comment_count'] = df['descendants'].fillna(0).astype(int)
    except Exception:
        df['comment_count'] = 0

# Title length already exists: df['title_len']
# Time-based features:
df['hour'] = df['time'].dt.hour
df['weekday'] = df['time'].dt.weekday  # Monday=0

# Scatter / hexbin plots: numeric vs. log_score
import matplotlib.pyplot as plt
import seaborn as sns

# For large data, sample a subset for plotting
sample_df = df.sample(n=min(len(df), 2000), random_state=0)

# title length vs log_score
plt.figure(figsize=(6,4))
sns.scatterplot(x=sample_df['title_len'], y=sample_df['log_score'], alpha=0.3)
plt.xlabel("Title Length")
plt.ylabel("log1p(Upvotes)")
plt.title("Title Length vs log1p(Upvotes)")
plt.show()

# comment count vs log_score
plt.figure(figsize=(6,4))
sns.scatterplot(x=sample_df['comment_count'], y=sample_df['log_score'], alpha=0.3)
plt.xlabel("Comment Count")
plt.ylabel("log1p(Upvotes)")
plt.title("Comment Count vs log1p(Upvotes)")
plt.show()

# hour of day vs log_score: boxplot or pointplot
plt.figure(figsize=(8,4))
sns.boxplot(x=df['hour'], y=df['log_score'])
plt.xlabel("Hour of Day")
plt.ylabel("log1p(Upvotes)")
plt.title("Distribution of log1p(Upvotes) by Hour of Day")
plt.show()

# day of week vs log_score
plt.figure(figsize=(8,4))
sns.boxplot(x=df['weekday'], y=df['log_score'])
plt.xlabel("Weekday (0=Mon)")
plt.ylabel("log1p(Upvotes)")
plt.title("Distribution of log1p(Upvotes) by Weekday")
plt.show()

# 2) Categorical feature EDA: domain boxplots
# If 'url' column exists, extract domain
if 'url' in df.columns:
    import urllib.parse
    def extract_domain(u):
        try:
            netloc = urllib.parse.urlparse(u).netloc
            # remove port if present
            return netloc.split(':')[0].lower()
        except:
            return ''
    df['domain'] = df['url'].fillna('').apply(extract_domain)
else:
    # Attempt to load url via join if not present
    try:
        tmp2 = pd.read_sql(
            """
            SELECT p.id, i.url
              FROM hacker_news.items_by_year_2024 AS p
              JOIN hacker_news.items AS i ON p.id = i.id
              WHERE i.url IS NOT NULL
              LIMIT 100000
            """,
            engine
        )
        df = df.merge(tmp2, on='id', how='left')
        import urllib.parse
        df['domain'] = df['url'].fillna('').apply(lambda u: urllib.parse.urlparse(u).netloc.split(':')[0].lower() if u else '')
    except Exception:
        df['domain'] = ''

# For domain boxplots, restrict to top N domains by frequency
top_domains = df['domain'].value_counts().nlargest(10).index.tolist()
df_top = df[df['domain'].isin(top_domains)].copy()

plt.figure(figsize=(10,5))
sns.boxplot(x='domain', y='log_score', data=df_top)
plt.xlabel("Domain (top 10)")
plt.ylabel("log1p(Upvotes)")
plt.title("log1p(Upvotes) by Domain (top 10)")
plt.xticks(rotation=45)
plt.show()