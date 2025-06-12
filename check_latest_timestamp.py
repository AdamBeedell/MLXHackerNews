from sqlalchemy import text
from sqlalchemy import create_engine
import pandas as pd
from urllib.parse import urlparse

DATABASE_URL = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    latest_time = conn.execute(text("""
        SELECT MAX(p.time)
        FROM hacker_news.items_by_year_2024 AS p
        JOIN hacker_news.items AS i ON p.id = i.id
        WHERE i.score IS NOT NULL
    """)).scalar()

print("Latest timestamp in this snapshot:", latest_time)

"""Latest timestamp in this snapshot: 2024-10-14 00:00:50"""

df = pd.read_sql(
    """SELECT p.id, i.title, i.score::INTEGER AS score, p.time::TIMESTAMP AS time, i.url
       FROM hacker_news.items_by_year_2024 AS p
       JOIN hacker_news.items AS i USING (id)
       WHERE i.score IS NOT NULL
    """, engine
)
df['domain'] = df['url'].apply(lambda u: urlparse(str(u)).netloc)

domain_stats = df.groupby('domain').agg(
    domain_post_count=('domain', 'size'),
    domain_total_upvotes=('score', 'sum')
).reset_index()
domain_stats['domain_mean_upvotes'] = domain_stats['domain_total_upvotes'] / domain_stats['domain_post_count']

domain_stats.to_csv('domain_stats_features.csv', index=False)
print("Saved domain_stats_features.csv!")

