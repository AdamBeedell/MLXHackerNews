from sqlalchemy import create_engine, text

connection_string = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(connection_string)

query = text("""
    SELECT
      i.id AS id,
      i.by AS by,
      i.title AS title,
      i.score AS score
    FROM hacker_news.items i
    WHERE 
      i.score IS NOT NULL AND
      i.title IS NOT NULL AND
      i.type IN ('story', 'poll', 'pollopt')
    OFFSET :offset
    LIMIT :limit
""")

def fetch_hacker_news_info(limit=10000, offset=0, include_comments=True):
  connection = engine.connect()
  result = connection.execute(query, {"limit": limit, "offset": offset})
  connection.close()
  return result



