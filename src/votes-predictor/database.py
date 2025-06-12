from sqlalchemy import create_engine, text

connection_string = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
engine = create_engine(connection_string)


def fetch_hacker_news_info(limit=10000, offset=0, include_comments=True):
  query = text("""
    SELECT
      i.id AS id,
      i.by AS by,
      i.title AS title,
      i.score AS score,
      i.time AS time
    FROM hacker_news.items i
    WHERE 
      i.score IS NOT NULL AND
      i.title IS NOT NULL AND
      i.type IN ('story', 'poll', 'pollopt') AND
      i.time < to_timestamp('2024-10-13 23:53:00', 'YYYY-MM-DD HH24:MI:SS') - INTERVAL '30 days'
    OFFSET :offset
    LIMIT :limit
  """)

  connection = engine.connect()
  result = connection.execute(query, {"limit": limit, "offset": offset})
  connection.close()
  return result

def fetch_hackernews_length():
  query = text("""
	SELECT
        COUNT(*) as count
    FROM hacker_news.items i
    WHERE 
      i.score IS NOT NULL AND
      i.title IS NOT NULL AND
      i.type IN ('story', 'poll', 'pollopt') AND
      i.time < to_timestamp('2024-10-13 23:53:00', 'YYYY-MM-DD HH24:MI:SS') - INTERVAL '30 days'
  """)

  connection = engine.connect()
  result = connection.execute(query)
  connection.close()
  return result[0].count


