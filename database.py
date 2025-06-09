import os
import psycopg2
from datetime import datetime

def connect():
  # Define your connection parameters
  conn_params = {
      "dbname": os.environ.get('POSTGRES_DB'),
      "user": os.environ.get('POSTGRES_USER'),
      "password": os.environ.get('POSTGRES_PASSWORD'),
      "host": os.environ.get('POSTGRES_HOST'), 
      "port": "5432"
  }
  # Establish the connection
  conn = psycopg2.connect(**conn_params)
  return conn

def save_history(prediction, true_value):
  conn = connect()
  insert_query = "INSERT INTO mnist_history (date, prediction, true_value) VALUES (%s, %s, %s);"
  # Insert data into the table
  data_to_insert = (datetime.now(), prediction, true_value)
  try:
    cur = conn.cursor()
    cur.execute(insert_query, data_to_insert)
    conn.commit()
  except Exception as error:
    print("Error while interacting with PostgreSQL", error)      
  finally:
    # Close the cursor and connection to free resources
    if cur:
      cur.close()
    if conn:
      conn.close()

# TODO never do this, use some kind of migrations manager
def init_table():
  conn = connect()
  # Create a cursor object
  cur = conn.cursor()
  try:
    # Create a table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS mnist_history (
        id SERIAL PRIMARY KEY, 
        date TIMESTAMP,
        prediction INT,
        true_value INT
    );
    """
    cur.execute(create_table_query)
    
    # Commit the transaction to save changes
    conn.commit()
    print("Table created!")
  except Exception as error:
    print("Error while interacting with PostgreSQL", error)
  finally:
    # Close the cursor and connection to free resources
    if cur:
      cur.close()
    if conn:
      conn.close()

# TODO paginate?
def get_history():
  conn = connect()
  cur = conn.cursor()
  try:
    select_query = "SELECT date, prediction, true_value FROM mnist_history ORDER BY date DESC;"
    cur.execute(select_query)
    rows = cur.fetchall()
    return rows
  except Exception as error:
    print("Error while interacting with PostgreSQL", error)      
  finally:
    if cur:
      cur.close()
    if conn:
      conn.close()
