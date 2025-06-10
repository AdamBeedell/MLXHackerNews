# Adam.py

#### pull data from postgress

## postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki


from dotenv import load_dotenv
load_dotenv("env.txt")
import os
import bz2
import csv
import requests
import psycopg2

#db_name = os.environ.get('POSTGRES_DB')
#user = os.environ.get('POSTGRES_USER')
#password = os.environ.get('POSTGRES_PASSWORD')
#host = os.environ.get('POSTGRES_HOST')



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


## queries tables in postgres
def pull_data(connection):
    cursor = connection.cursor()
    cursor.execute("""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'items%';
                   """)
    rows = cursor.fetchall()
    cursor.close()
    return rows


## queries a small set of data from the items table
def pull_data(connection):
    cursor = connection.cursor()
    cursor.execute("""
                    SELECT *
                    FROM hacker_news.items
                    LIMIT 100;
                   """)
    rows = cursor.fetchall()
    cursor.close()
    return rows

## queries a medium set of data from the items table and outputs it locally as a bz2 compressed csv file
def pull_data(connection):
    cursor = connection.cursor()
    cursor.execute("""
                    SELECT *
                   FROM hacker_news.items
                   LIMIT 10000;
                   """)
    with bz2.open('data.csv.bz2', 'wt') as f:
        writer = csv.writer(f)
        writer.writerow([desc[0] for desc in cursor.description])
    
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
        
        writer.writerow(rows)


    cursor.close()


# pulls entire dataset 
def pull_full_data(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM hacker_news.items;")

    with bz2.open('data.csv.bz2', 'wt') as f:
        writer = csv.writer(f)
        writer.writerow([desc[0] for desc in cursor.description])

        total_rows = 0
        chunk_size = 1000

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            writer.writerows(rows)
            total_rows += len(rows)
            print(f"Wrote {total_rows} rows...")

    cursor.close()
    print(f"✅ Done. Total rows written: {total_rows}")



connection = connect()
#tables = pull_data(connection)
data = pull_full_data(connection)




def pull_wikipedia_data():
    downloadurl = "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8"
    localfile = "wikipedia_data.txt.bz2"

    with requests.get(downloadurl, stream=True) as response:
        with bz2.open(localfile, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Data downloaded and saved to {localfile}")    


pull_wikipedia_data()



