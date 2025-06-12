

import pandas as pd
from urllib.parse import urlparse

def extract_domain(url):
    if pd.isna(url) or url == '':
        return '<unk>'
    try:
        return urlparse(url).netloc.replace('www.', '') or '<unk>'
    except:
        return '<unk>'

def clean_chunk(chunk):
    stories = chunk[chunk['type'] == 'story']
    stories = stories[stories['title'].notna()]
    stories = stories[stories['title'].str.len() > 3]
    stories = stories[['title', 'score', 'by', 'url']]
    stories['domain'] = stories['url'].apply(extract_domain)
    return stories[['title', 'score', 'by', 'domain']]

chunks = pd.read_csv("data.csv", chunksize=100_000)
with open("cleantrainingdata.csv", "w") as out_file:
    header_written = False
    for chunk in chunks:
        cleaned = clean_chunk(chunk)
        cleaned.to_csv(out_file, index=False, header=not header_written, mode='a')
        header_written = True