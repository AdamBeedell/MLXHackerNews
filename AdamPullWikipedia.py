import bz2
import csv
import requests

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

