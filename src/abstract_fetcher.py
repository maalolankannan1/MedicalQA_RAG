import os
import time
import requests
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def fetch_pubmed_abstract(pubid):
    params = {
        "id": pubid,
        "db": "pubmed",
        "rettype": "abstract",
        "retmode": "text",
    }
    r = requests.get(config.NCBI_EFETCH_URL, params=params)
    return r.text

def download_all_abstracts(df, output_dir=None, delay=None):
    output_dir = output_dir or config.DATA_RAW_DIR
    delay = delay or config.NCBI_REQUEST_DELAY
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        pubid = row["pubid"]
        filepath = os.path.join(output_dir, f"{pubid}.txt")

        if os.path.exists(filepath):
            continue

        abstract = fetch_pubmed_abstract(pubid)
        with open(filepath, "w") as f:
            f.write(abstract)
        print(f"Downloaded abstract for {pubid}")
        time.sleep(delay)

    print(f"Finished downloading abstracts to {output_dir}")

def fetch_abstracts_for_mesh(df, mesh, output_dir=None, delay=0.5):
    output_dir = output_dir or config.DATA_RAW_DIR
    os.makedirs(output_dir, exist_ok=True)

    pubids = df[df["context"].apply(lambda x: mesh in x["meshes"])]["pubid"]
    filepath = os.path.join(output_dir, f"{mesh}.txt")

    with open(filepath, "a") as f:
        for pubid in pubids:
            abstract = fetch_pubmed_abstract(pubid)
            f.write(f"Abstract for PubID: {pubid}\\n")
            f.write(abstract)
            f.write("\\nENDING THIS ABSTRACT\\n")
            time.sleep(delay)

    print(f"Saved {len(pubids)} abstracts for mesh '{mesh}' to {filepath}")
