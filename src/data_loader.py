import numpy as np
import pandas as pd
from collections import Counter
from datasets import load_dataset

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config


def load_pubmedqa():
    ds = load_dataset(config.PUBMEDQA_DATASET, config.PUBMEDQA_SUBSET)
    return ds["train"].to_pandas()


def get_dataset_stats(df):
    context_lengths = df["context"].apply(lambda x: len(x["contexts"]))

    label_dist = df["final_decision"].value_counts().to_dict()

    mesh_counts = Counter(
        mesh
        for row in df["context"]
        for mesh in row["meshes"]
    )

    return {
        "total_samples": len(df),
        "context_length_mean": float(np.mean(context_lengths)),
        "context_length_distribution": context_lengths.value_counts().sort_index().to_dict(),
        "label_distribution": label_dist,
        "top_meshes": mesh_counts.most_common(40),
        "unique_meshes": len(mesh_counts),
    }


def build_mesh_lookup(df):
    return {
        str(row["pubid"]): row["context"]["meshes"]
        for _, row in df.iterrows()
    }


def get_meshes_with_frequency(df, min_freq=5, max_freq=20):
    mesh_counts = Counter(
        mesh
        for row in df["context"]
        for mesh in row["meshes"]
    )
    return {
        word: count
        for word, count in sorted(mesh_counts.items(), key=lambda x: -x[1])
        if min_freq <= count <= max_freq
    }
