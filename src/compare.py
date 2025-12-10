# CREATES TWO COMPARISON TABLES WITH EVAL METRICS AND BENCHMARKS
# metrics.py -> results.py saves a dict for each method's best checkpoint
# compare.py takes data from there and creates tables
import pandas as pd
from typing import List
import os
import json


def get_metrics(paths: List[str]):
    out = {}
    for sub in paths:
        model_name_dir = {}
        method_dir = {}
        dir_path = os.path.join("results", sub)
        if not os.path.isdir(dir_path):
            continue

        for f in os.listdir(dir_path):
            if not f.endswith(".json"):
                continue

            p = os.path.join(dir_path, f)
            with open(p, "r") as fp:
                # example of p: results/metrics/meta-llama_Llama-3.2-3B_lora_20251122_094620.json
                model_name = f.split("/")[-1].split("_")[1]
                method_name = f.split("/")[-1].split("_")[2]



        out[sub] = ""
    return out




def final_comparison_table():
    pass