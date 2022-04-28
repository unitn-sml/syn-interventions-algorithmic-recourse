import pandas as pd
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt

import numpy as np

from argparse import ArgumentParser

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data", type=str, nargs="+", help="Path to the file with the experiment configuration")

    args = parser.parse_args()

    # Get the data and build a structure
    complete_data = {}
    for data in args.data:

        model = data.split("-")[1]

        df = pd.read_csv(data)

        for d in df["id"].unique():
            p = df[df.id==d]["program"].tolist()
            a = df[df.id==d]["arguments"].tolist()

            if d in complete_data:
                complete_data[d][model] = list(zip(p,a))
            else:
                complete_data[d] = {model: list(zip(p, a))}

    # Compute the distance
    distance = {}
    for d, v in complete_data.items():

        for a, _ in v.items():
            for b, _ in v.items():

                if a == b:
                    continue

                p_a = "-".join([f"{p}({arg})" for p,arg in v[a]])
                p_b = "-".join([f"{p}({arg})" for p,arg in v[b]])

                if a+"_"+b in distance:
                    distance[a + "_" + b].append(similar(p_a, p_b))
                elif b+"_"+a in distance:
                    continue
                else:
                    distance[a + "_" + b] = [similar(p_a, p_b)]


    print()
    for k in distance:
        print(k, np.mean(distance[k]))





