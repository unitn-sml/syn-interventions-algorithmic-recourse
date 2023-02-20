from argparse import ArgumentParser

import random
import numpy as np
import pandas as pd

seed = 2021
random.seed(seed)
np.random.seed(seed)

N_EPISODES = 10

import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
sns.set_palette(sns.color_palette("colorblind"))

LEGEND_FONT_SIZE = 13
AXES_TITLE_SIZE = 15
AXES_LABEL_SIZE = 15
TICKS_SIZE = 15

sns.set(style="white")
sns.set_palette(sns.color_palette("colorblind"))
sns.set_context("talk", rc={
    "font.size":LEGEND_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.titlesize":AXES_TITLE_SIZE,
    "axes.labelsize":AXES_LABEL_SIZE,
    "xtick.labelsize": TICKS_SIZE,
    "ytick.labelsize": TICKS_SIZE
})


import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = "Computer Modern Roman"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", nargs="+", type=str, help="Path to the correct traces", default="./analytics/evaluation/data/*")

    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1)

    order = {
        "cscf": 1,
        "cscf_small": 1.5,
        "mcts (train)": 2,
        "program (train)": 3,
        "mcts (predict)": 4,
        "program (predict)": 5
    }

    rename = {
        "cscf": "$M_{cscf}$",
        "cscf_small": "$M_{cscf}^{small}$",
        "mcts (train)": r"$M_{\hbox{FARE}}$ (train)",
        "mcts (predict)": r"$M_{\hbox{FARE}}$ (predict)",
        "program (train)": r"$M_{\hbox{E-FARE}}$ (train)",
        "program (predict)": r"$M_{\hbox{E-FARE}}$ (predict)"
    }

    dataorder = ["german", "adult", "syn", "syn\_long"]

    df = [
        ["cscf", "syn", 1000000],
        ["cscf", "syn_long", 1000000],
        ["cscf", "german", 340000],
        ["cscf", "adult", 1000000],
        ["cscf_small", "syn", 250000],
        ["cscf_small", "syn_long", 250000],
        ["cscf_small", "german", 85000],
        ["cscf_small", "adult", 250000],
        ["mcts (predict)", "syn", 2963],
        ["mcts (predict)", "syn_long", 2869],
        ["mcts (predict)", "german", 783.02],
        ["mcts (predict)", "adult", 3205],
        ["program (train)", "syn", 250*29.63], # Each mean cost is multiplied by the sampling
        ["program (train)", "syn_long", 250*28.69],
        ["program (train)", "german", 250*23.76],
        ["program (train)", "adult", 250*32.05],
        ["program (predict)", "syn", 0],
        ["program (predict)", "syn_long", 0],
        ["program (predict)", "german", 0],
        ["program (predict)", "adult", 0]
    ]
    for d in args.path:

        tmp = pd.read_csv(d)
        name = d.split("-")[1]

        values = np.array(tmp["Value"].tolist())*N_EPISODES

        total = 0
        for id, k in enumerate(values):
            total += k*4 # 4 is the total number of core used.

        df.append(["mcts (train)", name, total])

    df.sort(key=lambda x: order.get(x[0]))

    df = pd.DataFrame(df, columns=["method", "dataset", "\# of queries"])

    df["method"] = df["method"].apply(lambda x: rename.get(x))

    df['dataset'] = df['dataset'].apply(lambda x: "syn\_long" if x == "syn_long" else x)

    current_palette = sns.color_palette()
    # Fix color order to make it consistent
    new_color_order = [current_palette[2], current_palette[3], current_palette[0], current_palette[1], current_palette[4], current_palette[7]]

    brp = sns.barplot(x="dataset", y="\# of queries", hue="method",
                      order=dataorder, data=df, ax=ax, palette=new_color_order)

    brp.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.40))
    brp.set(yscale="log")
    plt.tight_layout()
    plt.savefig("blackbox_calls.png", bbox_inches='tight', dpi=600)

