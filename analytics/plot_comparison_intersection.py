
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser

sns.set(style="white")
sns.set_palette(sns.color_palette("colorblind"))
sns.set_context("paper")

LEGEND_FONT_SIZE = 15
AXES_TITLE_SIZE = 17
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

import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'

method_rename = {
    "mcts": r'$M_{\hbox{FARE}}$',
    "program": r'$M_{\hbox{E-FARE}}$',
    "agent_only": r'$M_{agent}$',
    "cscf": r'$M_{cscf}$',
    "cscf_small": r'$M_{cscf}^{small}$',
}

hue_order = ["cscf", "cscf_small", "mcts", "program"]
hue_order = [method_rename.get(x) for x in hue_order]

if __name__ == "__main__":

    df = pd.read_csv("analytics/evaluation/results.csv")
    dfb = pd.read_csv("analytics/evaluation/results_intersection.csv")

    order = ["german", "adult", "syn", "syn\_long"]

    df.rename(columns={'correct': 'accuracy', 'mean_cost': 'cost',
                       "mean_rule": "\# of predicates", "mean_length": "\# of actions"}, inplace=True)

    dfb.rename(columns={'correct': 'accuracy', 'mean_cost': 'cost',
                       "mean_rule": "\# of predicates", "mean_length": "\# of actions"}, inplace=True)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8,5))

    # Skip agent_only
    df = df[df.method!="agent_only"]
    dfb = dfb[dfb.method!="agent_only"]

    # Rename dataset
    df["method"] = df["method"].apply(lambda x: method_rename.get(x))
    dfb["method"] = dfb["method"].apply(lambda x: method_rename.get(x))
    df['dataset'] = df['dataset'].apply(lambda x: "syn\_long" if x == "syn_long" else x)
    dfb['dataset'] = dfb['dataset'].apply(lambda x: "syn\_long" if x == "syn_long" else x)

    cols = zip(["Length (Successful only)", "Cost (Successful only)"], ["\# of actions", "cost"])
    for idx, (t, k) in enumerate(cols):
        brp = sns.barplot(x="dataset", y=k, hue="method", data=dfb, ax=axs[0][idx], order=order)
        brp.legend_.remove()
        axs[0][idx].set_title(r"\textbf{"+t+"}")

    cols = zip(["Length", "Cost"],["\# of actions", "cost"])
    for idx, (t,k) in enumerate(cols):
        brp = sns.barplot(x="dataset", hue="method", y=k, data=df, ax=axs[1][idx], order=order)
        brp.legend_.remove()
        axs[1][idx].set_title(r"\textbf{"+t+"}")


    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df.method.unique()), bbox_to_anchor=(0.5, 1.13))

    plt.tight_layout()
    plt.savefig("length_cost.png", dpi=600, bbox_inches='tight')
