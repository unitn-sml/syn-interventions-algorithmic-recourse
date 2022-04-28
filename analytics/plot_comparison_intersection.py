
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser

sns.set(style="white")
sns.set_palette(sns.color_palette("colorblind"))
sns.set_context("paper", font_scale=1.3)

import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'

method_rename = {
    "mcts": r'$M_{mcts}$',
    "program": r'$M_{prog}$',
    "agent_only": r'$M_{agent}$',
    "cscf": r'$M_{cscf}$',
    "cscf_small": r'$M_{cscf}^{small}$',
}

hue_order = ["cscf", "cscf_small", "mcts", "program"]
hue_order = [method_rename.get(x) for x in hue_order]

if __name__ == "__main__":

    df = pd.read_csv("analytics/evaluation/results.csv")
    dfb = pd.read_csv("analytics/evaluation/results_intersection.csv")

    order = ["german", "adult", "syn", "syn_long"]

    df.rename(columns={'correct': 'accuracy', 'mean_cost': 'cost',
                       "mean_rule": "# of predicates", "mean_length": "# of actions"}, inplace=True)

    dfb.rename(columns={'correct': 'accuracy', 'mean_cost': 'cost',
                       "mean_rule": "# of predicates", "mean_length": "# of actions"}, inplace=True)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8,5))

    # Skip agent_only
    df = df[df.method!="agent_only"]
    dfb = dfb[dfb.method!="agent_only"]

    # Rename dataset
    df["method"] = df["method"].apply(lambda x: method_rename.get(x))
    dfb["method"] = dfb["method"].apply(lambda x: method_rename.get(x))

    cols = zip(["Length (Successful only)", "Cost (Successful only)"], ["# of actions", "cost"])
    for idx, (t, k) in enumerate(cols):
        brp = sns.barplot(x="dataset", y=k, hue="method", data=dfb, ax=axs[0][idx], order=order)
        brp.legend_.remove()
        axs[0][idx].set_title(t)

    cols = zip(["Length", "Cost"],["# of actions", "cost"])
    for idx, (t,k) in enumerate(cols):
        brp = sns.barplot(x="dataset", hue="method", y=k, data=df, ax=axs[1][idx], order=order)
        brp.legend_.remove()
        axs[1][idx].set_title(t)


    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df.method.unique()), bbox_to_anchor=(0.5, 1.1), fontsize="medium")

    plt.tight_layout()
    plt.savefig("length_cost.png", dpi=600, bbox_inches='tight')
