
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser

sns.set(style="white")
sns.set_palette(sns.color_palette("colorblind"))
sns.set_context("talk")

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

    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the result file traces", default="./analytics/evaluation/program_results.csv")

    args = parser.parse_args()

    df = pd.read_csv(args.path)

    order = ["german", "adult", "syn", "syn_long"]

    length = True
    cols = zip(["Accuracy", "Rule Length", "Length", "Cost"],["accuracy", "# of predicates", "# of actions", "cost"])
    if not length:
        cols = zip(["Accuracy", "Rule Length"], ["correct", "mean_rule"])

    df.rename(columns={'correct': 'accuracy', 'mean_cost': 'cost',
                       "mean_rule": "# of predicates", "mean_length": "# of actions"}, inplace=True)

    fig, axs = plt.subplots(ncols=2 if length else 2, nrows=2, figsize=(16,8))

    # Skip agent_only
    df = df[df.method!="agent_only"]

    # Rename dataset
    df["method"] = df["method"].apply(lambda x: method_rename.get(x))

    print(df)
    col = 0
    for idx, (t,k) in enumerate(cols):
        brp = sns.barplot(x="dataset", y=k, hue="sampling", data=df, ax=axs[col][idx%2], order=order)
        brp.legend_.remove()
        axs[col][idx%2].set_title(t)
        col += idx%2


    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df.sampling.unique()), bbox_to_anchor=(0.5, 1.05), fontsize="medium")

    plt.tight_layout()
    plt.savefig("accuracy_program_sampling.png", dpi=600, bbox_inches='tight')
