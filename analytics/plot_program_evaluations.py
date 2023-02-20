
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser

sns.set(style="white")
sns.set_palette(sns.color_palette("colorblind"))
sns.set_context("talk")

#import matplotlib.pyplot as plt
#plt.rcParams['mathtext.fontset'] = 'cm'

LEGEND_FONT_SIZE = 25
AXES_TITLE_SIZE = 25
AXES_LABEL_SIZE = 25
TICKS_SIZE = 20

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

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the result file traces", default="./analytics/evaluation/program_results.csv")

    args = parser.parse_args()

    df = pd.read_csv(args.path)

    order = ["german", "adult", "syn", "syn\_long"]

    length = True
    cols = zip(["Validity", "Rule Length", "Length", "Cost"],["validity", "\# of predicates", "\# of actions", "cost"])
    if not length:
        cols = zip(["Validity", "Rule Length"], ["correct", "mean_rule"])

    df.rename(columns={'correct': 'validity', 'mean_cost': 'cost',
                       "mean_rule": "\# of predicates", "mean_length": "\# of actions"}, inplace=True)

    fig, axs = plt.subplots(ncols=2 if length else 2, nrows=2, figsize=(16,8))

    # Skip agent_only
    df = df[df.method!="agent_only"]

    # Rename dataset
    df["method"] = df["method"].apply(lambda x: method_rename.get(x))

    df['dataset'] = df['dataset'].apply(lambda x: "syn\_long" if x == "syn_long" else x)

    print(df)
    col = 0
    for idx, (t,k) in enumerate(cols):
        brp = sns.barplot(x="dataset", y=k, hue="sampling", data=df, ax=axs[col][idx%2], order=order)
        brp.legend_.remove()
        axs[col][idx%2].set_title(r"\textbf{"+t+"}")
        col += idx%2


    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df.sampling.unique()), bbox_to_anchor=(0.5, 1.1), fontsize="medium")

    plt.tight_layout()
    plt.savefig("program_sampling_result.png", dpi=600, bbox_inches='tight')
