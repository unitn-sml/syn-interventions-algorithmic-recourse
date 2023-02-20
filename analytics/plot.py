
import pandas as pd
import seaborn as sns

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

from argparse import ArgumentParser

method_rename = {
    "mcts": r'$M_{\text{FARE}}$',
    "program": r'$M_{\text{E-FARE}}$',
    "agent_only": r'$M_{agent}$',
    "cscf": r'$M_{cscf}$',
    "cscf_small": r'$M_{cscf}^{small}$',
}

order_df = {
        "cscf": 4,
        "cscf_small": 5,
        "mcts": 1,
        "program": 2,
    }

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the result file traces", default="./analytics/evaluation/results.csv")

    args = parser.parse_args()

    df = pd.read_csv(args.path)

    order = ["german", "adult", "syn", "syn\_long"]

    length = True
    cols = zip(["Validity", "Length", "Cost"],["validity", "\# of actions", "cost"])
    if not length:
        cols = zip(["Accuracy", "Cost"], ["validity", "cost"])

    df.rename(columns={'correct': 'validity', 'mean_cost': 'cost', "mean_length": "\# of actions"}, inplace=True)

    df['dataset'] = df['dataset'].apply(lambda x: "syn\_long" if x == "syn_long" else x)

    # Sort values
    df=df.loc[df['method'].map(order_df).sort_values(ascending=True).index]

    fig, axs = plt.subplots(ncols=3 if length else 2, figsize=(16,5))

    # Skip agent_only
    df = df[df.method!="agent_only"]

    # Rename dataset
    df["method"] = df["method"].apply(lambda x: method_rename.get(x))

    print(df)

    for idx, (t,k) in enumerate(cols):
        brp = sns.barplot(x="dataset", y=k, hue="method", data=df, ax=axs[idx], order=order)
        brp.legend_.remove()
        axs[idx].set_title(r"\textbf"+"{"+t+"}")

        #box = brp.get_position()
        #brp.set_position([box.x0, box.y0, box.width, box.height*0.85])

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df.method.unique()), bbox_to_anchor=(0.5, 1.2))

    #axs[2 if length else 1].legend()
    #sns.move_legend(brp, "center right", bbox_to_anchor=(1.25, .45), title='Model', bbox_to_anchor=(0.5,0.5))

    plt.tight_layout()
    #plt.show()
    plt.savefig("accuracy_cost.png", bbox_inches='tight', dpi=600)
    #plt.savefig("output.svg", bbox_inches='tight', format="svg")
