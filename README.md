# Synthesizing explainable counterfactual policies for algorithmic recourse with program synthesis

This repository contains the code to reproduce the experiments for the paper "Synthesizing explainable counterfactual policies for algorithmic recourse with program synthesis" (https://arxiv.org/pdf/2201.07135). 


## 1 Set up the environment

We first need to install the library which contains the code for the agent and the various
procedures (training, evaluation). We use `conda` to manage the various packages.
```shell script
conda create --name syn_inter python=3.7
conda activate syn_inter
cd agent
pip install -r requirements.txt
python setup.py install
cd ../synthesize_interventions
pip install -r requirements.txt
```
We have now installed all the dependencies needed to run the project.

## 2 Run the experiments

### 2.2 Reproduce accuracy results

In order to reproduce the results of our paper, you can exploit directly the trained models which come in this repository.
You can find them in the directory `synthetize_interventions/models`. The script which generates the result csv files can be
called in the following way. **This will take some time, since we also have to evaluate the CSCF model.** If you just want to reproduce the
plots, see the instruction below. 
```shell script
conda activate syn_inter
cd synthesize_interventions
python analytics/evaluate_all.py
```
This will create a file `results.csv` which can be used to generate the Figure 5 of the paper. 
Then, if we want to recreate the same graph, we can use the following script:
```shell script
python analytics/plot.py results.csv
```
Alternatively, you can plot directly the results by using the pre-computed data:
```shell script
python analytics/plot.py analytics/evaluation/results.csv
```
You can inspect the `evaluate_all.py` script to check which are the single commands needed to
test each single model separately. The configuration file `analytics/config.yml` contains the path to the pre-trained models. 

### 2.3 Reproduce query results

The following command will re-generate the Figure 6 of the paper.

```shell script
conda activate syn_inter
cd synthesize_interventions
python analytics/plot_queries.py analytics/evaluation/data/*
```

### 2.4 Run a single experiment
If you want to regenerate also the models used for this experiments. You can follow the step below.
To run a single experiment, the command is the following: the argument of the flag `--config` can be one of the
experiment's configuration files that you can find in `synthetize_interventions/synthetizer/*`. For example, in order to train
the model for the german credit dataset:
```shell script
cd synthesize_interventions
mpirun -n 4 python3 -m generalized_alphanpi.core.run --config synthetizer/german/config_german.yml
```

### 2.5 Generate the programs
You can also regenerate the deterministic program used for these experiments by exploiting our pre-trained models.
You can find the command sequence below. **Bear in mind that this command will overwrite the pre-trained programs, which are located in `models/automa`.** 
```shell script
cd synthesize_interventions
python analytics/create_automa.py
```
### 2.6 Test effect of sampling on the programs performances
Alternatively, in order to test the effect of a diverse range of sampling techniques on the performance
of the deterministic program, you can run the following script (which will take a while):
```shell script
conda activate syn_inter
cd synthesize_interventions
mkdir program_size
python analytics/train_evaluate_program.py
```
This will save all the program models in the `program_size` directory for later use and it will also
generate a file called `program_results.csv` which can be used for plotting. In order to plot the results, please use the following
commands:
```shell script
cd synthesize_interventions
python analytics/plot_program_evaluations.py program_results.csv
```
