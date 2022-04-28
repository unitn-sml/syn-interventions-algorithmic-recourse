import numpy as np

from pymoo.core.survival import Survival
#from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
# Apparently this one is not available in Pymooo 5.0
#from pymoo.operators.crossover.biased_crossover import BiasedCrossover
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display import SingleObjectiveDisplay, MultiObjectiveDisplay
from pymoo.util.termination.default import (
    SingleObjectiveDefaultTermination,
    MultiObjectiveDefaultTermination,
)
#from pymoo.algorithms.soo.nonconvex.brkga import EliteBiasedSelection
from pymoo.core.selection import Selection
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.core.infill import InfillCriterion

import math

class Mating(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 **kwargs):

        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # how many parents need to be select for the mating - depending on number of offsprings remaining
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)

            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select, self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        _off = self.crossover.do(problem, pop, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)

        return _off

def split_by_feasibility(pop, eps=0.0, sort_infeasbible_by_cv=True):
    CV = pop.get("CV")

    b = (CV <= eps)

    feasible = np.where(b)[0]
    infeasible = np.where(~b)[0]

    if sort_infeasbible_by_cv:
        infeasible = infeasible[np.argsort(CV[infeasible, 0])]

    return feasible, infeasible

class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def do(self,
           problem,
           pop,
           *args,
           n_survive=None,
           return_indices=False,
           **kwargs):

        # make sure the population has at least one individual
        if len(pop) == 0:
            return pop

        if n_survive is None:
            n_survive = len(pop)

        n_survive_n = min(n_survive, len(pop))

        # if the split should be done beforehand
        if self.filter_infeasible and problem.n_constr > 0:

            # split feasible and infeasible solutions
            feas, infeas = split_by_feasibility(pop, eps=0.0, sort_infeasbible_by_cv=True)

            if len(feas) == 0:
                survivors = Population()
            else:
                survivors = self._do(problem, pop[feas], n_survive=min(len(feas), n_survive_n), **kwargs)

            # calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive_n - len(survivors)

            # if infeasible solutions needs to be added
            if n_remaining > 0:
                survivors = Population.merge(survivors, pop[infeas[:n_remaining]])

        else:
            survivors = self._do(problem, pop, *args, n_survive=n_survive_n, **kwargs)

        if return_indices:
            H = {}
            for k, ind in enumerate(pop):
                H[ind] = k
            return [H[survivor] for survivor in survivors]
        else:
            return survivors

    def _do(self, problem, pop, n_survive=None, out=None, **kwargs):
        F = pop.get("F")

        if F.shape[1] != 1:
            raise ValueError("FitnessSurvival can only used for single objective single!")

        return pop[np.argsort(F[:, 0])[:n_survive]]

class EliteBiasedSelection(Selection):

    def _do(self, pop, n_select, n_parents, **kwargs):
        _type = pop.get("type")
        elites = np.where(_type == "elite")[0]
        non_elites = np.where(_type == "non_elite")[0]

        # if through duplicate elimination no non-elites exist
        if len(non_elites) == 0:
            non_elites = elites

        # if there are no elites, randomly sample from them
        if len(elites) == 0:
            import random
            for e in pop:
                if random.random() <= 0.5:
                    e.set("type", "elite")
                else:
                    e.set("type", "non_elite")

            _type = pop.get("type")
            elites = np.where(_type == "elite")[0]
            non_elites = np.where(_type == "non_elite")[0]

        # do the mating selection - always one elite and one non-elites
        s_elite = elites[RandomSelection().do(elites, n_select, 1)[:, 0]]
        s_non_elite = non_elites[RandomSelection().do(non_elites, n_select, 1)[:, 0]]

        return np.column_stack([s_elite, s_non_elite])

class CEliteSurvival(Survival):
    def __init__(self, n_elites, eliminate_duplicates=None):
        super().__init__(False)
        self.n_elites = n_elites
        self.eliminate_duplicates = eliminate_duplicates

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        pop = DefaultDuplicateElimination(func=lambda p: p.get("F")).do(pop)

        if problem.n_obj == 1:
            pop = FitnessSurvival().do(problem, pop, len(pop))
            elites = pop[: self.n_elites]
            non_elites = pop[self.n_elites :]
        else:
            # Only use feasible solutions for NDS and getting the elites
            _feas = pop.get("feasible")[:, 0]
            if _feas.any():
                F = pop.get("F")[_feas]
                I = NonDominatedSorting(method="efficient_non_dominated_sort").do(
                    F, only_non_dominated_front=True
                )
                elites = pop[_feas][I]
                _I = np.arange(len(pop))
                assert len(_I[_feas][I]) == len(I)
                assert len(_I[_feas][I]) <= len(_feas)
                I = _I[_feas][I]
            else:
                I = NonDominatedSorting(method="efficient_non_dominated_sort").do(
                    pop.get("F"), only_non_dominated_front=True
                )
                elites = pop[I]

            non_elites = pop[[k for k in range(len(pop)) if k not in I]]

        assert len(elites) + len(non_elites) == len(pop), (len(elites), len(non_elites))
        elites.set("type", ["elite"] * len(elites))
        non_elites.set("type", ["non_elite"] * len(non_elites))

        return pop


class CSCF(GeneticAlgorithm):
    def __init__(
        self,
        n_elites=200,
        n_offsprings=700,
        n_mutants=100,
        bias=0.7,
        sampling=FloatRandomSampling(),
        survival=None,
        display=SingleObjectiveDisplay(),
        eliminate_duplicates=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_elites : int
            Number of elite individuals
        n_offsprings : int
            Number of offsprings to be generated through mating of an elite and a non-elite individual
        n_mutants : int
            Number of mutations to be introduced each generation
        bias : float
            Bias of an offspring inheriting the allele of its elite parent
        eliminate_duplicates : bool or class
            The duplicate elimination is more important if a decoding is used. The duplicate check has to be
            performed on the decoded variable and not on the real values. Therefore, we recommend passing
            a DuplicateElimination object.
            If eliminate_duplicates is simply set to `True`, then duplicates are filtered out whenever the
            objective values are equal.
        """

        super().__init__(
            pop_size=n_elites + n_offsprings + n_mutants,
            n_offsprings=n_offsprings,
            sampling=sampling,
            selection=EliteBiasedSelection(),
            crossover=BinomialCrossover(bias, prob=1.0),
            mutation=NoMutation(),
            survival=CEliteSurvival(n_elites),
            display=display,
            eliminate_duplicates=True,
            mating=Mating(EliteBiasedSelection(), BinomialCrossover(bias, prob=1.0), NoMutation()),
            **kwargs
        )

        self.n_elites = n_elites
        self.n_mutants = n_mutants
        self.bias = bias
        # This is overwritten later anyway, so don't mind
        self.default_termination = SingleObjectiveDefaultTermination()

    def _infill(self):
        # actually do the mating given the elite selection and biased crossover
        off = self.mating.do(
            self.problem, self.pop, n_offsprings=self.n_offsprings, algorithm=self
        )

        return off

    def _advance(self, infills=None, **kwargs):
        pop = self.pop
        elites = np.where(pop.get("type") == "elite")[0]

        # create the mutants randomly to fill the population with
        mutants = FloatRandomSampling().do(self.problem, self.n_mutants, algorithm=self)

        # evaluate all the new solutions
        to_evaluate = Population.merge(infills, mutants)
        self.evaluator.eval(self.problem, to_evaluate, algorithm=self)

        # finally merge everything together and sort by fitness
        pop = Population.merge(pop[elites], to_evaluate)

        # the do survival selection - set the elites for the next round
        self.pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)

