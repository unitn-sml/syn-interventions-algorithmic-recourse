import numpy as np
import decimal


class Decoder(object):
    """
    docstring
    """

    def __init__(self, problem):
        self.problem = problem
        self.invalid_genotype_value = 1.0

    def decode_without_repair(self, x):
        _x = x.copy()
        # two parts, first sequence, second values
        assert _x.ndim == 1

        split_point = len(x) // 2

        sequence_part = _x[:split_point]
        value_part = _x[split_point:]

        assert len(sequence_part) == len(value_part)
        sequence_phenotype, fixed_sequence_genotype = self.get_sequence_pheno(
            sequence_part
        )
        assert sequence_phenotype.dtype == np.float64, sequence_phenotype.dtype
        value_phenotype = self.get_values_pheno(value_part, sequence_phenotype)
        assert value_phenotype.dtype == np.float64, value_phenotype.dtype

        # now twice as long
        phenotype = np.concatenate([sequence_phenotype, value_phenotype])
        assert phenotype.dtype == np.float64, phenotype.dtype
        return phenotype, np.concatenate([fixed_sequence_genotype, x[split_point:]])

    def get_decoded_and_fix(self, x):
        phenotype, fixed_genotype = self.decode_without_repair(x)
        assert phenotype.dtype == np.float64, phenotype.dtype
        assert fixed_genotype.dtype == x.dtype
        assert fixed_genotype.shape == x.shape
        return phenotype, fixed_genotype

    def decode(self, x):
        phenotype, fixed_genotype = self.decode_without_repair(x)
        assert phenotype.dtype == np.float64, phenotype.dtype
        assert phenotype.shape == x.shape
        return phenotype

    def get_sequence_pheno(self, sequence_genotype):
        """
        Returns the sorted representation of the genotype

        Parameters
        ----------
        sequence_genotype : array
            Genotype of the sequence

        Returns
        -------
        array
            Indices of sorted genotype, i.e. phenotype
        """
        inactive = self.evaluate_is_inactive(sequence_genotype)
        phenotype_candidate = np.argsort(sequence_genotype)
        phenotype = np.full(len(phenotype_candidate), -1)
        phenotype[: sum(~inactive)] = phenotype_candidate[: sum(~inactive)]
        assert set(np.arange(len(sequence_genotype))[~inactive]).union([-1]) == set(
            phenotype
        ).union([-1]), phenotype

        genotype_candidate = sequence_genotype.copy()
        genotype_candidate = genotype_candidate[phenotype_candidate]
        return phenotype.astype(np.float64), genotype_candidate.astype(np.float64)

    def evaluate_is_inactive(self, sequence_genotype):
        """Simple heuristic that defines that an action is inactive
        if its value is above 0.5 in the genotype form

        Args:
            sequence_genotype ([type]): [description]

        Returns:
            [type]: [description]
        """
        return sequence_genotype > 0.5

    def get_values_pheno(self, values_genotype, prog_idx):
        value_phenotype = np.zeros(len(values_genotype), dtype=np.float64)

        for k, (action_dict_id, geno_val) in enumerate(zip(prog_idx, values_genotype)):

            if action_dict_id != -1:
                phenotype_value = self.get_interpolated_phenotype(geno_val, action_dict_id)
            else:
                phenotype_value = -1
            value_phenotype[k] = phenotype_value
        return value_phenotype

    def get_interpolated_phenotype(self, genotype_value, action_id):
        """
        Interpolates the genotype interval of [0,1] to the respective phenotype range

        Parameters
        ----------
        genotype_value : float
            Genotype representation of the current value
        action_idx : int
            Index of the respective action

        Returns
        -------
        int or float
            Genotype representation
        """
        assert 0.0 <= action_id <= self.problem.n_actions, action_id

        GENOTYPE_INTERVAL = [0.0, 1.0]

        #program_name = self.problem.env.get_program_from_index(int(action_id))
        program_name = self.problem.available_actions.get(int(action_id))[0]
        args_type = self.problem.env.programs_library.get(program_name)["args"]
        arguments = self.problem.env.arguments.get(args_type)

        xl = 0
        xu = len(arguments)-1

        phenotype_interval = [xl, xu]
        assert phenotype_interval[0] <= phenotype_interval[1], phenotype_interval
        # * If xl == xu, then the interpolated value will always be the same
        # * (independent of the input value)
        phenotype_value = np.interp(
            genotype_value, GENOTYPE_INTERVAL, phenotype_interval
        )

        #print(program_name, arguments, xl, xu, int(phenotype_value), action_id)

        return int(phenotype_value)

        #if action_id in self.problem.cat_actions_idx:
            # Phenotype value is only idx at this point
            # So we retrieve it from the provided mapping
        #    phenotype_value = self.problem.bounds_and_values[int(action_id)][
        #        int(phenotype_value)
        #    ]
        #    return float(phenotype_value)
        #else:
        #    return float(phenotype_value)
