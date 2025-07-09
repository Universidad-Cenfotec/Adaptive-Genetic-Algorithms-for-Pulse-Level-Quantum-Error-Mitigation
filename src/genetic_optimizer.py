import secrets

import numpy as np
from deap import base, creator, tools
from scipy.linalg import inv, pinv
from scipy.spatial.distance import pdist
from scoop import futures

from src.gate_config import DEFAULT_SETTING_ARGS

WORST_FIDELITY = 0.0
MIN_POPULATION_SIZE = 2
EPSILON = 1e-10
REPLACE_RATIO = 0.1  # Ratio of population to replace during diversity action
EVALUATOR_MESSAGE = "The 'evaluator' does not have an 'evaluate' method."

class GeneticOptimizer:
    """
    Genetic algorithm optimization with feedback-based mutation, crossover adjustment,
    diversity control using Mahalanobis distance, elitism, early stopping,
    and improved parallelization (now via SCOOP).
    """

    def __init__(
        self,
        evaluator,
        use_default=True,  # noqa: FBT002
        population_size=300,
        num_generations=100,
        mutation_probability=0.2,
        crossover_probability=0.5,
        feedback_threshold=0.01,
        feedback_interval=10,
        early_stopping_rounds=150,
        diversity_threshold=1.8,
        diversity_action="mutate",  # or 'replace'
    ):
        """
        Initializes the genetic optimizer with given hyperparameters.

        Args:
            evaluator: Object with an 'evaluate' method returning a fitness tuple.
            use_default (bool): Whether to initialize individuals using DEFAULT_SETTING_ARGS.
            population_size (int): GA population size.
            num_generations (int): Number of GA generations to run.
            mutation_probability (float): Probability of mutation per individual.
            crossover_probability (float): Probability to apply crossover to a pair of individuals.
            feedback_threshold (float): Improvement threshold for adjusting mutation/crossover probabilities.
            feedback_interval (int): # generations between each feedback check.
            early_stopping_rounds (int): If no improvement after these many checks, stop early.
            diversity_threshold (float): If population diversity (Mahalanobis) < threshold, apply a diversity action.
            diversity_action (str): 'mutate' or 'replace' to handle low diversity.
            n_jobs (int or None): # of parallel jobs (informational only for SCOOP).

        """
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.feedback_threshold = feedback_threshold
        self.feedback_interval = feedback_interval
        self.early_stopping_rounds = early_stopping_rounds
        self.diversity_threshold = diversity_threshold
        self.diversity_action = diversity_action
        self.use_default = use_default

        # DEAP setup
        self.toolbox = self._setup_toolbox()
        self.hall_of_fame = tools.HallOfFame(1)
        self.logbook = tools.Logbook()

        # Tracking improvement
        self.last_avg_fitness = None
        self.no_improvement = 0

    def _setup_toolbox(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self._cx_dict)
        toolbox.register("mutate", self._mut_dict)
        toolbox.register("select", tools.selTournament, tournsize=3)

        if not hasattr(self.evaluator, "evaluate"):
            raise AttributeError(EVALUATOR_MESSAGE)

        toolbox.register("evaluate", self.evaluator.evaluate)

        # --- SCOOP for parallelism ---
        toolbox.register("map", futures.map)

        print("Method 'evaluate' successfully registered in the toolbox.")
        return toolbox


    def _init_individual(self, icls):
        individual = {}
        if self.use_default:
            # Use DEFAULT_SETTING_ARGS
            for gate, params in DEFAULT_SETTING_ARGS.items():
                individual[gate] = params.copy()
        else:
            # Random initialization
            rng = np.random.default_rng()
            for gate in DEFAULT_SETTING_ARGS:
                individual[gate] = {
                    "num_tslots": rng.integers(1, 10),
                    "evo_time": rng.uniform(0.1, 3),
                }
        return icls(individual)

    def _cx_dict(self, ind1, ind2):
        # Dictionary-based crossover
        for key in ind1:
            if secrets.randbelow(100) < int(self.crossover_probability * 100):
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    def _mut_dict(self, ind):
        # Dictionary-based mutation
        rng = np.random.default_rng()

        def mutate_snot(individual):
            individual["SNOT"]["num_tslots"] = rng.integers(1, 10)
            individual["SNOT"]["evo_time"] = rng.uniform(0.1, 3)

        def mutate_x(individual):
            individual["X"]["num_tslots"] = rng.integers(1, 5)
            individual["X"]["evo_time"] = rng.uniform(0.1, 1)

        def mutate_cnot(individual):
            individual["CNOT"]["num_tslots"] = rng.integers(1, 20)
            individual["CNOT"]["evo_time"] = rng.uniform(0.1, 10)

        def mutate_swap(individual):
            individual["SWAP"]["num_tslots"] = rng.integers(1, 15)
            individual["SWAP"]["evo_time"] = rng.uniform(0.1, 5)

        def mutate_cphase(individual):
            individual["CPHASE"]["num_tslots"] = rng.integers(1, 15)
            individual["CPHASE"]["evo_time"] = rng.uniform(0.1, 5)

        def mutate_rz(individual):
            individual["RZ"]["num_tslots"] = rng.integers(1, 10)
            individual["RZ"]["evo_time"] = rng.uniform(0.1, 3)

        def mutate_globalphase(individual):
            individual["GLOBALPHASE"]["num_tslots"] = rng.integers(1, 10)
            individual["GLOBALPHASE"]["evo_time"] = rng.uniform(0.1, 3)

        dispatcher = {
            "SNOT": mutate_snot,
            "X": mutate_x,
            "CNOT": mutate_cnot,
            "SWAP": mutate_swap,
            "CPHASE": mutate_cphase,
            "RZ": mutate_rz,
            "GLOBALPHASE": mutate_globalphase,
        }

        key = secrets.choice(list(ind.keys()))
        dispatcher[key](ind)
        return (ind,)

    def calculate_diversity(self, population):
        if len(population) < MIN_POPULATION_SIZE:
            return WORST_FIDELITY

        # Flatten individuals to 2D array
        data = []
        for ind in population:
            vector = []
            for gate in sorted(ind.keys()):
                vector.extend([ind[gate]["num_tslots"], ind[gate]["evo_time"]])
            data.append(vector)
        data = np.array(data)

        # Covariance matrix
        covariance_matrix = np.cov(data, rowvar=False)
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * EPSILON

        try:
            inv_covariance_matrix = inv(covariance_matrix)
        except np.linalg.LinAlgError:
            inv_covariance_matrix = pinv(covariance_matrix)

        pairwise_distances = pdist(data, metric="mahalanobis", VI=inv_covariance_matrix)
        return np.mean(pairwise_distances)

    def _adjust_probabilities(self, current_avg_fitness):
        if self.last_avg_fitness is not None:
            improvement = current_avg_fitness - self.last_avg_fitness
            if improvement < self.feedback_threshold:
                self.mutation_probability = min(1.0, self.mutation_probability + 0.05)
                self.crossover_probability = min(1.0, self.crossover_probability + 0.05)
                print(f"Improvement {improvement:.4f} < threshold => Increase mutation/crossover.")
            else:
                self.mutation_probability = max(0.1, self.mutation_probability - 0.05)
                self.crossover_probability = max(0.3, self.crossover_probability - 0.05)
                print(f"Improvement {improvement:.4f} >= threshold => Decrease mutation/crossover.")

    def run(self, csv_logger=None, csv_filename="genetic_algorithm_log.csv"):
        """
        Main GA loop.
        If csv_logger is provided, logbook entries will be written to CSV.
        """
        # Initial population
        pop = self.toolbox.population(n=self.population_size)

        # Fitness stats
        stats_fidelity = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fidelity.register("avg", np.mean)
        stats_fidelity.register("std", np.std)
        stats_fidelity.register("min", np.min)
        stats_fidelity.register("max", np.max)

        self.logbook.header = ["gen", "nevals", *stats_fidelity.fields, "diversity"]

        # Logbook header
        if csv_logger is not None:
            csv_logger.write_logbook_header(self.logbook, csv_filename)

        for gen in range(self.num_generations):
            # Selection, clone
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if secrets.randbelow(100) < int(self.crossover_probability * 100):
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if secrets.randbelow(100) < int(self.mutation_probability * 100):
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals (parallel via SCOOP)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            # Replace population
            pop[:] = offspring

            # Elitism
            elite = tools.selBest(pop, 1)
            pop[:1] = elite

            # Hall of fame
            self.hall_of_fame.update(pop)

            # Diversity
            diversity = self.calculate_diversity(pop)

            # Record stats
            record = stats_fidelity.compile(pop)
            record["diversity"] = diversity
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(self.logbook.stream)

            # Write logbook entry
            if csv_logger is not None:
                self.logbook[-1].update(record)  # ensure data is stored
                csv_logger.append_logbook_entry(self.logbook, csv_filename)

            # Feedback-based probability adjustment
            if (gen + 1) % self.feedback_interval == 0:
                current_avg_fitness = record["avg"]
                self._adjust_probabilities(current_avg_fitness)
                self.last_avg_fitness = current_avg_fitness

            # Diversity control
            if diversity < self.diversity_threshold:
                print(f"Diversity {diversity:.4f} < threshold {self.diversity_threshold}.")
                if self.diversity_action == "mutate":
                    self._apply_mutation_action(pop)
                elif self.diversity_action == "replace":
                    self._apply_replace_action(pop)
                self._evaluate_population(pop)

            # Early stopping
            if self.last_avg_fitness is not None:
                if record["avg"] > self.last_avg_fitness + self.feedback_threshold:
                    self.no_improvement = 0
                else:
                    self.no_improvement += 1
            if self.no_improvement >= self.early_stopping_rounds:
                print(f"No improvement in the last {self.early_stopping_rounds} generations. Stopping early.")
                break

        # With SCOOP, no need to shut down our own pool
        return pop, self.logbook

    def _apply_mutation_action(self, population):
        for ind in population:
            self.toolbox.mutate(ind)
            del ind.fitness.values
        print("Applied mutation to entire population for diversity.")

    def _apply_replace_action(self, population):
        num_replace = int(REPLACE_RATIO * self.population_size)
        for _ in range(num_replace):
            new_ind = self.toolbox.individual()
            replace_idx = secrets.randbelow(self.population_size)
            population[replace_idx] = new_ind
        print(f"Replaced {num_replace} individuals for diversity.")

    def _evaluate_population(self, population):
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        print("Re-evaluated fitness for affected individuals after diversity action.")
