import csv
import multiprocessing
import secrets
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from deap import base, creator, tools
from scipy.linalg import inv, pinv
from scipy.spatial.distance import pdist

from src.gate_config import DEFAULT_SETTING_ARGS

# Constants to replace magic values
WORST_FIDELITY = 0.0
MIN_POPULATION_SIZE = 2
EPSILON = 1e-10  # Regularization term for covariance matrix
REPLACE_RATIO = 0.1  # Ratio of population to replace during diversity action

EVALUATOR_MESSAGE = "The 'evaluator' does not have an 'evaluate' method."

class GeneticOptimizer:
    """
    Genetic algorithm optimization with feedback-based mutation, crossover adjustment,
    diversity control using Mahalanobis distance, elitism, early stopping, and improved parallelization.
    """

    def __init__(
        self,
        evaluator,
        use_default,
        population_size=300,
        num_generations=100,
        mutation_probability=0.2,
        crossover_probability=0.5,
        feedback_threshold=0.01,
        feedback_interval=10,
        early_stopping_rounds=20,
        diversity_threshold=1.5,
        diversity_action="mutate",  # Can be 'mutate' or 'replace'
        n_jobs=None,
    ):
        """
        Initializes the GeneticOptimizer with various hyperparameters.

        Args:
            evaluator: An object that must implement an 'evaluate' method
                       (e.g., evaluator.evaluate(individual)).
            population_size (int): Size of the GA population.
            num_generations (int): Number of generations to run.
            mutation_probability (float): Probability of mutation per individual.
            crossover_probability (float): Probability of applying crossover.
            feedback_threshold (float): Improvement threshold for adjusting mutation/crossover probabilities.
            feedback_interval (int): Number of generations between feedback adjustments.
            early_stopping_rounds (int): Early stopping if no improvement after these many generations.
            diversity_threshold (float): If population diversity falls below this, apply a diversity action.
            diversity_action (str): 'mutate' or 'replace' strategy for diversity control.
            n_jobs (int, optional): Number of parallel jobs (default is max(4, cpu_count())).
            use_default (bool): Whether to initialize individuals using default settings.

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
        self.n_jobs = n_jobs if n_jobs else max(4, multiprocessing.cpu_count())
        self.use_default = use_default  # Almacenar el uso de configuraciones por defecto

        self.executor = ProcessPoolExecutor(max_workers=self.n_jobs)

        self.toolbox = self._setup_toolbox()
        self.hall_of_fame = tools.HallOfFame(1)
        self.logbook = tools.Logbook()
        self.last_avg_fitness = None
        self.no_improvement = 0

    def _setup_toolbox(self):
        """
        Sets up the DEAP toolbox: creating types (FitnessMax, Individual),
        and registering operators (mate, mutate, select).
        """
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

        # Check that the evaluator has an 'evaluate' method
        if hasattr(self.evaluator, "evaluate"):
            toolbox.register("evaluate", self.evaluator.evaluate)
            print("Method 'evaluate' successfully registered in the toolbox.")
        else:
            raise AttributeError(EVALUATOR_MESSAGE)

        def parallel_map(func, data):
            return list(self.executor.map(func, data, chunksize=10))

        toolbox.register("map", parallel_map)
        return toolbox

    def _init_individual(self, icls):
        """
        Creates an individual (dictionary) with initial parameters based on
        either DEFAULT_SETTING_ARGS or SPECIFIC_SETTING_ARGS,
        or random values if use_default=False and use_specific_config=False.
        """
        individual = {}

        if self.use_default:
            # Use DEFAULT_SETTING_ARGS
            for gate, params in DEFAULT_SETTING_ARGS.items():
                individual[gate] = params.copy()

        else:
            # Fallback: random initialization
            rng = np.random.default_rng()
            for gate in DEFAULT_SETTING_ARGS:
                individual[gate] = {
                    "num_tslots": rng.integers(1, 10),
                    "evo_time": rng.uniform(0.1, 3),
                }

        return icls(individual)

    def _cx_dict(self, ind1, ind2):
        """
        Dictionary-based crossover: with probability 'crossover_probability',
        swap the sub-dictionaries for each key.
        """
        for key in ind1:
            if secrets.randbelow(100) < int(self.crossover_probability * 100):
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    def _mut_dict(self, ind):
        """
        Dictionary-based mutation: randomly mutates one of the gates
        ('SNOT', 'X', or 'CNOT') by reassigning 'num_tslots' and 'evo_time'.
        """
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

        dispatcher = {
            "SNOT": mutate_snot,
            "X": mutate_x,
            "CNOT": mutate_cnot,
        }

        key = secrets.choice(list(ind.keys()))
        dispatcher[key](ind)
        return ind,

    def calculate_diversity(self, population):
        """
        Calculates population diversity using the average Mahalanobis distance
        among all pairs of individuals in parameter space.

        Args:
            population (list): List of individuals in the population.

        Returns:
            float: Average Mahalanobis distance.

        """
        if len(population) < MIN_POPULATION_SIZE:
            return WORST_FIDELITY  # No diversity if population has less than 2 individuals

        # Flatten individuals into a 2D array
        data = []
        for ind in population:
            vector = []
            for gate in sorted(ind.keys()):  # Sort keys to ensure consistent ordering
                vector.extend([ind[gate]["num_tslots"], ind[gate]["evo_time"]])
            data.append(vector)
        data = np.array(data)

        # Compute the covariance matrix
        covariance_matrix = np.cov(data, rowvar=False)

        # Regularization to ensure invertibility
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * EPSILON

        # Handle singular covariance matrix by using pseudo-inverse
        try:
            inv_covariance_matrix = inv(covariance_matrix)
        except np.linalg.LinAlgError:
            inv_covariance_matrix = pinv(covariance_matrix)

        # Compute pairwise Mahalanobis distances
        pairwise_distances = pdist(data, metric="mahalanobis", VI=inv_covariance_matrix)

        # Average Mahalanobis distance
        return np.mean(pairwise_distances)

    def _adjust_probabilities(self, current_avg_fitness):
        """
        Adjusts the mutation and crossover probabilities based on improvement
        relative to 'feedback_threshold'.
        If improvement < threshold, both are increased slightly;
        else, both are decreased slightly (within specified bounds).
        """
        if self.last_avg_fitness is not None:
            improvement = current_avg_fitness - self.last_avg_fitness
            if improvement < self.feedback_threshold:
                self.mutation_probability = min(1.0, self.mutation_probability + 0.05)
                self.crossover_probability = min(1.0, self.crossover_probability + 0.05)
                print(f"Improvement {improvement:.4f} < threshold. "
                      f"Increasing mutation and crossover probability.")
            else:
                self.mutation_probability = max(0.1, self.mutation_probability - 0.05)
                self.crossover_probability = max(0.3, self.crossover_probability - 0.05)
                print(f"Improvement {improvement:.4f} >= threshold. "
                      f"Decreasing mutation and crossover probability.")

    def run(self, csv_filename="genetic_algorithm_log.csv"):
        """
        Main loop of the genetic algorithm:
          - Create population
          - Evaluate fitness of invalid individuals
          - Perform selection, crossover, mutation
          - Maintain elitism
          - Apply diversity actions if needed
          - Adjust mutation/crossover feedback
          - Use early stopping after consecutive no-improvement rounds
        """
            # Create initial population
        pop = self.toolbox.population(n=self.population_size)

        # Define statistics for fitness
        stats_fidelity = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fidelity.register("avg", np.mean)
        stats_fidelity.register("std", np.std)
        stats_fidelity.register("min", np.min)
        stats_fidelity.register("max", np.max)

        # Prepare logbook
        self.logbook.header = ["gen", "nevals", *stats_fidelity.fields, "diversity"]

        csv_path = Path(csv_filename)
        with csv_path.open(mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.logbook.header)

            for gen in range(self.num_generations):
                # Selection and clone
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))

                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
                    if secrets.randbelow(100) < int(self.crossover_probability * 100):
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutation
                for mutant in offspring:
                    if secrets.randbelow(100) < int(self.mutation_probability * 100):
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                if invalid_ind:
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses, strict=False):
                        ind.fitness.values = fit

                # Replace population
                pop[:] = offspring

                # Elitism: preserve top 1 individual
                elite = tools.selBest(pop, 1)
                pop[:1] = elite

                # Update Hall of Fame
                self.hall_of_fame.update(pop)

                # Calculate diversity using Mahalanobis distance
                diversity = self.calculate_diversity(pop)

                # Gather stats
                record = stats_fidelity.compile(pop)
                record["diversity"] = diversity
                self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                print(self.logbook.stream)
                writer.writerow([gen, len(invalid_ind), *list(record.values())])

                # Feedback-based probability adjustment
                if (gen + 1) % self.feedback_interval == 0:
                    current_avg_fitness = record["avg"]
                    self._adjust_probabilities(current_avg_fitness)
                    self.last_avg_fitness = current_avg_fitness

                # Diversity control
                if diversity < self.diversity_threshold:
                    print(f"Diversity {diversity:.4f} < threshold {self.diversity_threshold}. "
                          f"Applying diversity action: {self.diversity_action}.")
                    if self.diversity_action == "mutate":
                        self._apply_mutation_action(pop)
                    elif self.diversity_action == "replace":
                        self._apply_replace_action(pop)
                    # Re-evaluate fitness of affected individuals
                    self._evaluate_population(pop)

                # Early stopping check
                if self.last_avg_fitness is not None:
                    if record["avg"] > self.last_avg_fitness + self.feedback_threshold:
                        self.no_improvement = 0
                    else:
                        self.no_improvement += 1

                if self.no_improvement >= self.early_stopping_rounds:
                    print(f"No improvement in the last {self.early_stopping_rounds} generations. Stopping early.")
                    break

        # Shutdown parallel executor
        self.executor.shutdown(wait=True)

        return pop, self.logbook

    def _apply_mutation_action(self, population):
        """
        Applies mutation to the entire population to increase diversity.
        """
        for ind in population:
            self.toolbox.mutate(ind)
            del ind.fitness.values
        print("Applied mutation to entire population for diversity.")

    def _apply_replace_action(self, population):
        """
        Replaces a portion of the population with new individuals to increase diversity.
        """
        num_replace = int(REPLACE_RATIO * self.population_size)
        for _ in range(num_replace):
            new_ind = self.toolbox.individual()
            replace_idx = secrets.randbelow(self.population_size)
            population[replace_idx] = new_ind
        print(f"Replaced {num_replace} individuals in the population for diversity.")

    def _evaluate_population(self, population):
        """
        Evaluates the fitness of all individuals in the population that have invalid fitness.
        """
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses, strict=False):
                ind.fitness.values = fit
        print("Re-evaluated fitness for affected individuals after diversity action.")
