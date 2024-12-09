import csv
import multiprocessing
import random
import secrets
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from deap import base, creator, tools

EVALUATOR_MESSAGE = "The 'evaluator' does not have an 'evaluate' method."

class GeneticOptimizer:
    """
    Genetic algorithm optimization with feedback-based mutation, crossover adjustment,
    diversity control, elitism, early stopping, and improved parallelization.
    """

    def __init__(
        self,
        evaluator,
        population_size=300,
        num_generations=100,
        mutation_probability=0.2,
        crossover_probability=0.5,
        feedback_threshold=0.01,
        feedback_interval=10,
        early_stopping_rounds=20,
        diversity_threshold=0.5,
        diversity_action='mutate',  # Can be 'mutate' or 'replace'
        n_jobs=None,
    ):
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

        self.executor = ProcessPoolExecutor(max_workers=self.n_jobs)

        self.toolbox = self._setup_toolbox()
        self.hall_of_fame = tools.HallOfFame(1)
        self.logbook = tools.Logbook()
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
        rng = np.random.default_rng()
        return icls({
            "SNOT": {
                "num_tslots": rng.integers(1, 10),
                "evo_time": rng.uniform(0.1, 3),
            },
            "X": {
                "num_tslots": rng.integers(1, 5),
                "evo_time": rng.uniform(0.1, 1),
            },
            "CNOT": {
                "num_tslots": rng.integers(1, 20),
                "evo_time": rng.uniform(0.1, 10),
            },
        })

    def _cx_dict(self, ind1, ind2):
        sys_random = random.SystemRandom()
        for key in ind1:
            if sys_random.random() < self.crossover_probability:
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    def _mut_dict(self, ind):
        rng = np.random.default_rng()

        def mutate_snot(ind):
            ind["SNOT"]["num_tslots"] = rng.integers(1, 10)
            ind["SNOT"]["evo_time"] = rng.uniform(0.1, 3)

        def mutate_x(ind):
            ind["X"]["num_tslots"] = rng.integers(1, 5)
            ind["X"]["evo_time"] = rng.uniform(0.1, 1)

        def mutate_cnot(ind):
            ind["CNOT"]["num_tslots"] = rng.integers(1, 20)
            ind["CNOT"]["evo_time"] = rng.uniform(0.1, 10)

        dispatcher = {
            "SNOT": mutate_snot,
            "X": mutate_x,
            "CNOT": mutate_cnot,
        }

        key = secrets.choice(list(ind.keys()))
        dispatcher[key](ind)
        return ind,

    def calculate_diversity(self, population):
        distances = []
        individuals = list(population)
        N = len(individuals)
        for i in range(N - 1):
            for j in range(i + 1, N):
                xi = np.array([val for subdict in individuals[i].values() for val in subdict.values()])
                xj = np.array([val for subdict in individuals[j].values() for val in subdict.values()])
                distance = np.linalg.norm(xi - xj)
                distances.append(distance)
        if len(distances) == 0:
            return 0
        return np.mean(distances)

    def _adjust_probabilities(self, current_avg_fitness):
        if self.last_avg_fitness is not None:
            improvement = current_avg_fitness - self.last_avg_fitness
            if improvement < self.feedback_threshold:
                self.mutation_probability = min(1.0, self.mutation_probability + 0.05)
                self.crossover_probability = min(1.0, self.crossover_probability + 0.05)
                print(f"Improvement {improvement:.4f} < threshold. Increasing mutation and crossover probability.")
            else:
                self.mutation_probability = max(0.1, self.mutation_probability - 0.05)
                self.crossover_probability = max(0.3, self.crossover_probability - 0.05)
                print(f"Improvement {improvement:.4f} >= threshold. Decreasing mutation and crossover probability.")

    def run(self, csv_filename="genetic_algorithm_log.csv"):
        pop = self.toolbox.population(n=self.population_size)
        stats_fidelity = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fidelity.register("avg", np.mean)
        stats_fidelity.register("std", np.std)
        stats_fidelity.register("min", np.min)
        stats_fidelity.register("max", np.max)

        self.logbook.header = ["gen", "nevals", *stats_fidelity.fields, "diversity"]

        csv_path = Path(csv_filename)
        with csv_path.open(mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.logbook.header)

            for gen in range(self.num_generations):
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))

                for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
                    if secrets.randbelow(100) < self.crossover_probability * 100:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if secrets.randbelow(100) < self.mutation_probability * 100:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses, strict=False):
                    ind.fitness.values = fit

                pop[:] = offspring

                elite = tools.selBest(pop, 1)
                pop[:1] = elite

                self.hall_of_fame.update(pop)

                diversity = self.calculate_diversity(pop)

                record = stats_fidelity.compile(pop)
                record["diversity"] = diversity
                self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                print(self.logbook.stream)
                writer.writerow([gen, len(invalid_ind), *list(record.values())])

                if (gen + 1) % self.feedback_interval == 0:
                    current_avg_fitness = record["avg"]
                    self._adjust_probabilities(current_avg_fitness)
                    self.last_avg_fitness = current_avg_fitness

                if diversity < self.diversity_threshold:
                    print(f"Diversidad {diversity:.4f} < umbral {self.diversity_threshold}. Aplicando acción de diversidad: {self.diversity_action}.")
                    if self.diversity_action == 'mutate':
                        for ind in pop:
                            self.toolbox.mutate(ind)
                            del ind.fitness.values
                    elif self.diversity_action == 'replace':
                        num_replace = int(0.1 * self.population_size)
                        for _ in range(num_replace):
                            new_ind = self.toolbox.individual()
                            replace_idx = random.randint(0, self.population_size - 1)
                            pop[replace_idx] = new_ind
                    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses, strict=False):
                        ind.fitness.values = fit

                if self.last_avg_fitness is not None:
                    if record["avg"] > self.last_avg_fitness + self.feedback_threshold:
                        self.no_improvement = 0
                    else:
                        self.no_improvement += 1

                if self.no_improvement >= self.early_stopping_rounds:
                    print(f"No hubo mejora en las últimas {self.early_stopping_rounds} generaciones. Parando temprano.")
                    break

        self.executor.shutdown(wait=True)

        return pop, self.logbook