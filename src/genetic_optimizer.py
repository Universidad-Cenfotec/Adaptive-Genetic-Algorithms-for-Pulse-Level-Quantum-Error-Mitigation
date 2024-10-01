import numpy as np
import random
from deap import base, creator, tools

class GeneticOptimizer:
    """
    Encapsulates the genetic algorithm optimization.
    """
    def __init__(
        self,
        evaluator,
        population_size=50,
        num_generations=10,
        mutation_probability=0.2,
        crossover_probability=0.5
    ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.toolbox = self._setup_toolbox()
        self.hall_of_fame = tools.HallOfFame(1)
        self.logbook = tools.Logbook()

    def _setup_toolbox(self):
        """
        Sets up the DEAP toolbox.
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self._cx_dict)
        toolbox.register("mutate", self._mut_dict)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluator.evaluate)
        return toolbox

    def _init_individual(self, icls):
        """
        Initializes an individual with random parameters.
        """
        return icls({
            "SNOT": {
                "num_tslots": np.random.randint(1, 10),
                "evo_time": np.random.uniform(0.1, 3)
            },
            "X": {
                "num_tslots": np.random.randint(1, 5),
                "evo_time": np.random.uniform(0.1, 1)
            },
            "CNOT": {
                "num_tslots": np.random.randint(1, 20),
                "evo_time": np.random.uniform(0.1, 10)
            }
        })

    def _cx_dict(self, ind1, ind2):
        """
        Crossover function for individuals.
        """
        for key in ind1.keys():
            if random.random() < 0.5:
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    def _mut_dict(self, ind):
        """
        Mutation function for individuals using a dispatcher.
        """
        def mutate_snot(ind):
            ind["SNOT"]["num_tslots"] = np.random.randint(1, 10)
            ind["SNOT"]["evo_time"] = np.random.uniform(0.1, 3)
        
        def mutate_x(ind):
            ind["X"]["num_tslots"] = np.random.randint(1, 5)
            ind["X"]["evo_time"] = np.random.uniform(0.1, 1)

        def mutate_cnot(ind):
            ind["CNOT"]["num_tslots"] = np.random.randint(1, 20)
            ind["CNOT"]["evo_time"] = np.random.uniform(0.1, 10)

        dispatcher = {
            "SNOT": mutate_snot,
            "X": mutate_x,
            "CNOT": mutate_cnot
        }

        key = random.choice(list(ind.keys()))
        dispatcher[key](ind) 
        return ind,


    def run(self):
        """
        Runs the genetic algorithm optimization. 
        """
        pop = self.toolbox.population(n=self.population_size)
        # Define statistics
        stats_fidelity = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fidelity.register("avg", np.mean)
        stats_fidelity.register("std", np.std)
        stats_fidelity.register("min", np.min)
        stats_fidelity.register("max", np.max)
        # Configuring the logbook
        self.logbook.header = ["gen", "nevals"] + stats_fidelity.fields
        for gen in range(self.num_generations):
            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_probability:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < self.mutation_probability:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Replacement
            pop[:] = offspring
            # Hall of Fame
            self.hall_of_fame.update(pop)
            # Record statistics
            record = stats_fidelity.compile(pop)
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(self.logbook.stream)
        return pop