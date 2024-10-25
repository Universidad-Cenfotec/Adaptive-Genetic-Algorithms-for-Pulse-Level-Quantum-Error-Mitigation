import csv
import multiprocessing
import random
import secrets
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

CROSSOVER_THRESHOLD = 0.5

class GeneticOptimizer:
    """
    Genetic algorithm optimization with feedback-based mutation, crossover adjustment,
    diversity control, elitism, early stopping, and improved parallelization.
    """

    def __init__(
        self,
        evaluator,
        population_size=100,          # Ajustado según recomendaciones
        num_generations=150,          # Ajustado según recomendaciones
        mutation_probability=0.2,
        crossover_probability=0.5,
        feedback_threshold=0.01,      # Threshold para mejora de fitness
        feedback_interval=10,         # Intervalo para chequeo de feedback
        early_stopping_rounds=20,     # Generaciones sin mejora para detención temprana
        n_jobs=None,                  # Número de procesos paralelos
    ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.feedback_threshold = feedback_threshold
        self.feedback_interval = feedback_interval
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs if n_jobs else max(4, multiprocessing.cpu_count())

        # Crear el pool de procesos una sola vez
        self.pool = multiprocessing.Pool(self.n_jobs)

        self.toolbox = self._setup_toolbox()
        self.hall_of_fame = tools.HallOfFame(1)
        self.logbook = tools.Logbook()
        self.last_avg_fitness = None
        self.no_improvement = 0

        # Para graficar el progreso
        self.generations = []
        self.avg_fitnesses = []
        self.diversities = []
        self.mutation_probs = []
        self.crossover_probs = []

    def _setup_toolbox(self):
        """
        Configura el toolbox de DEAP.
        """
        # Evitar recrear clases si ya existen
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self._cx_dict)
        toolbox.register("mutate", self._mut_dict)
        toolbox.register("select", tools.selTournament, tournsize=3)  # Mantener selección por torneo

        # Registrar el método de evaluación
        if hasattr(self.evaluator, 'evaluate'):
            toolbox.register("evaluate", self.evaluator.evaluate)
            print("Método 'evaluate' registrado correctamente en el toolbox.")
        else:
            raise AttributeError("El 'evaluator' no tiene un método 'evaluate'.")

        # Registrar la función map para paralelización con chunksize
        toolbox.register("map", lambda func, data: self.pool.map(func, data, chunksize=10))

        return toolbox

    def _init_individual(self, icls):
        """
        Inicializa un individuo con parámetros aleatorios usando np.random.Generator.
        """
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
        """
        Función de crossover para individuos.
        """
        sys_random = random.SystemRandom()
        for key in ind1:
            if sys_random.random() < CROSSOVER_THRESHOLD:
                ind1[key], ind2[key] = ind2[key], ind1[key]
        return ind1, ind2

    def _mut_dict(self, ind):
        """
        Función de mutación para individuos usando un dispatcher.
        """
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

    def _adjust_probabilities(self, current_avg_fitness):
        """
        Ajusta las probabilidades de mutación y crossover basándose en el feedback.
        """
        if self.last_avg_fitness is not None:
            improvement = current_avg_fitness - self.last_avg_fitness
            if improvement < self.feedback_threshold:
                # Aumentar probabilidades si la mejora es pequeña
                self.mutation_probability = min(1.0, self.mutation_probability + 0.05)
                self.crossover_probability = min(1.0, self.crossover_probability + 0.05)
                print(f"Mejora {improvement:.4f} < umbral. Aumentando probabilidad de mutación y crossover.")
            else:
                # Reducir probabilidades si la mejora es buena
                self.mutation_probability = max(0.1, self.mutation_probability - 0.05)
                self.crossover_probability = max(0.3, self.crossover_probability - 0.05)
                print(f"Mejora {improvement:.4f} >= umbral. Reduciendo probabilidad de mutación y crossover.")

    def _measure_diversity(self, population):
        """
        Mide la diversidad genotípica basada en individuos únicos en la población.
        """
        def individual_to_tuple(individual):
            """
            Convierte un individuo (diccionario con estructuras anidadas) en una tupla plana y hashable.
            """
            return (
                individual["SNOT"]["num_tslots"],
                individual["SNOT"]["evo_time"],
                individual["X"]["num_tslots"],
                individual["X"]["evo_time"],
                individual["CNOT"]["num_tslots"],
                individual["CNOT"]["evo_time"],
            )

        unique_individuals = len(set(individual_to_tuple(ind) for ind in population))
        return unique_individuals / len(population)

    def _plot_progress(self):
        """
        Grafica el progreso del algoritmo genético.
        """
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.generations, self.avg_fitnesses, label="Fitness Promedio")
        plt.xlabel("Generaciones")
        plt.ylabel("Fitness Promedio")
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(self.generations, self.diversities, label="Diversidad")
        plt.xlabel("Generaciones")
        plt.ylabel("Diversidad")
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(self.generations, self.mutation_probs, label="Probabilidad de Mutación")
        plt.xlabel("Generaciones")
        plt.ylabel("Probabilidad de Mutación")
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(self.generations, self.crossover_probs, label="Probabilidad de Crossover")
        plt.xlabel("Generaciones")
        plt.ylabel("Probabilidad de Crossover")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    def run(self, csv_filename="genetic_algorithm_log.csv"):
        """
        Ejecuta la optimización del algoritmo genético y guarda el log en un archivo CSV.
        Utiliza ajuste dinámico basado en feedback, control de diversidad, elitismo y paralelización mejorada.
        """
        pop = self.toolbox.population(n=self.population_size)
        stats_fidelity = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fidelity.register("avg", np.mean)
        stats_fidelity.register("std", np.std)
        stats_fidelity.register("min", np.min)
        stats_fidelity.register("max", np.max)

        # Configuración del logbook
        self.logbook.header = ["gen", "nevals", *stats_fidelity.fields]

        # Crear o sobrescribir el archivo CSV con el encabezado del log
        with open(csv_filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.logbook.header)

            for gen in range(self.num_generations):
                # Selección
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))

                # Crossover y mutación
                for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
                    if secrets.randbelow(100) < self.crossover_probability * 100:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if secrets.randbelow(100) < self.mutation_probability * 100:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluación
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses, strict=False):
                    ind.fitness.values = fit

                # Reemplazo
                pop[:] = offspring

                # Elitismo: Añadir el mejor individuo a la siguiente generación
                elite = tools.selBest(pop, 1)
                pop[:1] = elite

                # Hall of Fame
                self.hall_of_fame.update(pop)

                # Registro de estadísticas
                record = stats_fidelity.compile(pop)
                diversity = self._measure_diversity(pop)
                self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                print(self.logbook.stream)

                # Añadir los datos de la generación al archivo CSV
                writer.writerow([gen, len(invalid_ind), *list(record.values())])

                # Ajustar probabilidades cada 'feedback_interval' generaciones
                if (gen + 1) % self.feedback_interval == 0:
                    current_avg_fitness = record['avg']
                    self._adjust_probabilities(current_avg_fitness)
                    self.last_avg_fitness = current_avg_fitness

                # Implementar detención temprana
                if self.last_avg_fitness is not None:
                    if record['avg'] > self.last_avg_fitness + self.feedback_threshold:
                        self.no_improvement = 0
                    else:
                        self.no_improvement += 1

                if self.no_improvement >= self.early_stopping_rounds:
                    print(f"No hay mejora en las últimas {self.early_stopping_rounds} generaciones. Deteniendo anticipadamente.")
                    break

                # Guardar el progreso para graficar
                self.generations.append(gen)
                self.avg_fitnesses.append(record['avg'])
                self.diversities.append(diversity)
                self.mutation_probs.append(self.mutation_probability)
                self.crossover_probs.append(self.crossover_probability)

        # Graficar el progreso
        self._plot_progress()

        # Cerrar el pool después de que el algoritmo termine
        self.pool.close()
        self.pool.join()

        return pop
