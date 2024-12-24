import unittest

from qutip import Options

from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from src.evaluator import Evaluator
from src.genetic_optimizer import GeneticOptimizer
from src.noise_model import NoiseModel


def constant_evaluate(individual):  # noqa: ARG001
    return (0.8,)

def counting_evaluate(individual):
    counting_evaluate.calls += 1
    return counting_evaluate.original_evaluate(individual)

class TestGeneticOptimizer(unittest.TestCase):

    def setUp(self):
        num_qubits = 3
        self.circuit = DeutschJozsaCircuit(num_qubits)
        self.noise_model = NoiseModel(
            num_qubits=num_qubits,
            t1=50.0,
            t2=30.0,
            bit_flip_prob=0.02,
            phase_flip_prob=0.02,
        )
        self.solver_options = Options(nsteps=100000, store_states=True)
        self.evaluator = Evaluator(self.circuit, self.noise_model, self.solver_options)
        self.optimizer = GeneticOptimizer(
            evaluator=self.evaluator,
            population_size=5,
            num_generations=5,
            mutation_probability=0.2,
            crossover_probability=0.5,
            feedback_threshold=0.01,
            feedback_interval=2,
            early_stopping_rounds=3,
            n_jobs=1,
            use_default=True
        )

    def test_run(self):
        population, logbook = self.optimizer.run(csv_filename="test_genetic_algorithm_log.csv")
        self.assertTrue(len(population) > 0)
        self.assertIsInstance(population[0], dict)
        self.assertTrue(hasattr(population[0], "fitness"))
        self.assertTrue(population[0].fitness.valid)
        self.assertTrue(len(logbook) <= self.optimizer.num_generations)


    def test_crossover(self):
        individual1 = self.optimizer.toolbox.individual()
        individual2 = self.optimizer.toolbox.individual()
        original_individual1 = individual1.copy()
        original_individual2 = individual2.copy()
        self.optimizer.crossover_probability = 1.0
        offspring1, offspring2 = self.optimizer._cx_dict(individual1, individual2)  # noqa: SLF001
        self.assertNotEqual(offspring1, original_individual1)
        self.assertNotEqual(offspring2, original_individual2)

    def test_adjust_probabilities_increase(self):
        self.optimizer.mutation_probability = 0.2
        self.optimizer.crossover_probability = 0.5
        self.optimizer.last_avg_fitness = 0.9
        self.optimizer._adjust_probabilities(current_avg_fitness=0.905)  # noqa: SLF001
        self.assertAlmostEqual(self.optimizer.mutation_probability, 0.25)
        self.assertAlmostEqual(self.optimizer.crossover_probability, 0.55)

    def test_adjust_probabilities_decrease(self):
        self.optimizer.mutation_probability = 0.2
        self.optimizer.crossover_probability = 0.5
        self.optimizer.last_avg_fitness = 0.9
        self.optimizer._adjust_probabilities(current_avg_fitness=0.915) # noqa: SLF001
        self.assertAlmostEqual(self.optimizer.mutation_probability, 0.15)
        self.assertAlmostEqual(self.optimizer.crossover_probability, 0.45)

    def test_early_stopping(self):
        original_evaluate = self.evaluator.evaluate
        self.evaluator.evaluate = constant_evaluate
        population, logbook = self.optimizer.run()
        self.assertTrue(len(logbook) < self.optimizer.num_generations)
        self.assertEqual(self.optimizer.no_improvement, self.optimizer.early_stopping_rounds)
        self.evaluator.evaluate = original_evaluate

    def test_population_diversity(self):
        self.solver_options = Options(nsteps=100000, store_states=True)

        evaluator = Evaluator(self.circuit, self.noise_model, self.solver_options)
        optimizer = GeneticOptimizer(
            evaluator=evaluator,
            population_size=10,
            num_generations=10,
            mutation_probability=0.2,
            crossover_probability=0.5,
            feedback_threshold=0.01,
            feedback_interval=2,
            early_stopping_rounds=3,
            n_jobs=1,
            use_default=True
        )
        population, _ = optimizer.run()
        unique_individuals = {str(ind) for ind in population}
        self.assertTrue(len(unique_individuals) > 1)

    def test_toolbox_setup(self):
        self.assertTrue(hasattr(self.optimizer.toolbox, "individual"))
        self.assertTrue(hasattr(self.optimizer.toolbox, "population"))
        self.assertTrue(hasattr(self.optimizer.toolbox, "mate"))
        self.assertTrue(hasattr(self.optimizer.toolbox, "mutate"))
        self.assertTrue(hasattr(self.optimizer.toolbox, "select"))
        self.assertTrue(hasattr(self.optimizer.toolbox, "evaluate"))

    def test_invalid_evaluator(self):
        with self.assertRaises(AttributeError):
            invalid_evaluator = object()
            optimizer = GeneticOptimizer(evaluator=invalid_evaluator, use_default=True)
            optimizer._setup_toolbox() # noqa: SLF001

if __name__ == "__main__":
    unittest.main()
