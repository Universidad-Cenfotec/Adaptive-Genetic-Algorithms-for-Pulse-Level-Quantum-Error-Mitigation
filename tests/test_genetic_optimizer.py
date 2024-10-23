import unittest

from qutip import Options

from circuits.deutsch_jozsa_circuit import (
    DeutschJozsaCircuit,
)
from src.evaluator import Evaluator
from src.genetic_optimizer import GeneticOptimizer
from src.noise_model import NoiseModel


class TestGeneticOptimizer(unittest.TestCase):

    def setUp(self):
        num_qubits = 3
        self.circuit = DeutschJozsaCircuit(num_qubits)
        self.noise_model = NoiseModel(
            num_qubits,
            t1=50.0,
            t2=30.0,
            bit_flip_prob=0.02,
            phase_flip_prob=0.02,
        )
        self.solver_options = Options(nsteps=100000, store_states=True)
        self.evaluator = Evaluator(self.circuit, self.noise_model, self.solver_options)
        self.optimizer = GeneticOptimizer(self.evaluator, population_size=5, num_generations=3)

    def test_run(self):
        population = self.optimizer.run()
        self.assertTrue(len(population) > 0)
        self.assertIsInstance(population[0], dict)  # Individuals should be dicts


if __name__ == "__main__":
    unittest.main()