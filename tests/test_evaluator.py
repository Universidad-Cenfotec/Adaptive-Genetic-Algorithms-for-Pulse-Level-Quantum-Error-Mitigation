import unittest
from src.evaluator import Evaluator
from src.noise_model import NoiseModel
from src.quantum_circuit import QuantumCircuit
from qutip import basis, Options

class TestEvaluator(unittest.TestCase):

    def setUp(self):
        num_qubits = 3
        self.circuit = QuantumCircuit(num_qubits)
        self.noise_model = NoiseModel(num_qubits)
        self.solver_options = Options(nsteps=100000, store_states=True)
        self.evaluator = Evaluator(self.circuit, self.noise_model, self.solver_options)

    def test_initialization(self):
        self.assertEqual(self.evaluator.num_qubits, self.circuit.num_qubits)

    def test_evaluate(self):
        individual = {
            "SNOT": {"num_tslots": 5, "evo_time": 1.5},
            "X": {"num_tslots": 3, "evo_time": 0.5},
            "CNOT": {"num_tslots": 10, "evo_time": 5.0},
        }
        fidelity_score = self.evaluator.evaluate(individual)
        self.assertTrue(0 <= fidelity_score[0] <= 1)

if __name__ == "__main__":
    unittest.main()
