import unittest
from src.noise_model import NoiseModel
from qutip import Qobj

class TestNoiseModel(unittest.TestCase):

    def test_initialization(self):
        num_qubits = 3
        noise_model = NoiseModel(num_qubits)
        self.assertEqual(noise_model.num_qubits, num_qubits)
        self.assertEqual(noise_model.T1, 30.0)
        self.assertEqual(noise_model.T2, 15.0)

    def test_noise_channels(self):
        num_qubits = 2
        noise_model = NoiseModel(num_qubits)
        c_ops = noise_model._create_noise_channels()
        self.assertTrue(len(c_ops) > 0)  # Check that collapse operators are generated
        self.assertTrue(isinstance(c_ops[0], Qobj))

if __name__ == "__main__":
    unittest.main()
