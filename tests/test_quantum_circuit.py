import unittest
from src.quantum_circuit import QuantumCircuit
from qutip import basis

class TestQuantumCircuit(unittest.TestCase):

    def test_initialization(self):
        num_qubits = 3
        circuit = QuantumCircuit(num_qubits)
        self.assertEqual(circuit.num_qubits, num_qubits)

    def test_initial_state(self):
        num_qubits = 3
        circuit = QuantumCircuit(num_qubits)
        expected_state = basis([2] * num_qubits, [0] * num_qubits)
        self.assertTrue(circuit.initial_state == expected_state)

    def test_target_state(self):
        num_qubits = 3
        circuit = QuantumCircuit(num_qubits)
        expected_state = basis([2] * num_qubits, [0] * num_qubits)
        self.assertTrue(circuit.target_state == expected_state)

if __name__ == "__main__":
    unittest.main()
