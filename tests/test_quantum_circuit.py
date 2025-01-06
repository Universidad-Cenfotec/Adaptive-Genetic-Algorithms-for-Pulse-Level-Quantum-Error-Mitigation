import unittest

from qutip import basis

from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.grover_circuit import (
    GroverCircuit,
)


class TestQuantumCircuit(unittest.TestCase):

    def test_initialization_grover(self):
        num_qubits = 4
        circuit = GroverCircuit(num_qubits)
        self.assertEqual(circuit.num_qubits, num_qubits)

    def test_initialization_deutsch_jozsa(self):
        num_qubits = 4
        circuit = DeutschJozsaCircuit(num_qubits)
        self.assertEqual(circuit.num_qubits, num_qubits)

    def test_initial_state_grover(self):
        num_qubits = 4
        circuit = GroverCircuit(num_qubits)
        expected_state = basis([2] * num_qubits, [0] * num_qubits)
        self.assertTrue(circuit.initial_state == expected_state)

    def test_initial_state_deutsch_jozsa(self):
        num_qubits = 4
        circuit = DeutschJozsaCircuit(num_qubits)
        expected_state = basis([2] * num_qubits, [0] * num_qubits)
        self.assertTrue(circuit.initial_state == expected_state)

    def test_target_state_grover(self):
        num_qubits = 4
        circuit = GroverCircuit(num_qubits)
        expected_state = basis([2] * num_qubits, [1] * num_qubits)
        self.assertTrue(circuit.target_state == expected_state)

    def test_target_state_deutsch_jozsa(self):
        num_qubits = 4
        circuit = DeutschJozsaCircuit(num_qubits)
        expected_state = basis([2] * num_qubits, [0] * num_qubits)
        self.assertTrue(circuit.target_state == expected_state)


if __name__ == "__main__":
    unittest.main()
