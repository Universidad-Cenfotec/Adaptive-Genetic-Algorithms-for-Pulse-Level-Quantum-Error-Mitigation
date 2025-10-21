import unittest

from qutip import basis, tensor

from circuits.bell_circuit import BellCircuit
from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.ghz_circuit import GHZCircuit
from circuits.grover_circuit import GroverCircuit
from circuits.single_qubit_pi_circuit import SingleQubitPiCircuit
from circuits.teleportation_pre_measurement_circuit import (
    TeleportationPreMeasurementCircuit,
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

    def test_target_state_bell(self):
        circuit = BellCircuit(2)
        zero = basis(2, 0)
        one = basis(2, 1)
        expected_state = (tensor(zero, zero) + tensor(one, one)).unit()
        self.assertTrue(circuit.target_state == expected_state)

    def test_target_state_ghz(self):
        circuit = GHZCircuit(3)
        zero = basis(2, 0)
        one = basis(2, 1)
        expected_state = (tensor(zero, zero, zero) + tensor(one, one, one)).unit()
        self.assertTrue(circuit.target_state == expected_state)

    def test_target_state_single_qubit_pi(self):
        circuit = SingleQubitPiCircuit(1)
        expected_state = basis([2], [1])
        self.assertTrue(circuit.target_state == expected_state)

    def test_target_state_teleportation_pre_measurement(self):
        circuit = TeleportationPreMeasurementCircuit(3)
        zero = basis(2, 0)
        one = basis(2, 1)
        expected_state = 0.5 * (
            tensor(zero, zero, zero)
            + tensor(one, zero, zero)
            + tensor(zero, one, one)
            + tensor(one, one, one)
        )
        self.assertTrue(circuit.target_state == expected_state)


if __name__ == "__main__":
    unittest.main()
