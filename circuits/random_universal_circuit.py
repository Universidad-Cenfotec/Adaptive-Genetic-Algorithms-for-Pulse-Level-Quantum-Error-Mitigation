import numpy as np
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class RandomUniversalCircuit(QuantumCircuitBase):
    """
    Random universal circuit with many gates of various types
    to test pulse optimization under more complex conditions.
    """

    def _create_circuit(self):
        circuit = QubitCircuit(self.num_qubits)

        rng = np.random.default_rng()

        num_layers = 20  # more gates → deeper circuit
        gates = ["SNOT", "X", "CNOT", "SWAP", "CPHASE", "RZ"]

        for _ in range(num_layers):
            gate = rng.choice(gates)

            if gate == "SNOT":
                q = rng.integers(0, self.num_qubits)
                circuit.add_gate("SNOT", targets=q)

            elif gate == "X":
                q = rng.integers(0, self.num_qubits)
                circuit.add_gate("X", targets=q)

            elif gate == "CNOT":
                q1, q2 = rng.choice(self.num_qubits, size=2, replace=False)
                circuit.add_gate("CNOT", controls=q1, targets=q2)

            elif gate == "SWAP":
                q1, q2 = rng.choice(self.num_qubits, size=2, replace=False)
                circuit.add_gate("SWAP", targets=[q1, q2])

            elif gate == "CPHASE":
                q1, q2 = rng.choice(self.num_qubits, size=2, replace=False)
                angle = rng.uniform(-np.pi, np.pi)
                circuit.add_gate("CPHASE", controls=q1, targets=q2, arg_value=angle)

            elif gate == "RZ":
                q = rng.integers(0, self.num_qubits)
                angle = rng.uniform(-np.pi, np.pi)
                circuit.add_gate("RZ", targets=q, arg_value=angle)

        return circuit

    def _get_target_state(self):
        # For testing, assume target is all-zero state
        return self._get_initial_state()
