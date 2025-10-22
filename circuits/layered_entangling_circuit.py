import numpy as np
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class LayeredEntanglingCircuit(QuantumCircuitBase):
    """
    Circuit with repeated layers of entangling gates to increase
    circuit depth and pulse complexity.
    """

    def __init__(self, num_qubits, num_layers=5):
        self.num_layers = num_layers
        super().__init__(num_qubits)

    def _create_circuit(self):
        circuit = QubitCircuit(self.num_qubits)
        rng = np.random.default_rng()

        for _layer in range(self.num_layers):
            # SNOT layer
            for q in range(self.num_qubits):
                circuit.add_gate("SNOT", targets=q)

            # CNOT ladder
            for q in range(self.num_qubits - 1):
                circuit.add_gate("CNOT", controls=q, targets=q+1)

            # RZ layer with random angles
            for q in range(self.num_qubits):
                angle = rng.uniform(-np.pi, np.pi)
                circuit.add_gate("RZ", targets=q, arg_value=angle)

            # CPHASE layer
            for q in range(self.num_qubits - 1):
                angle = rng.uniform(-np.pi, np.pi)
                circuit.add_gate("CPHASE", controls=q, targets=q+1, arg_value=angle)

            # SWAP layer
            if self.num_qubits > 1:
                q1, q2 = rng.choice(self.num_qubits, size=2, replace=False)
                circuit.add_gate("SWAP", targets=[q1, q2])

        return circuit

    def _get_target_state(self):
        # For testing, assume target is all-zero state
        return self._get_initial_state()
