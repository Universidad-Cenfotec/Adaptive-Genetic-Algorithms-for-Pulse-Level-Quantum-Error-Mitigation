import numpy as np
from qutip import basis, tensor
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class BellCircuit(QuantumCircuitBase):
    """
    Generates a 2-qubit Bell state using a Hadamard followed by a CNOT.
    """

    REQUIRED_NUM_QUBITS = 2

    def _create_circuit(self):
        if self.num_qubits != self.REQUIRED_NUM_QUBITS:
            msg = f"Bell circuit requires exactly {self.REQUIRED_NUM_QUBITS} qubits."
            raise ValueError(msg)

        circuit = QubitCircuit(self.num_qubits)
        circuit.add_gate("SNOT", targets=0)
        circuit.add_gate("CNOT", controls=0, targets=1)
        return circuit

    def _get_target_state(self):
        """
        Target Bell state |Φ+> = (|00> + |11>)/√2.
        """
        zero = basis(2, 0)
        one = basis(2, 1)
        bell_state = tensor(zero, zero) + tensor(one, one)
        return (1.0 / np.sqrt(2.0)) * bell_state
