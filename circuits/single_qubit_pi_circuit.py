from qutip import basis
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class SingleQubitPiCircuit(QuantumCircuitBase):
    """
    Applies a π pulse (Pauli-X) to flip a single qubit from |0> to |1>.
    """

    REQUIRED_NUM_QUBITS = 1

    def _create_circuit(self):
        if self.num_qubits != self.REQUIRED_NUM_QUBITS:
            msg = f"Single-qubit π circuit requires exactly {self.REQUIRED_NUM_QUBITS} qubit."
            raise ValueError(msg)

        circuit = QubitCircuit(self.num_qubits)
        circuit.add_gate("X", targets=0)
        return circuit

    def _get_target_state(self):
        """
        Target state |1> after a π rotation about the X-axis.
        """
        return basis([2], [1])
