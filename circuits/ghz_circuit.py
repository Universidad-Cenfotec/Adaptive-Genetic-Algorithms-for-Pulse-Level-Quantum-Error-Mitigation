import numpy as np
from qutip import basis, tensor
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class GHZCircuit(QuantumCircuitBase):
    """
    Builds a 3-qubit GHZ state using a Hadamard and two controlled-NOT gates.
    """

    REQUIRED_NUM_QUBITS = 3

    def _create_circuit(self):
        if self.num_qubits != self.REQUIRED_NUM_QUBITS:
            msg = f"GHZ circuit requires exactly {self.REQUIRED_NUM_QUBITS} qubits."
            raise ValueError(msg)

        circuit = QubitCircuit(self.num_qubits)
        circuit.add_gate("SNOT", targets=0)
        circuit.add_gate("CNOT", controls=0, targets=1)
        circuit.add_gate("CNOT", controls=0, targets=2)
        return circuit

    def _get_target_state(self):
        """
        Target GHZ state |GHZ> = (|000> + |111>)/√2.
        """
        zero = basis(2, 0)
        one = basis(2, 1)
        all_zero = tensor(zero, zero, zero)
        all_one = tensor(one, one, one)
        return (1.0 / np.sqrt(2.0)) * (all_zero + all_one)
