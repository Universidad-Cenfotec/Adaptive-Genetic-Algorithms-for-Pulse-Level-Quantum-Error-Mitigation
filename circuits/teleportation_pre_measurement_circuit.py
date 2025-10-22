from qutip import basis, tensor
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class TeleportationPreMeasurementCircuit(QuantumCircuitBase):
    """
    Implements the standard three-qubit teleportation protocol
    up to the measurement stage.
    """

    REQUIRED_NUM_QUBITS = 3

    def _create_circuit(self):
        if self.num_qubits != self.REQUIRED_NUM_QUBITS:
            msg = (
                "Teleportation pre-measurement circuit requires exactly "
                f"{self.REQUIRED_NUM_QUBITS} qubits."
            )
            raise ValueError(msg)

        circuit = QubitCircuit(self.num_qubits)
        circuit.add_gate("SNOT", targets=1)
        circuit.add_gate("CNOT", controls=1, targets=2)
        circuit.add_gate("CNOT", controls=0, targets=1)
        circuit.add_gate("SNOT", targets=0)
        return circuit

    def _get_target_state(self):
        """
        State before measurement for an initial |000> input:
        (|000> + |100> + |011> + |111>) / 2.
        """
        zero = basis(2, 0)
        one = basis(2, 1)
        ket_000 = tensor(zero, zero, zero)
        ket_100 = tensor(one, zero, zero)
        ket_011 = tensor(zero, one, one)
        ket_111 = tensor(one, one, one)
        return 0.5 * (ket_000 + ket_100 + ket_011 + ket_111)
