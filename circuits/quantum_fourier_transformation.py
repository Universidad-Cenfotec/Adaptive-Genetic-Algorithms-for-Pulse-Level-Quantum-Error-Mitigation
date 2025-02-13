from qutip_qip.algorithms import qft_gate_sequence

from src.quantum_circuit import QuantumCircuitBase


class QuantumFourierCircuit(QuantumCircuitBase):
    """
    Quantum circuit for the standard QFT on 'num_qubits' qubits.
    Uses qft_gate_sequence from qutip_qip.algorithms (QuTiP).
    """

    def _create_circuit(self):
        """
        Build the QFT circuit using qft_gate_sequence,
        then add each gate to a QubitCircuit.

        By default, we do not apply the final SWAP (bit-reversal),
        since we set swapping=False.
        """
        # Generate the QFT sequence of gates for the given number of qubits
        # to_cnot=True => uses CNOT-based controlled-phase
        # swapping=False => does not do the final bit-reversal swap
        return qft_gate_sequence(
            self.num_qubits,
            swapping=False,
            to_cnot=True
        )

    def _get_target_state(self):
        qc = qft_gate_sequence(self.num_qubits, swapping=False, to_cnot=True)
        return qc.run(self._get_initial_state())
