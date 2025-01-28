import math

from qutip import basis
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
        """
        If we start from |0...0>, the QFT(|0...0>) is a uniform superposition
            (1 / sqrt(2^n)) * sum_{k=0 to 2^n - 1} |k>.

        This is the ideal final state for the standard QFT on all-zero input,
        ignoring any global phases or bit-reversal.
        """
        dim = 2 ** self.num_qubits
        psi = 0
        for k in range(dim):
            psi += basis(dim, k)
        # Normalize
        return psi / math.sqrt(dim)
