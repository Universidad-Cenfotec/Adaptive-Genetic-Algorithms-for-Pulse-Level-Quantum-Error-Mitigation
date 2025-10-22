import numpy as np
from qutip import Qobj
from qutip_qip.circuit import Gate, QubitCircuit
from qutip_qip.decompose import decompose_one_qubit_gate

from src.quantum_circuit import QuantumCircuitBase


class InverseQuantumFourierCircuit(QuantumCircuitBase):
    """
    Standalone class implementing the Inverse QFT on 'num_qubits' qubits,
    mirroring the structure of QuantumFourierCircuit but all in one file.
    """

    def _create_circuit(self):
        """
        Build the IQFT circuit by calling a local method that generates
        a QubitCircuit of gates (the 'iqft_gate_sequence').
        By default, we skip the final SWAP by setting swapping=False.
        """
        return self._iqft_gate_sequence(
            N=self.num_qubits,
            swapping=False,
            to_cnot=True
        )

    def _get_target_state(self):
        """
        If there is no noise or additional transformations,
        we can obtain the 'ideal' final state by running the
        IQFT circuit on the initial state.
        """
        qc = self._iqft_gate_sequence(
            N=self.num_qubits,
            swapping=False,
            to_cnot=True
        )
        return qc.run(self._get_initial_state())

    def _iqft_gate_sequence(self, N=1, swapping=True, to_cnot=False):  # noqa: FBT002, N803
        """
        Internal method generating the gate sequence (a QubitCircuit)
        that implements the inverse QFT on N qubits.

        Parameters
        ----------
        N : int
            Number of qubits.
        swapping : bool
            Whether to include the SWAP gates (the inverse of the QFT final swaps).
        to_cnot : bool
            If True, decompose controlled-phase gates into CNOT + single-qubit rotations.

        Returns
        -------
        qc : QubitCircuit
            A QubitCircuit implementing the IQFT from left to right.

        """
        if N < 1:
            msg = "Minimum number of qubits is 1."
            raise ValueError(msg)

        qc = QubitCircuit(N)

        # Single-qubit case: IQFT is just H, same as QFT for one qubit.
        if N == 1:
            qc.add_gate("SNOT", targets=[0])
            return qc

        # If swapping=True, we do SWAP at the start (inverse of the final QFT swaps).
        if swapping:
            for i in range(N // 2):
                qc.add_gate("SWAP", targets=[i, N - i - 1])

        # Traverse qubits in reverse order
        for i in reversed(range(N)):
            # Hadamard on qubit i
            qc.add_gate("SNOT", targets=[i])

            # Controlled-phase with negative angles
            for j in reversed(range(i)):
                angle = -np.pi / (2 ** (i - j))
                if not to_cnot:
                    qc.add_gate(
                        "CPHASE",
                        targets=[j],  # cphase acts on target=j, controlled by i
                        controls=[i],
                        arg_label=r"{-\pi/2^{%d}}" % (i - j),
                        arg_value=angle
                    )
                else:
                    # Decompose cphase into CNOT + single-qubit rotations
                    decomposed_gates = self._cphase_to_cnot([j], [i], angle)
                    qc.gates.extend(decomposed_gates)

        return qc

    def _cphase_to_cnot(self, targets, controls, arg_value):
        """
        Decompose a controlled-phase gate into CNOT + single-qubit rotations,
        supporting negative angles. Mirrors the approach used in the QFT code.
        """
        rotation = Qobj([[1.0, 0.0],
                         [0.0, np.exp(1.0j * arg_value)]])
        decomposed_gates = list(decompose_one_qubit_gate(rotation, method="ZYZ_PauliX"))
        new_gates = []

        # Insert the first piece
        gate = decomposed_gates[0]
        gate.targets = targets
        new_gates.append(gate)

        new_gates.append(Gate("CNOT", targets=targets, controls=controls))

        gate = decomposed_gates[4]
        gate.targets = targets
        new_gates.append(gate)

        new_gates.append(Gate("CNOT", targets=targets, controls=controls))

        # Small RZ rotation on the control qubit
        new_gates.append(Gate("RZ", targets=controls, arg_value=arg_value / 2))

        gate = decomposed_gates[7]
        gate.arg_value += arg_value / 4
        new_gates.append(gate)
        return new_gates
