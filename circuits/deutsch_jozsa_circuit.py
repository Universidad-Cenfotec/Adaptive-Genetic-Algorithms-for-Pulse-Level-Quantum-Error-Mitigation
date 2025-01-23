from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class DeutschJozsaCircuit(QuantumCircuitBase):
    """
    Quantum circuit for the Deutsch-Jozsa algorithm with n qubits.
    Qubits 0 to n-2: Input qubits
    Qubit n-1       : Ancilla qubit
    """

    REQUIRED_NUM_QUBITS = 1

    def _create_circuit(self):
        if self.num_qubits == 1:
            error_message = (
                f"This Deutsch-Jozsa implementation requires more than {self.REQUIRED_NUM_QUBITS} qubits."
            )
            raise ValueError(error_message)

        circuit = QubitCircuit(self.num_qubits)

        # Step 1: Apply an X gate to the ancilla
        circuit.add_gate("X", targets=self.num_qubits-1)

        # Step 2: Apply Hadamard gates to all qubits
        for qubit in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=qubit)

        # Step 3: Example oracle implementation
        # Flip the ancilla qubit conditioned on the input qubits being in the state |1...1>
        # Specifically, apply a controlled NOT (CNOT) gate from each input qubit to the ancilla
        for cnotqubit in range(self.num_qubits-1):
            circuit.add_gate("CNOT", controls=cnotqubit, targets=self.num_qubits-1)

        # Step 4: Apply Hadamard gates again to the input qubits
        for qubit in range(self.num_qubits-1):
            circuit.add_gate("SNOT", targets=qubit)
        return circuit

    def _get_target_state(self):
        """
        The target state for the Deutsch-Jozsa algorithm
        is typically |0...0>, especially if we're expecting
        to measure the input qubits in the standard basis
        after the algorithm finishes.
        """
        return self._get_initial_state()
