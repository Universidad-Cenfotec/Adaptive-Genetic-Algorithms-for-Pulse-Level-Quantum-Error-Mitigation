from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class DeutschJozsaCircuit(QuantumCircuitBase):
    """
    Quantum circuit for the Deutsch-Jozsa algorithm with 4 qubits.
    Qubits 0, 1, 2: Input qubits
    Qubit 3       : Ancilla
    """

    def _create_circuit(self):
        # Ensure that num_qubits = 4
        if self.num_qubits != 4:
            raise ValueError("This Deutsch-Jozsa circuit implementation requires exactly 4 qubits.")

        circuit = QubitCircuit(self.num_qubits)

        # Step 1: Apply an X gate to the ancilla (qubit 3)
        circuit.add_gate("X", targets=3)

        # Step 2: Apply Hadamard (SNOT) gates to all 4 qubits [0, 1, 2, 3]
        for qubit in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=qubit)

        # Step 3: Oracle example.
        # For a typical Deutsch-Jozsa demonstration,
        # let's mark the state |111> on the 3 input qubits by flipping the ancilla:
        circuit.add_gate("CNOT", controls=0, targets=3)
        circuit.add_gate("CNOT", controls=1, targets=3)
        circuit.add_gate("CNOT", controls=2, targets=3)

        # Step 4: Apply Hadamard (SNOT) again on the first three qubits [0, 1, 2]
        for qubit in range(3):
            circuit.add_gate("SNOT", targets=qubit)

        return circuit

    def _get_target_state(self):
        """
        The target state for the Deutsch-Jozsa algorithm 
        is typically |0000>, especially if we're expecting 
        to measure the input qubits in the standard basis 
        after the algorithm finishes.
        """
        return self._get_initial_state()
