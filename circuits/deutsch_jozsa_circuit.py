from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class DeutschJozsaCircuit(QuantumCircuitBase):
    """
    Quantum circuit for the Deutsch-Jozsa algorithm.
    """

    def _create_circuit(self):
        circuit = QubitCircuit(self.num_qubits)
        # Step 1: Apply X gate to the last qubit
        circuit.add_gate("X", targets=self.num_qubits - 1)
        # Step 2: Apply Hadamard gates to all qubits
        for qubit in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=qubit)
        # Step 3: Apply CNOT gates from qubits 0 and 1 to qubit 2
        circuit.add_gate("CNOT", controls=0, targets=2)
        circuit.add_gate("CNOT", controls=1, targets=2)
        # Step 4: Apply Hadamard gates to the first two qubits
        for qubit in range(2):
            circuit.add_gate("SNOT", targets=qubit)
        return circuit

    def _get_target_state(self):
        """
        Target state for Deutsch-Jozsa algorithm (|000>)
        """
        return self._get_initial_state()
