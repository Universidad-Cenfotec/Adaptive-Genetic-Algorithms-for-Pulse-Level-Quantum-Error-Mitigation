from qutip import basis
from qutip_qip.circuit import QubitCircuit

class QuantumCircuit:
    """
    Encapsulates the creation and configuration of the quantum circuit.
    """
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = self._create_circuit()
        self.initial_state = self._get_initial_state()
        self.target_state = self._get_target_state()

    def _create_circuit(self):
        """
        Creates the Deutsch-Jozsa circuit.
        """
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

    def _get_initial_state(self):
        """
        Returns the initial state |000>
        """
        return basis([2] * self.num_qubits, [0] * self.num_qubits)

    def _get_target_state(self):
        """
        Returns the target state |000> (assuming a constant function)
        """
        return basis([2] * self.num_qubits, [0] * self.num_qubits)