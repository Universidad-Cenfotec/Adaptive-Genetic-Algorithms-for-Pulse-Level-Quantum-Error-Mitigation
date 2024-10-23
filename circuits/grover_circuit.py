from qutip import basis
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class GroverCircuit(QuantumCircuitBase):
    """
    Quantum circuit for Grover's algorithm.
    """

    def _create_circuit(self):
        circuit = QubitCircuit(self.num_qubits)
        # Step 1: Apply Hadamard gates to all qubits
        for qubit in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=qubit)
        # Step 2: Apply Oracle (example: marking the state |11>)
        circuit.add_gate("X", targets=0)  # Example oracle, modify as needed
        circuit.add_gate("X", targets=1)
        circuit.add_gate("CNOT", controls=0, targets=1)
        circuit.add_gate("X", targets=0)
        circuit.add_gate("X", targets=1)
        # Step 3: Apply Grover diffusion operator (Hadamard, X, CNOT, X, Hadamard)
        for qubit in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=qubit)
            circuit.add_gate("X", targets=qubit)
        circuit.add_gate("CNOT", controls=0, targets=1)
        for qubit in range(self.num_qubits):
            circuit.add_gate("X", targets=qubit)
            circuit.add_gate("SNOT", targets=qubit)
        return circuit

    def _get_target_state(self):
        """
        Target state for Grover's algorithm (|11>)
        """
        return basis([2] * self.num_qubits, [1] * self.num_qubits)  # Target state |11...1>
