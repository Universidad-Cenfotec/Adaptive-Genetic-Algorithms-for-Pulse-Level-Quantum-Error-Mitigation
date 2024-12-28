from qutip import basis
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class GroverCircuit(QuantumCircuitBase):
    """
    Example: 4-qubit Grover's algorithm circuit marking the state |1111>.
    """

    def _create_circuit(self):
        if self.num_qubits != 4:
            raise ValueError("This Grover circuit is set up for exactly 4 qubits.")

        circuit = QubitCircuit(self.num_qubits)

        ############################################################
        # STEP 1: HADAMARD ON ALL QUBITS
        ############################################################
        # Create an equal superposition over the 16 possible states.
        for q in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=q)

        ############################################################
        # STEP 2: ORACLE (Marking |1111>)
        #
        # One way to implement a 'phase flip' on |1111> is:
        #   - Apply X to all qubits (so |1111> becomes |0000>)
        #   - Apply a chain of CNOTs or a multi-controlled gate
        #   - Apply X to all qubits again
        #
        # This effectively adds a phase of -1 to |1111>.
        ############################################################
        # Flip all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        # Chain of CNOTs to flip the last qubit if all are 0 (equivalently, original was |1111>)
        circuit.add_gate("CNOT", controls=0, targets=1)
        circuit.add_gate("CNOT", controls=1, targets=2)
        circuit.add_gate("CNOT", controls=2, targets=3)

        # Flip all qubits back
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        ############################################################
        # STEP 3: DIFFUSION OPERATOR
        #
        # For 4 qubits, the diffusion operator is:
        #   - Hadamard on all qubits
        #   - X on all qubits
        #   - Multi-controlled Z (again, can be done via chain of CNOTs)
        #   - X on all qubits
        #   - Hadamard on all qubits
        ############################################################

        # 3a. Hadamard on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=q)

        # 3b. X on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        # 3c. Chain of CNOTs to perform a multi-controlled phase flip
        circuit.add_gate("CNOT", controls=0, targets=1)
        circuit.add_gate("CNOT", controls=1, targets=2)
        circuit.add_gate("CNOT", controls=2, targets=3)

        # 3d. X on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        # 3e. Hadamard on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=q)

        return circuit

    def _get_target_state(self):
        """
        By default, let's consider the 'winning' or 'marked' state as |1111>.
        Adjust if you need a different marked state or multiple targets.
        """
        return basis([2] * self.num_qubits, [1] * self.num_qubits)
