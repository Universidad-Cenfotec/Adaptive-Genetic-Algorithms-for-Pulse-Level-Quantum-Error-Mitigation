from qutip import basis
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class GroverCircuit(QuantumCircuitBase):
    """
    Example: Grover's algorithm circuit marking the state |111...1> for n qubits.
    """

    def _create_circuit(self):
        if self.num_qubits == 1:
            error_message = (
                f"This Grover circuit requires more than {self.REQUIRED_NUM_QUBITS} qubits."
            )
            raise ValueError(error_message)

        circuit = QubitCircuit(self.num_qubits)

        ############################################################
        # STEP 1: HADAMARD ON ALL QUBITS
        ############################################################
        # Create an equal superposition over all possible states for the given number of qubits.
        # Mathematically: Apply H^{\otimes n} to the initial state |0...0>, resulting in:
        #     |\psi\rangle = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle
        for q in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=q)

        ############################################################
        # STEP 2: ORACLE (Marking a specific state |111...1>)
        #
        # This implementation applies a phase flip to the marked state |111...1>.
        # General steps:
        #   - Apply X to all qubits (so |111...1> becomes |000...0>)
        #   - Apply a chain of CNOTs or a multi-controlled gate
        #   - Apply X to all qubits again
        # This effectively adds a phase of -1 to the marked state.
        ############################################################
        # Flip all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        # Chain of CNOTs to flip the last qubit if all are 0 (equivalently, original was |111...1>)
        for q in range(self.num_qubits-1):
            circuit.add_gate("CNOT", controls=q, targets=q+1)

        # Flip all qubits back
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        ############################################################
        # STEP 3: DIFFUSION OPERATOR
        #
        # The diffusion operator amplifies the amplitude of the marked state.
        # General steps for n qubits:
        #   - Hadamard on all qubits
        #   - X on all qubits
        #   - Multi-controlled Z (can be implemented via a chain of CNOTs)
        #   - X on all qubits
        #   - Hadamard on all qubits
        ############################################################

        # 3a. Hadamard on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=q)

        # 3b. X on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        # Chain of CNOTs to flip the last qubit if all are 0 (equivalently, original was |111...1>)
        for q in range(self.num_qubits-1):
            circuit.add_gate("CNOT", controls=q, targets=q+1)

        # 3d. X on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("X", targets=q)

        # 3e. Hadamard on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=q)

        return circuit

    def _get_target_state(self):
        """
        By default, consider the 'winning' or 'marked' state as |111...1> for n qubits.
        Adjust if you need a different marked state or multiple targets.
        """
        return basis([2] * self.num_qubits, [1] * self.num_qubits)
