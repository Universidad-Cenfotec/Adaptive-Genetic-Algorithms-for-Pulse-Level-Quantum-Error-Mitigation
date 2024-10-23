from abc import ABC, abstractmethod

from qutip import basis


class QuantumCircuitBase(ABC):
    """
    Abstract base class for quantum circuits.
    Defines the structure for creating a quantum circuit.
    """

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuit = self._create_circuit()
        self.initial_state = self._get_initial_state()
        self.target_state = self._get_target_state()

    @abstractmethod
    def _create_circuit(self):
        """
        Abstract method to create the quantum circuit.
        Must be implemented by derived classes.
        """

    def _get_initial_state(self):
        """
        Returns the initial quantum state |000...0>
        """
        return basis([2] * self.num_qubits, [0] * self.num_qubits)

    @abstractmethod
    def _get_target_state(self):
        """
        Abstract method to define the target state.
        Must be implemented by derived classes.
        """
