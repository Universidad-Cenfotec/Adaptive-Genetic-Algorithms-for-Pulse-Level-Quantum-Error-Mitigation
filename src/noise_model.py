import numpy as np
from qutip import basis, destroy, sigmax, sigmaz

from .quantum_utils import QuantumUtils

NON_NEGATIVE_QUBIT_ERROR = "The number of qubits must be non-negative."

class NoiseModel:
    """
    Encapsulates the creation of noise channels.
    """

    def __init__(self, num_qubits, t1=30.0, t2=15.0, bit_flip_prob=0.05, phase_flip_prob=0.05):
        if num_qubits < 0:
            raise ValueError(NON_NEGATIVE_QUBIT_ERROR)
        self.num_qubits = num_qubits
        self.t1 = t1
        self.t2 = t2
        self.bit_flip_prob = bit_flip_prob
        self.phase_flip_prob = phase_flip_prob
        self.c_ops = self._create_noise_channels()

    def _create_noise_channels(self):
        """
        Creates the collapse operators representing different types of noise.
        """
        c_ops = []
        # Relaxation and dephasing noise for each qubit
        for qubit in range(self.num_qubits):
            # Relaxation operator
            relax_op = QuantumUtils.expand_operator(destroy(2), qubit, self.num_qubits)
            c_ops.append(np.sqrt(1.0 / self.t1) * relax_op)
            # Dephasing operator
            dephase_op = QuantumUtils.expand_operator(basis(2, 1) * basis(2, 1).dag(), qubit, self.num_qubits)
            c_ops.append(np.sqrt(1.0 / (2 * self.t2)) * dephase_op)
        # Bit Flip
        for qubit in range(self.num_qubits):
            bit_flip_op = QuantumUtils.expand_operator(sigmax(), qubit, self.num_qubits)
            c_ops.append(np.sqrt(self.bit_flip_prob) * bit_flip_op)
        # Phase Flip
        for qubit in range(self.num_qubits):
            phase_flip_op = QuantumUtils.expand_operator(sigmaz(), qubit, self.num_qubits)
            c_ops.append(np.sqrt(self.phase_flip_prob) * phase_flip_op)
        # Bit-Phase Flip
        for qubit in range(self.num_qubits):
            bit_phase_flip_op = QuantumUtils.expand_operator(sigmax() * sigmaz(), qubit, self.num_qubits)
            c_ops.append(np.sqrt(self.bit_flip_prob * self.phase_flip_prob) * bit_phase_flip_op)
        # # Depolarizing
        for qubit in range(self.num_qubits):
            p = self.bit_flip_prob  # Depolarizing probability
            depol_bit_flip = QuantumUtils.expand_operator(sigmax(), qubit, self.num_qubits)
            depol_phase_flip = QuantumUtils.expand_operator(sigmaz(), qubit, self.num_qubits)
            depol_bit_phase_flip = QuantumUtils.expand_operator(sigmax() * sigmaz(), qubit, self.num_qubits)
            c_ops.append(np.sqrt(p / 3) * depol_bit_flip)
            c_ops.append(np.sqrt(p / 3) * depol_phase_flip)
            c_ops.append(np.sqrt(p / 3) * depol_bit_phase_flip)
        return c_ops
