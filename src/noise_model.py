# src/noise_model.py
import numpy as np
from qutip import destroy, basis, sigmax, sigmaz
from .quantum_utils import QuantumUtils

class NoiseModel:
    """
    Encapsulates the creation of noise channels.
    """
    def __init__(self, num_qubits, T1=30.0, T2=15.0, bit_flip_prob=0.05, phase_flip_prob=0.05):
        if num_qubits < 0:
            raise ValueError("The number of qubits must be non-negative.")
        self.num_qubits = num_qubits
        self.T1 = T1
        self.T2 = T2
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
            c_ops.append(np.sqrt(1.0 / self.T1) * relax_op)
            # Dephasing operator
            dephase_op = QuantumUtils.expand_operator(basis(2, 1) * basis(2, 1).dag(), qubit, self.num_qubits)
            c_ops.append(np.sqrt(1.0 / (2 * self.T2)) * dephase_op)
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
        # Depolarizing
        for qubit in range(self.num_qubits):
            p = self.bit_flip_prob  # Depolarizing probability
            depol_bit_flip = QuantumUtils.expand_operator(sigmax(), qubit, self.num_qubits)
            depol_phase_flip = QuantumUtils.expand_operator(sigmaz(), qubit, self.num_qubits)
            depol_bit_phase_flip = QuantumUtils.expand_operator(sigmax() * sigmaz(), qubit, self.num_qubits)
            c_ops.append(np.sqrt(p / 3) * depol_bit_flip)
            c_ops.append(np.sqrt(p / 3) * depol_phase_flip)
            c_ops.append(np.sqrt(p / 3) * depol_bit_phase_flip)
        return c_ops
