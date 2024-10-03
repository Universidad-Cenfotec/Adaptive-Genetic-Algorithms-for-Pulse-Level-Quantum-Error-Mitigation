from qutip import qeye, tensor


class QuantumUtils:
    @staticmethod
    def expand_operator(op, qubit, num_qubits):
        """
        Expands a single-qubit operator to the full Hilbert space of multiple qubits.
        """
        operators = []
        for i in range(num_qubits):
            if i == qubit:
                operators.append(op)
            else:
                operators.append(qeye(2))
        return tensor(operators)
