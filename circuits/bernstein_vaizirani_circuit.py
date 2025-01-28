import random

from qutip import basis, tensor
from qutip_qip.circuit import QubitCircuit

from src.quantum_circuit import QuantumCircuitBase


class BernsteinVaziraniCircuit(QuantumCircuitBase):
    """
    Bernstein-Vazirani circuit with a random hidden bitstring (by default).
    
    Qubits 0 .. (n-2) : Input qubits
    Qubit (n-1)       : Ancilla qubit

    The secret_string is a list of bits (0 or 1) of length (num_qubits - 1).
    If secret_string is None, it is generated randomly.
    """

    REQUIRED_NUM_QUBITS = 2  # at least 1 input + 1 ancilla

    def __init__(self, num_qubits, secret_string=None):
        """
        :param num_qubits: Total number of qubits (input + 1 ancilla).
        :param secret_string: Optional. If None, a random bitstring 
                              of length (num_qubits - 1) is generated.
        """
        if num_qubits < self.REQUIRED_NUM_QUBITS:
            raise ValueError(
                f"Bernstein-Vazirani requires at least {self.REQUIRED_NUM_QUBITS} qubits."
            )

        # If user does not supply a secret string, generate a random one:
        if secret_string is None:
            secret_string = [random.randint(0, 1) for _ in range(num_qubits - 1)]
        elif len(secret_string) != num_qubits - 1:
            raise ValueError(
                f"Length of 'secret_string' must be (num_qubits - 1). "
                f"Received {len(secret_string)}, expected {num_qubits - 1}."
            )

        self.secret_string = secret_string
        super().__init__(num_qubits)

    def _create_circuit(self):
        """
        1) Initialize ancilla qubit to |1> by applying X.
        2) Apply Hadamard gates to all qubits (including ancilla).
        3) Oracle: For each bit i where secret_string[i] == 1, apply CNOT(i -> ancilla).
        4) Apply Hadamard gates again to the input qubits (0 .. n-2).
        """
        circuit = QubitCircuit(self.num_qubits)

        # 1) Ancilla to |1>
        circuit.add_gate("X", targets=self.num_qubits - 1)

        # 2) Hadamard on all qubits
        for q in range(self.num_qubits):
            circuit.add_gate("SNOT", targets=q)

        # 3) Oracle: For each '1' bit in secret_string, apply CNOT
        for i, bit in enumerate(self.secret_string):
            if bit == 1:
                circuit.add_gate("CNOT", controls=i, targets=self.num_qubits - 1)

        # 4) Hadamard again on input qubits only (0 .. n-2)
        for q in range(self.num_qubits - 1):
            circuit.add_gate("SNOT", targets=q)

        return circuit

    def _get_target_state(self):
        """
        In an ideal scenario with no noise, measuring the input qubits yields the secret string,
        and the ancilla remains in |1>.

        Hence, final state is |s_0 s_1 ... s_(n-2)> tensor |1>.
        """
        basis_list = []
        for bit in self.secret_string:
            basis_list.append(basis(2, bit))  # |0> or |1> for each input bit
        # Ancilla is |1>
        basis_list.append(basis(2, 1))

        return tensor(basis_list)
