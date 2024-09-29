from qutip import fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

class Evaluator:
    """
    Provides the evaluation function for the genetic algorithm.
    """
    def __init__(self, quantum_circuit, noise_model, solver_options):
        self.initial_state = quantum_circuit.initial_state
        self.target_state = quantum_circuit.target_state
        self.num_qubits = quantum_circuit.num_qubits
        self.circuit = quantum_circuit.circuit
        self.c_ops = noise_model.c_ops
        self.solver_options = solver_options

    def evaluate(self, individual):
        """
        Evaluates an individual by running the circuit with the given parameters and computing the fidelity.
        """
        processor = OptPulseProcessor(
            num_qubits=self.num_qubits,
            model=SpinChainModel(self.num_qubits, setup="linear")
        )
        processor.load_circuit(
            self.circuit, setting_args=individual, merge_gates=False
        )
        # Run the evolution with noise
        result = processor.run_state(self.initial_state, options=self.solver_options, c_ops=self.c_ops)
        # Compute fidelity with the target state
        fidelity_score = fidelity(result.states[-1], self.target_state)
        return (fidelity_score,)
