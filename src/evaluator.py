from qutip import fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from src.visualizer import Visualizer


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

    def evaluate(self, individual, plot_pulses = False, outputdir = "", circuit_name = ""):  # noqa: FBT002
        """
        Evaluates an individual by running the circuit with the given parameters and computing the fidelity.
        Returns (0.0,) in case of an error during evaluation.
        """
        try:
            processor = OptPulseProcessor(
                num_qubits=self.num_qubits,
                model=SpinChainModel(self.num_qubits, setup="linear"),
            )
            processor.load_circuit(
                self.circuit, setting_args=individual, merge_gates=False,
            )
            # Run the evolution with noise
            result = processor.run_state(
                self.initial_state,
                options=self.solver_options,
                c_ops=self.c_ops
            )

            if plot_pulses:
                processor.plot_pulses(title=f"Pulses with optimization {circuit_name}", dpi=600)[0].savefig(outputdir/"pulseswithoptimization")

            # Visualization
                Visualizer.plot_pulses(
                    processor,
                    f"Optimized Pulses for {circuit_name}",
                    filename=outputdir / f"{circuit_name}_optimized_pulses.jpg"
                )

            # Compute fidelity with the target state
            return (fidelity(result.states[-1], self.target_state) , )

        except (ValueError, TypeError) as e:
            print(f"Value or Type error evaluating individual: {e}")
            return (0.0,)
        except AttributeError as e:
            print(f"Attribute error evaluating individual: {e}")
            return (0.0,)
        except Exception as e:
            print(f"Unexpected error evaluating individual: {e}")
            return (0.0,)
