import argparse
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

from qutip import Options, fidelity
from qutip_qip.device import OptPulseProcessor, SpinChainModel

from circuits.bernstein_vaizirani_circuit import BernsteinVaziraniCircuit
from circuits.deutsch_jozsa_circuit import DeutschJozsaCircuit
from circuits.grover_circuit import GroverCircuit
from circuits.inverse_quantum_fourier_transformation import InverseQuantumFourierCircuit
from circuits.quantum_fourier_transformation import QuantumFourierCircuit
from src.csv_logger import CSVLogger
from src.evaluator import Evaluator
from src.gate_config import DEFAULT_SETTING_ARGS
from src.genetic_optimizer import GeneticOptimizer
from src.noise_model import NoiseModel
from src.visualizer import Visualizer

warnings.filterwarnings("ignore")

UNSUPPORTED_ALGORITHM_SPECIFIED = "Unsupported algorithm specified."

def run_algorithm_without_optimization(
    quantum_circuit, num_qubits, circuit_name, noise_model, logger, output_dir
):
    """
    Runs the circuit under noise WITHOUT GA optimization, returns fidelity.
    """
    # Solver options
    solver_options = Options(nsteps=100000, store_states=True)

    processor_no_opt = OptPulseProcessor(
        num_qubits=num_qubits,
        model=SpinChainModel(num_qubits, setup="linear"),
    )
    processor_no_opt.load_circuit(
        quantum_circuit.circuit, setting_args=DEFAULT_SETTING_ARGS, merge_gates=False
    )

    result_no_opt = processor_no_opt.run_state(
        quantum_circuit.initial_state,
        options=solver_options,
        c_ops=noise_model.c_ops,
    )

    fidelity_no_opt = fidelity(result_no_opt.states[-1], quantum_circuit.target_state)
    print(f"\nFidelity WITHOUT optimization for {circuit_name}: {fidelity_no_opt:.4f}")

    # Write CSV summary
    logger.write_summary_no_optimization(noise_model, fidelity_no_opt)
    processor_no_opt.plot_pulses(title=f"Pulses without optimization {circuit_name}", dpi=600)[0].savefig(output_dir/"pulseswithoutoptimization")
    # Visualization
    Visualizer.plot_pulses(
        processor_no_opt,
        f"Optimized Pulses for {circuit_name}",
        filename=output_dir / f"{circuit_name}_non_optimized_pulses.jpg"
    )
    return fidelity_no_opt

def run_algorithm(
    quantum_circuit,
    num_qubits,
    circuit_name,
    population_size,
    num_generations,
    noise_model,
    logger,
    output_dir,
):
    """
    Runs the genetic optimization on the circuit under noise, returns fidelity.
    """
    solver_options = Options(nsteps=100000, store_states=True)

    # Example evaluator that must implement "evaluate(individual)"
    evaluator = Evaluator(quantum_circuit, noise_model, solver_options)

    # GA optimizer
    optimizer = GeneticOptimizer(
        evaluator=evaluator,
        population_size=population_size,
        num_generations=num_generations,
        use_default=True,
    )

    # Run GA
    pop, logbook = optimizer.run(csv_logger=logger, csv_filename=output_dir / f"{circuit_name}_log.csv")
    best_individual = optimizer.hall_of_fame[0]
    evaluator.evaluate(best_individual, plot_pulses=True, outputdir=output_dir, circuit_name=circuit_name)
    best_fidelity = max(logbook.select("avg"))
    print(f"\nBest individual for {circuit_name}: {best_individual}")
    print(f"Fidelity (best_individual): {best_fidelity:.4f}")

    # Use best individual to set pulses
    processor_optimized = OptPulseProcessor(
        num_qubits=num_qubits, model=SpinChainModel(num_qubits, setup="linear")
    )
    processor_optimized.load_circuit(
        quantum_circuit.circuit, setting_args=best_individual, merge_gates=False
    )

    # Write summary CSV
    logger.write_summary_optimization(
        noise_model,
        best_individual,
        best_fidelity,
        population_size,
        num_generations
    )
    # Write pulses CSV
    logger.write_pulses(processor_optimized)

    # Visualization
    Visualizer.plot_pulses(
        processor_optimized,
        f"Optimized Pulses for {circuit_name}",
        filename=output_dir / f"{circuit_name}_optimized_pulses.jpg"
    )
    Visualizer.plot_fidelity_evolution(
        logbook,
        filename=output_dir / f"{circuit_name}_fidelity_evolution.jpg"
    )
    Visualizer.plot_histogram_fidelities(
        pop,
        filename=output_dir / f"{circuit_name}_histogram_fidelities.jpg"
    )

    # Parameter plots
    parameters = ["SNOT", "X", "CNOT"]
    Visualizer.plot_parameter_evolution(
        pop, parameters,
        filename_prefix=str(output_dir / f"{circuit_name}_parameter_evolution")
    )
    Visualizer.plot_correlation(
        pop, parameters,
        output_dir=output_dir,
        filename=output_dir / f"{circuit_name}_correlation_matrix.jpg"
    )
    Visualizer.plot_histogram_parameters(
        pop, parameters,
        filename_prefix=str(output_dir / f"{circuit_name}_histogram_parameters")
    )

    return best_fidelity

def main():
    try:
        parser = argparse.ArgumentParser(
            description="Run quantum algorithms with or without GA optimization under noise."
        )
        parser.add_argument("--algorithm", type=str, choices=["grover", "deutsch-jozsa", "bernstein-vazirani", "qft", "iqft"], required=True,
                            help="Specify which algorithm to run: 'grover' or 'deutsch-jozsa'.")
        parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits to use in the circuit.")
        parser.add_argument("--num_generations", type=int, default=100, help="Generations for GA.")
        parser.add_argument("--population_size", type=int, default=50, help="Population size for GA.")
        parser.add_argument("--t1", type=float, default=50.0, help="T1 relaxation time.")
        parser.add_argument("--t2", type=float, default=30.0, help="T2 dephasing time.")
        parser.add_argument("--bit_flip_prob", type=float, default=0.02, help="Bit-flip probability.")
        parser.add_argument("--phase_flip_prob", type=float, default=0.02, help="Phase-flip probability.")
        args = parser.parse_args()

        # Choose circuit
        if args.algorithm == "grover":
            circuit_name = f"Grover_{args.num_qubits}Q"
            quantum_circuit = GroverCircuit(args.num_qubits)
        elif args.algorithm == "deutsch-jozsa":
            circuit_name = f"DeutschJozsa_{args.num_qubits}Q"
            quantum_circuit = DeutschJozsaCircuit(args.num_qubits)
        elif args.algorithm == "bernstein-vazirani":
            circuit_name = f"BernsteinVazirani_{args.num_qubits}Q"
            quantum_circuit = BernsteinVaziraniCircuit(args.num_qubits, None)
        elif args.algorithm == "qft":
            circuit_name = f"QFT_{args.num_qubits}Q"
            quantum_circuit = QuantumFourierCircuit(args.num_qubits)
        elif args.algorithm == "iqft":
            circuit_name = f"IQFT_{args.num_qubits}Q"
            quantum_circuit = InverseQuantumFourierCircuit(args.num_qubits)
        else:
            raise ValueError(UNSUPPORTED_ALGORITHM_SPECIFIED)  # noqa: TRY301

        # Create output directory
        timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("output_circuits") / f"{circuit_name}_{timestamp_folder}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create noise model
        noise_model = NoiseModel(
            args.num_qubits,
            t1=args.t1,
            t2=args.t2,
            bit_flip_prob=args.bit_flip_prob,
            phase_flip_prob=args.phase_flip_prob,
        )

        # Create CSV logger
        logger = CSVLogger(circuit_name, output_dir=output_dir)

        # Track total experiment time
        start_time = time.time()

        # 1) Run WITHOUT optimization
        fidelity_no_opt = run_algorithm_without_optimization(
            quantum_circuit,
            args.num_qubits,
            circuit_name + "_No_Opt",
            noise_model,
            logger,
            output_dir=output_dir
        )

        # 2) Run WITH optimization
        fidelity_opt = run_algorithm(
            quantum_circuit,
            args.num_qubits,
            circuit_name + "_With_Opt",
            population_size=args.population_size,
            num_generations=args.num_generations,
            noise_model=noise_model,
            logger=logger,
            output_dir=output_dir,
        )

        # Calculate total time elapsed
        total_time = time.time() - start_time
        print(f"\nTotal experiment time: {total_time:.2f} seconds")

        # 3) Compare using Visualizer
        Visualizer.plot_fidelity_comparison(fidelity_no_opt, fidelity_opt, circuit_name, output_dir)

        # Log total time
        logger.write_experiment_time(total_time)

    except Exception as e:
        print("\n[ERROR] An error occurred during execution!", file=sys.stderr)
        print(f"[DETAILS] {e!s}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)  # 🔴 Exit with an error code

if __name__ == "__main__":
    main()
