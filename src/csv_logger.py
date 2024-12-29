import csv
from pathlib import Path


class CSVLogger:
    """
    Handles all CSV-writing logic in a centralized fashion.
    """

    def __init__(self, circuit_name):
        self.circuit_name = circuit_name

    def write_summary_no_optimization(self, noise_model, fidelity_no_opt):
        """
        Writes a CSV row summarizing a 'no optimization' run for a circuit.
        """
        summary_file = Path(f"{self.circuit_name}_summary_no_optimization.csv")
        fieldnames = [
            "circuit_name", "t1", "t2",
            "bit_flip_prob", "phase_flip_prob", "fidelity"
        ]

        write_header = not summary_file.exists()
        with summary_file.open(mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            writer.writerow({
                "circuit_name": self.circuit_name,
                "t1": noise_model.t1,
                "t2": noise_model.t2,
                "bit_flip_prob": noise_model.bit_flip_prob,
                "phase_flip_prob": noise_model.phase_flip_prob,
                "fidelity": fidelity_no_opt
            })

    def write_summary_optimization(
        self,
        noise_model,
        best_individual,
        best_fidelity,
        final_fidelity_with_noise,
        population_size,
        num_generations,
    ):
        """
        Writes a CSV row summarizing a 'with optimization' run for a circuit.
        """
        summary_file = Path(f"{self.circuit_name}_summary_optimization.csv")
        fieldnames = [
            "circuit_name", "population_size", "num_generations", "t1", "t2",
            "bit_flip_prob", "phase_flip_prob", "best_individual", "best_fidelity",
            "final_fidelity_with_noise"
        ]

        write_header = not summary_file.exists()
        with summary_file.open(mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            writer.writerow({
                "circuit_name": self.circuit_name,
                "population_size": population_size,
                "num_generations": num_generations,
                "t1": noise_model.t1,
                "t2": noise_model.t2,
                "bit_flip_prob": noise_model.bit_flip_prob,
                "phase_flip_prob": noise_model.phase_flip_prob,
                "best_individual": str(best_individual),
                "best_fidelity": best_fidelity,
                "final_fidelity_with_noise": final_fidelity_with_noise
            })

    def write_pulses(self, processor, filename_suffix="_pulses_optimization.csv"):
        """
        Saves the time slots and pulse amplitudes for the optimized processor.
        """
        pulses_file = Path(f"{self.circuit_name}{filename_suffix}")
        tlist = processor.get_full_tlist()
        coeffs = processor.get_full_coeffs()  # one array per control channel

        pulse_headers = ["time"] + [f"pulse_channel_{i+1}" for i in range(len(coeffs))]

        with pulses_file.open(mode="w", newline="") as pf:
            writer = csv.writer(pf)
            writer.writerow(pulse_headers)
            num_points = len(tlist) - 1
            for idx in range(num_points):
                row = [tlist[idx]] + [c[idx] for c in coeffs]
                writer.writerow(row)

    def write_fidelity_comparison(self, fidelity_no_opt, fidelity_opt):
        """
        Compares fidelities (no opt vs. opt) in a single CSV row.
        """
        comparison_file = Path(f"{self.circuit_name}_fidelity_comparison.csv")
        fieldnames = [
            "circuit_name", "fidelity_no_optimization",
            "fidelity_with_optimization", "fidelity_improvement"
        ]
        write_header = not comparison_file.exists()
        with comparison_file.open(mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            writer.writerow({
                "circuit_name": self.circuit_name,
                "fidelity_no_optimization": fidelity_no_opt,
                "fidelity_with_optimization": fidelity_opt,
                "fidelity_improvement": fidelity_opt - fidelity_no_opt
            })

    def write_logbook_header(self, logbook, csv_filename):
        """
        Writes the header for the GA logbook.
        """
        csv_path = Path(csv_filename)
        with csv_path.open(mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(logbook.header)

    def append_logbook_entry(self, logbook, csv_filename):
        """
        Appends the latest logbook record to the CSV log.
        """
        csv_path = Path(csv_filename)
        record = logbook[-1]

        with csv_path.open(mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # We must ensure the order matches logbook.header
            row_data = [record[k] for k in logbook.header if k in record]
            writer.writerow(row_data)
