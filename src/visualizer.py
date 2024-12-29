from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.csv_logger import CSVLogger


class Visualizer:
    """
    Handles plotting for genetic algorithm optimization results.
    """

    @staticmethod
    def plot_pulses(processor, title, filename="optimized_pulses.jpg"):
        coeffs = processor.coeffs
        tlist = processor.get_full_tlist()
        control_labels = processor.get_control_labels()

        plt.figure(figsize=(12, 6))
        for i, coeff in enumerate(coeffs):
            plt.step(
                tlist[:-1],
                coeff,
                where="post",
                label=f"Control {i}: {control_labels[i]}",
            )
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(visible=True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_fidelity_evolution(logbook, filename="fidelity_evolution.jpg"):
        genetic_book = pd.DataFrame(logbook)
        plt.figure(figsize=(10, 5))
        plt.plot(genetic_book["gen"], genetic_book["avg"], label="Average Fidelity")
        plt.plot(genetic_book["gen"], genetic_book["max"], label="Max Fidelity")
        plt.fill_between(
            genetic_book["gen"],
            genetic_book["avg"] - genetic_book["std"],
            genetic_book["avg"] + genetic_book["std"],
            alpha=0.2,
            label="Standard Deviation",
        )
        plt.xlabel("Generation")
        plt.ylabel("Fidelity")
        plt.title("Fidelity Evolution during Optimization")
        plt.legend()
        plt.grid(visible=True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_histogram_fidelities(pop, filename="histogram_fidelities.jpg"):
        final_fidelities = [ind.fitness.values[0] for ind in pop]
        plt.figure(figsize=(8, 6))
        plt.hist(final_fidelities, bins=20, color="skyblue", edgecolor="black")
        plt.title("Distribution of Fidelities in Final Population")
        plt.xlabel("Fidelity")
        plt.ylabel("Number of Individuals")
        plt.grid(visible=True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_parameter_evolution(pop, parameters, filename_prefix="parameter_evolution"):
        for gate in parameters:
            num_tslots = [ind[gate]["num_tslots"] for ind in pop]
            evo_time = [ind[gate]["evo_time"] for ind in pop]

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            sns.boxplot(y=num_tslots, color="lightgreen")
            plt.title(f"Evolution of num_tslots for {gate}")
            plt.ylabel("num_tslots")
            plt.grid(visible=True)

            plt.subplot(1, 2, 2)
            sns.boxplot(y=evo_time, color="lightcoral")
            plt.title(f"Evolution of evo_time for {gate}")
            plt.ylabel("evo_time")
            plt.grid(visible=True)

            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_{gate}.jpg")
            plt.close()

    @staticmethod
    def plot_correlation(pop, parameters, output_dir, filename="correlation_matrix.jpg"):
        population_data = []
        for ind in pop:
            data = {
                "Fidelity": ind.fitness.values[0],
                "SNOT_num_tslots": ind["SNOT"]["num_tslots"],
                "SNOT_evo_time": ind["SNOT"]["evo_time"],
                "X_num_tslots": ind["X"]["num_tslots"],
                "X_evo_time": ind["X"]["evo_time"],
                "CNOT_num_tslots": ind["CNOT"]["num_tslots"],
                "CNOT_evo_time": ind["CNOT"]["evo_time"],
            }
            population_data.append(data)

        df = pd.DataFrame(population_data)

        # Heatmap
        plt.figure(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix between Parameters and Fidelity")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Scatter plots
        for gate in parameters:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.scatterplot(
                x=f"{gate}_num_tslots", y="Fidelity", data=df, alpha=0.7
            )
            plt.title(f"Fidelity vs num_tslots for {gate}")
            plt.xlabel("num_tslots")
            plt.ylabel("Fidelity")
            plt.grid(visible=True)

            plt.subplot(1, 2, 2)
            sns.scatterplot(
                x=f"{gate}_evo_time", y="Fidelity", data=df, alpha=0.7
            )
            plt.title(f"Fidelity vs evo_time for {gate}")
            plt.xlabel("evo_time")
            plt.ylabel("Fidelity")
            plt.grid(visible=True)

            plt.tight_layout()
            plt.savefig(str(output_dir / f"{gate}_fidelity_vs_parameters.jpg"))
            plt.close()

    @staticmethod
    def plot_histogram_parameters(pop, parameters, filename_prefix="histogram_parameters"):
        population_data = []
        for ind in pop:
            data = {
                "SNOT_num_tslots": ind["SNOT"]["num_tslots"],
                "SNOT_evo_time": ind["SNOT"]["evo_time"],
                "X_num_tslots": ind["X"]["num_tslots"],
                "X_evo_time": ind["X"]["evo_time"],
                "CNOT_num_tslots": ind["CNOT"]["num_tslots"],
                "CNOT_evo_time": ind["CNOT"]["evo_time"],
            }
            population_data.append(data)

        df = pd.DataFrame(population_data)

        for gate in parameters:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            sns.histplot(
                df[f"{gate}_num_tslots"], bins=10, kde=True,
                color="green", edgecolor="black"
            )
            plt.title(f"Distribution of num_tslots for {gate}")
            plt.xlabel("num_tslots")
            plt.ylabel("Frequency")
            plt.grid(visible=True)

            plt.subplot(1, 2, 2)
            sns.histplot(
                df[f"{gate}_evo_time"], bins=10, kde=True,
                color="purple", edgecolor="black"
            )
            plt.title(f"Distribution of evo_time for {gate}")
            plt.xlabel("evo_time")
            plt.ylabel("Frequency")
            plt.grid(visible=True)

            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_{gate}.jpg")
            plt.close()

    @staticmethod
    def plot_fidelity_comparison(fidelity_no_opt, fidelity_opt, circuit_name):
        """
        Simple bar chart to compare fidelities (no opt vs. with opt).
        """
        import matplotlib.pyplot as plt

        logger = CSVLogger(circuit_name)
        logger.write_fidelity_comparison(fidelity_no_opt, fidelity_opt)

        labels = ["No Optimization", "Genetic Optimization"]
        fidelities = [fidelity_no_opt, fidelity_opt]
        colors = ["red", "green"]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, fidelities, color=colors)
        plt.ylim(0, 1)
        plt.ylabel("Fidelity")
        plt.title(f"Fidelity Comparison for {circuit_name}")

        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom"
            )

        output_dir = Path("output_circuits") / circuit_name
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = output_dir / f"{circuit_name}_fidelities_comparison.jpg"
        plt.savefig(plot_filename)
        plt.close()

        print(f"\nFidelity comparison plot saved at {plot_filename}")
