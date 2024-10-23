
# Pulse Quantum Error Correction

## Overview

This project implements a **genetic algorithm** to optimize a **Deutsch-Jozsa** and **Grover quantum circuits** in the presence of noise. The primary goal is to find the optimal parameters for quantum gates, maximizing the fidelity of the final quantum state with respect to the target state under realistic noise conditions. This project leverages **QuTiP**, **DEAP**, and other quantum computing libraries.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Unit Testing](#unit-testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Quantum Circuit Simulation**: Implements both the **Deutsch-Jozsa** and **Grover** algorithms.
- **Genetic Algorithm Optimization**: Uses a genetic algorithm to optimize pulse-level quantum gate parameters.
- **Noise Model**: Includes realistic noise models (e.g., bit flip, phase flip, and relaxation).
- **Visualization**: Provides tools for visualizing the optimization process and pulse sequences.

## Project Structure

```
quantum_optimization/
├── circuits/                     # New folder for quantum circuits
│   ├── __init__.py
│   ├── quantum_circuit_base.py   # Abstract base class for quantum circuits
│   ├── deutsch_jozsa_circuit.py  # Deutsch-Jozsa circuit implementation
│   └── grover_circuit.py         # Grover circuit implementation
├── src/
│   ├── __init__.py                # Initialization for the package
│   ├── evaluator.py               # Evaluation of genetic algorithm individuals
│   ├── genetic_optimizer.py       # Genetic algorithm implementation
│   ├── noise_model.py             # Noise models used in simulation
│   ├── quantum_utils.py           # Utility functions for quantum operations
│   └── visualizer.py              # Visualization of results and pulse sequences
├── tests/
│   ├── test_evaluator.py          # Unit tests for Evaluator class
│   ├── test_genetic_optimizer.py  # Unit tests for GeneticOptimizer class
│   ├── test_noise_model.py        # Unit tests for NoiseModel class
│   ├── test_quantum_circuit.py    # Unit tests for QuantumCircuit class
│   └── test_visualizer.py         # Unit tests for Visualizer class
├── .github/
│   └── workflows/
│       └── test.yml               # GitHub Actions workflow for continuous testing
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview (this file)
```

## Installation

### Python Version

This project requires **Python 3.12**. Make sure you have Python 3.12 installed before proceeding.

To check your Python version, run:

```
python --version
```

### Steps to install

Clone the repository

```
git clone https://github.com/Universidad-Cenfotec/Pulse-Quantum-Error-Correction.git

cd quantum_optimization
```
Create and activate a virtual environment

You can use venv or any other virtual environment manager:

```
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```
Install dependencies
Install the required Python packages listed in the requirements.txt file:

```
pip install -r requirements.txt
```

## Usage

```
python3 main.py --algorithm deutsch-jozsa
python3 main.py --algorithm grover
```

## Unit Testing

To run unit tests, use the following command:

```
python -m unittest discover -s tests
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the project.
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

William Aguilar-Calvo - [@Github](https://github.com/william2215) - wil-20-01@live.com

Project Link: [https://github.com/Universidad-Cenfotec/Pulse-Quantum-Error-Correction](https://github.com/Universidad-Cenfotec/Pulse-Quantum-Error-Correction)