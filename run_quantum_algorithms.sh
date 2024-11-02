#!/bin/bash

echo "Starting batch processing for quantum algorithms..."

echo "Running Grover's Algorithm with custom parameters..."
python3 main.py --algorithm grover --num_generations 3 --population_size 5 --t1 60.0 --t2 40.0 --bit_flip_prob 0.01 --phase_flip_prob 0.01 &

echo "Running Deutsch-Jozsa Algorithm with custom parameters..."
python3 main.py --algorithm deutsch-jozsa --num_generations 3 --population_size 5 --t1 50.0 --t2 30.0 --bit_flip_prob 0.02 --phase_flip_prob 0.02 &

wait

echo "Batch processing completed."
