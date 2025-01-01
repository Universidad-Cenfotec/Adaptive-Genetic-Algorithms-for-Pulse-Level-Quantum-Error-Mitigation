#!/bin/bash

echo "Starting 5 parallel executions of the Grover Algorithm..."

TOTAL_EXECUTIONS=5
MAX_PARALLEL=1  # Maximum number of parallel jobs

for ((i=1; i<=TOTAL_EXECUTIONS; i++))
do
    echo "Execution #$i: Running grover Algorithm..."
    python3 main.py --algorithm grover \
                    --num_generations 15 \
                    --population_size 50 \
                    --t1 50.0 \
                    --t2 30.0 \
                    --bit_flip_prob 0.02 \
                    --phase_flip_prob 0.02 \
                    > "output_grover_$i.log" 2>&1 &

    # Check the number of background jobs
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL ))
    do
        sleep 1  # Wait for a second before checking again
    done
done

# Wait for all background executions to finish
wait

echo "All parallel executions have completed."

# 250 y 500 gen