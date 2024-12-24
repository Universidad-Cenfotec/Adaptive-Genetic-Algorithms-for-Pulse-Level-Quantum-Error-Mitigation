#!/bin/bash

echo "Starting 50 parallel executions of the Deutsch-Jozsa Algorithm..."

TOTAL_EXECUTIONS=10
MAX_PARALLEL=3  # Maximum number of parallel jobs

for ((i=1; i<=TOTAL_EXECUTIONS; i++))
do
    echo "Execution #$i: Running Deutsch-Jozsa Algorithm..."
    python3 main.py --algorithm deutsch-jozsa \
                    --num_generations 5 \
                    --population_size 10 \
                    --t1 50.0 \
                    --t2 30.0 \
                    --bit_flip_prob 0.02 \
                    --phase_flip_prob 0.02 \
                    > "output_deutsch_jozsa_$i.log" 2>&1 &

    # Check the number of background jobs
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL ))
    do
        sleep 1  # Wait for a second before checking again
    done
done

# Wait for all background executions to finish
wait

echo "All parallel executions have completed."
