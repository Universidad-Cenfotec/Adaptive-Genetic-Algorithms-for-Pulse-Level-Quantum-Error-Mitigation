#!/usr/bin/env bash

# Ordered scenarios (FASTEST to SLOWEST)
SCENARIOS=(
    "bernstein-vazirani 4 30 15 0.01 0.01"
    "bernstein-vazirani 4 50 30 0.02 0.02"
    "bernstein-vazirani 4 100 60 0.05 0.05"
    "deutsch-jozsa 4 30 15 0.01 0.01"
    "deutsch-jozsa 4 50 30 0.02 0.02"
    "deutsch-jozsa 4 100 60 0.05 0.05"
    "grover 4 30 15 0.01 0.01"
    "grover 4 50 30 0.02 0.02"
    "grover 4 100 60 0.05 0.05"
    "iqft 4 30 15 0.01 0.01"
    "iqft 4 50 30 0.02 0.02"
    "iqft 4 100 60 0.05 0.05"
    "qft 4 30 15 0.01 0.01"
    "qft 4 50 30 0.02 0.02"
    "qft 4 100 60 0.05 0.05"
    "random-universal 4 30 15 0.01 0.01"
    "random-universal 4 50 30 0.02 0.02"
    "random-universal 4 100 60 0.05 0.05"
    "layered-entangling 4 30 15 0.01 0.01"
    "layered-entangling 4 50 30 0.02 0.02"
    "layered-entangling 4 100 60 0.05 0.05"
)

# Experiment parameters
GEN_COUNT=200
POPULATION=100
MAX_RETRIES=2
CPU_THRESHOLD=80  # If CPU usage is above this, wait before starting a new experiment
EMAIL=""

# Log directory setup
LOG_DIR="results"
SESSION_LOG_DIR="$LOG_DIR/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$SESSION_LOG_DIR"

# Function to monitor CPU load before starting an experiment
wait_for_cpu() {
    while [[ $(awk '{print $1}' <(grep 'cpu ' /proc/stat) | awk '{print 100 - ($5 * 100 / ($1+$2+$3+$4+$5+$6+$7+$8))}') > $CPU_THRESHOLD ]]; do
        echo "🚀 High CPU usage detected (>${CPU_THRESHOLD}%), waiting before next experiment..."
        sleep 10
    done
}

# Function to send an email notification
send_email() {
    local SUBJECT="$1"
    local MESSAGE="$2"
    echo -e "$MESSAGE" | mail -s "$SUBJECT" "$EMAIL"
}

# Function to run experiment with retry mechanism
run_experiment() {
    local SCENARIO="$1"
    read -r ALG NUM_QUBITS T1 T2 BIT_FLIP PHASE_FLIP <<< "$SCENARIO"
    local RETRIES=0
    local SUCCESS=0

    while [[ $RETRIES -le $MAX_RETRIES ]]; do
        wait_for_cpu  

        TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
        ALG_DIR="$SESSION_LOG_DIR/$ALG"
        mkdir -p "$ALG_DIR"

        LOGFILE="$ALG_DIR/T1-${T1}_T2-${T2}_BF-${BIT_FLIP}_PF-${PHASE_FLIP}_attempt${RETRIES}_${TIMESTAMP}.log"

        ATTEMPT=$((RETRIES+1))
        echo "[RUNNING] $ALG | T1=$T1, T2=$T2, Bit-Flip=$BIT_FLIP, Phase-Flip=$PHASE_FLIP (Attempt $ATTEMPT)"

        START_TIME=$(date +%s)

        python -m scoop -n 7 main.py \
            --algorithm "$ALG" \
            --num_qubits "$NUM_QUBITS" \
            --num_generations "$GEN_COUNT" \
            --population_size "$POPULATION" \
            --t1 "$T1" \
            --t2 "$T2" \
            --bit_flip_prob "$BIT_FLIP" \
            --phase_flip_prob "$PHASE_FLIP" \
            > "$LOGFILE" 2>&1
        
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "[✅ SUCCESS] $ALG | Time: ${DURATION}s"
            echo "$ALG | Success | Time: ${DURATION}s" >> "$SESSION_LOG_DIR/summary.log"

            # ✅ Send success email
            send_email "✅ Quantum Experiment Completed" "Experiment Completed: $ALG\nT1=$T1, T2=$T2\nBit-Flip=$BIT_FLIP, Phase-Flip=$PHASE_FLIP\nTime Taken: ${DURATION}s\nCheck logs: $LOGFILE"
            SUCCESS=1
            break
        else
            echo "[❌ FAILED] Attempt $ATTEMPT for $ALG"
            ((RETRIES++))
        fi
    done

    if [[ $SUCCESS -eq 0 ]]; then
        echo "[🔥 CRITICAL FAILURE] ❌ $ALG | Skipping..." | tee -a "$SESSION_LOG_DIR/errors.log"

        # ✅ Send failure email
        send_email "⚠️ Quantum Experiment Failed" "Experiment Failed: $ALG\nT1=$T1, T2=$T2\nBit-Flip=$BIT_FLIP, Phase-Flip=$PHASE_FLIP\nCheck logs: $SESSION_LOG_DIR/errors.log"
    fi
}

# Start total execution timer
SCRIPT_START_TIME=$(date +%s)

# Run experiments sequentially
for SCENARIO in "${SCENARIOS[@]}"; do
    run_experiment "$SCENARIO"
done

# Calculate total execution time
SCRIPT_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
echo "✅ All scenarios completed in ${TOTAL_RUNTIME}s."

# Send final summary email
FAILURE_COUNT=$(wc -l < "$SESSION_LOG_DIR/errors.log" 2>/dev/null || echo 0)
send_email "Quantum Experiments Completed" "All experiments finished.\nTotal runtime: ${TOTAL_RUNTIME}s.\nFailures: $FAILURE_COUNT.\nCheck logs: $SESSION_LOG_DIR."

echo "📧 Email sent with summary!"
