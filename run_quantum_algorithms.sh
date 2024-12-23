TOTAL_EJECUCIONES=5

ejecutar_deutsch_jozsa() {
    local i=$1
    echo "Ejecución #$i: Ejecutando Algoritmo Deutsch-Jozsa..."
    python3 main.py --algorithm deutsch-jozsa \
                    --num_generations 50 \
                    --population_size 100 \
                    --t1 50.0 \
                    --t2 30.0 \
                    --bit_flip_prob 0.02 \
                    --phase_flip_prob 0.02 \
                    > "salida_deutsch_jozsa_$i.log" 2>&1
}

export -f ejecutar_deutsch_jozsa
