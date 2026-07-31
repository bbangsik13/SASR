#!/bin/bash

(
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate gs

  RHO_XYZ=1e0
  RHO_ROTATION=5e-4 
  RHO_SCALING=2e-1 
  RHO_OPACITY=1e-5
  RHO_F_DC=1e-1
  RHO_F_REST=1e-2
  MIN_KERNEL=1 
  MAX_GAMMA=0.95 
  MAX_KERNEL=15 
  THRESHOLD=0.0005

  if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
  else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^$' | wc -l)
  fi

  declare -a GPU_PIDS
  for i in {00..09}; do
    for DATA in data/nerf_llff_data/*; do
      LOGS=logs/llff
      OUTPUT=output/llff
      SCENE=$(basename $DATA)
      EXP=${SCENE}_${i}

      if [ -f $OUTPUT/$EXP/results.json ]; then
        continue
      fi
      mkdir -p $LOGS

      while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
          if [ -z "${GPU_PIDS[$gpu]}" ] || ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then

            CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "
              python -u train.py \
                -s $DATA -m $OUTPUT/$EXP \
                --random_background \
                --eval -r 8 --n_views 3 --port $((gpu+6223)) \
                --iterations 10000 --position_lr_max_steps 10000 \
                --densify_until_iter 10000 \
                --densify_grad_threshold $THRESHOLD \
                --minimizer SASR --max_kernel $MAX_KERNEL --min_kernel $MIN_KERNEL --gamma $MAX_GAMMA \
                --minimizer_from_iter 3000 \
                --rho_xyz $RHO_XYZ --rho_rotation $RHO_ROTATION --rho_scaling $RHO_SCALING \
                --rho_opacity $RHO_OPACITY --rho_f_dc $RHO_F_DC --rho_f_rest $RHO_F_REST && \
              python render.py -s $DATA -m $OUTPUT/$EXP && \
              python metrics.py -m $OUTPUT/$EXP
            " > $LOGS/${EXP}.out 2>&1 &

            GPU_PIDS[$gpu]=$!   
            break 2             
          fi
        done
        sleep 5
      done
    done
  done


  for pid in "${GPU_PIDS[@]}"; do
    if [ -n "$pid" ]; then
      wait $pid
    fi
  done
) &

echo $! > run_subshell.pid
