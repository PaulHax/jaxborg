#!/bin/bash
#SBATCH --job-name=JAXBorg-parity-retest        # Job name
#SBATCH --account=itm
#SBATCH --output=output.txt           # Standard output file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00                # Maximum runtime (D-HH:MM:SS)

cd /home/local/KHQ/dena.mujtaba/jaxborg


uv run python scripts/dev/parity_gate.py \
  --train \
  --recipe default \
  --seeds 42,100,200 \
  --train-launcher local \
  --parallel-train 2 \
  --eval-episodes 100 \
  --eval-workers 48 \
  --tost-margin 284 \
  --run-fast-tests 