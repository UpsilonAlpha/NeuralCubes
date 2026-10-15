# Training on a cluster

`train.sbatch` runs `train_nca.py` on a SLURM cluster with one GPU per job, as a job
array with one task per target.

1. Copy the repository to the cluster and create an environment once:

   ```sh
   python -m venv ~/venvs/nca
   ~/venvs/nca/bin/pip install -r training/requirements.txt "jax[cuda12]"
   ```

   Check that JAX sees the GPU: `~/venvs/nca/bin/python -c "import jax; print(jax.devices())"`.

2. Edit the `#SBATCH` lines for your cluster (partition, account, time limit), then submit
   from the repository root:

   ```sh
   sbatch --array=0-4 training/cluster/train.sbatch          # bunny tree mushroom cactus torus
   TARGETS="tree" ROTATION=full sbatch --array=0 training/cluster/train.sbatch
   ```

   Settings come from environment variables: `TARGETS`, `SIZE` (default 32),
   `ROTATION` (`none`, `yaw`, `full`; default `yaw`), `ITERATIONS` (default 8000),
   `OUT` (default `runs/`), `VENV`.

3. Weights land in `runs/<target>.json` every 200 iterations, so you can load a
   half-trained model in the viewer while the job runs. Copy finished ones to
   `web/models/` to ship them with the site.

Jobs checkpoint to `runs/<target>.ckpt` (weights, optimizer state and the sample pool).
Resubmitting the same command resumes from it; with `--requeue` a preempted job does that
on its own. Delete the checkpoint to start over.

The script's defaults (32³ grid, hidden 128, batch 8, 48 to 96 steps) are sized for one
modern GPU. Add `--remat` to the `python` line if the GPU runs out of memory on bigger
grids. Without SLURM, run the same `python training/train_nca.py ...` line directly.
