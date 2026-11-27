# SpMV and SpMM on GPUs (Group 10)

CUDA kernels for sparse matrix-vector (SpMV) and sparse matrix-dense matrix (SpMM, k = 32) multiplication, compared with a CPU baseline and cuSPARSE.

Kernels in `kernels/main.cu`: CSR scalar, CSR vector and ELL SpMV; naive and coalesced CSR SpMM; cuSPARSE SpMV/SpMM.

## Build

Needs the CUDA toolkit (nvcc + cuSPARSE).

```bash
make                                # default: -O3 -arch=sm_80 (A30 / A100)
make FLAGS="-O3 -arch=sm_75"        # e.g. T4 on Colab
```

On the college cluster, build on the login node:

```bash
module load cuda/11.3
make FLAGS="-O3 -arch=sm_80 -ccbin /usr/bin/g++"
```

## Run

One matrix:

```bash
./bin/bench data/pwtk.mtx
```

All matrices on the cluster (SLURM, 1 GPU):

```bash
sbatch run.sh                       # output: results/results_<jobid>.txt
```

For each kernel it prints the average time (1 warm-up + 50 runs), GFLOP/s and PASS/FAIL against the CPU result.

## Results and graphs


Final A30 results: `results/results_16918.txt`, `results/results.csv` and the plots in `results/`.
Report: `report/SpMV_SpMM_Report.pdf`.

## Layout

| Path | Contents |
|---|---|
| `src/` | COO / CSR / ELL structs, .mtx reader, format conversion |
| `csrToEll/` | CPU reference SpMV/SpMM, `checkResult` |
| `kernels/main.cu` | GPU kernels, cuSPARSE calls, timing, checks |
| `data/` | Test matrices (sources in `data/README.md`) |
| `run.sh` | SLURM job script |
| `scripts/` | Output parser and plotting |
| `results/` | Raw outputs, CSV, graphs |
| `report/` | Project report |
