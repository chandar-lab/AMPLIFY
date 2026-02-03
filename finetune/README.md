# Evaluation Experiments 

## Directory Structure and Required Files

In the project folder, please ensure you have the following directories and files:

1. **Checkpoints Directory**

   This directory should contain the trained model checkpoints. In particular, the following path must exist:

   ```
   checkpoint/esm2_t33_650M_UR50D_esm2_t30_150M_UR50D_None_1_0.5_0.5_loss_large_1.0_foldseek_gearnet_1_1_0.5_0.5_loss_large_0.8_foldseek_gearnet_3
   ```

2. **Special Model Checkpoints**

   In addition to the general checkpoints, please ensure that you have the following special model directories:
   
   - **ISM Model:**  
     ```
     checkpoint/ISM/ism_model
     ```
   - **ESM-s Model:**  
     ```
     checkpoint/ESM-s/esm_s_model
     ```

3. **Data Directory**

   Create a directory named `saprot_data` to store the necessary huggingface dataset.


## ESM-2

```bash
bash slurm/ESM2_TEST.sh
```

## ISM and ESM-s

```bash
bash slurm/ESM2_O_TEST.sh
``` 

## AMPLIFY

```bash
bash slurm/AMPLIFY_TEST.sh
```





# Training Experiments 

## Reliance (Using 8 GPUs)

- `Our.sh` and `Ablation1.sh` rely on `Reference.sh`.
- `Ablation3.sh` relies on `Ablation3r.sh`.
- `Ablation4.sh` relies on `Ablation4r.sh`.

## Reference Models

- **AMGEN1:** Train `"chandar-lab/AMPLIFY_120M"` on the validation set.
- **AMGEN2:** Train `"facebook/esm2_t30_150M_UR50D"` on the validation set.

Run:
```bash
sbatch slurm/Reference.sh
```


## Proposed Method

- **AMGEN3:** Train `"chandar-lab/AMPLIFY_350M"` using our proposed method.
- **AMGEN4:** Train `"facebook/esm2_t33_650M_UR50D"` using our proposed method.

Run:
```bash
sbatch slurm/Our.sh
```

## Ablation Studies

### Ablation 1: Loss Weight


- Change `loss_weight` from `[1, 0.5, 0.5]` to `[1, 0.0, 0.5]`.
- Change `loss_weight` from `[1, 0.5, 0.5]` to `[1, 0.5, 0.0]`.
- Change `loss_weight` from `[1, 0.5, 0.5]` to `[1, 0.0, 0.0]`.

Run:
```bash
sbatch slurm/Ablation1.sh
```

### Ablation 2: Sample Mode

- Remove the reference model and use `loss_large`.
- Remove the reference model and use `loss_small`.
- Remove the reference model and `ratio=1.0`

Run:
```bash
sbatch slurm/Ablation2.sh
```

### Ablation 3: Structural Token Type

- Change `struc_token_type` from `foldseek` to `protoken`.
- Change `struc_token_type` from `foldseek` to `aido`.

Run:
```bash
sbatch slurm/Ablation3r.sh
sbatch slurm/Ablation3.sh
```

### Ablation 4: Structural Embedding Type

- Change `struc_embed_type` from `gearnet` to `af2`.

Run:
```bash
sbatch slurm/Ablation4r.sh
sbatch slurm/Ablation4.sh
```