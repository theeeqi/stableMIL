# StableMIL: Entropy-Stabilized Attention-based Multiple Instance Learning for Morphologically Variable Whole Slide Images
## Data Preprocess
we follow the CLAM's WSI processing solution (https://github.com/mahmoodlab/CLAM)
## Classification
enter the folder "classification"
```bash
cd classification
```
We assume that you have already extracted WSI features using UNI and stored them in `data_root_dir/UNI`.
``` bash
CUDA_VISIBLE_DEVICES=0 python train.py 
    --data_root_dir ./data \
    --csv_path ./labels/survival_data.csv \
    --split_dir ./splits/5fold \
    --results_dir ./experiments \
    --exp_code stableMIL \
    --aggregate_num 256 \
    --k_neighbors 8 \
    --task subtype \
    --ref_size 512 
```
## Survival Prediction
enter the folder "survival"
```bash
cd survival
```
We assume that you have already extracted WSI features using UNI and stored them in `data_root_dir/UNI`.
``` bash
CUDA_VISIBLE_DEVICES=0 python train_survival.py 
    --data_root_dir ./data \
    --csv_path ./labels/survival_data.csv \
    --split_dir ./splits/5fold \
    --results_dir ./experiments \
    --exp_code stableMIL \
    --aggregate_num 256 \
    --k_neighbors 8 \
    --task subtype \
    --ref_size 512 
```
