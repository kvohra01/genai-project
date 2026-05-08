# GenAI Diffusion Project

This project trains and evaluates a diffusion model on image datasets (fish and animals) using Jupyter notebooks.

## Project Structure

- `preprocess.py` - downloads and preprocesses dataset images into `pixel_values`.
- `model.ipynb` - main notebook for training and experiments.
- `processed_fish_256/` - saved preprocessed fish dataset (created by preprocessing).
- `ema_model.pt` - example saved model weights for inference/experiments.


```

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install datasets kagglehub pillow numpy pandas tqdm matplotlib jupyter ipykernel
```

If you are on CPU-only, install the CPU PyTorch wheel instead of CUDA.

## 1) Preprocess Data (Run First)

From the project root, run:

```bash
python preprocess.py
```

This creates/updates `processed_fish_256/` with preprocessed tensors used by the notebooks.

## 2) Open Notebook

```bash
jupyter notebook
```

Open `model.ipynb`.

## 3) Two Ways to Use `model.ipynb`

### Path A: Train Again (from scratch)

1. In `model.ipynb`, run cells from the first training section top-to-bottom.
2. This rebuilds dataset tensors, initializes the model, and trains.
3. Training outputs include generated samples and saved checkpoints/weights.

### Path B: Load Existing Weights and Run Experiments

1. In `model.ipynb`, go to the experiment/inference section.
2. Load weights (for example `ema_model.pt`).
3. Run the experiment cells for:
   - image generation
   - image interpolation on dataset images
   - image interpolation on generated images

This path is faster when you want results without retraining.

*Note:* In the rerun section images are saved with the new epoch number instead of the actual epoch number so the last 2000 epochs are numbered 1-2000 instead of 3000-5000. They have been renamed correctly in the final_train_samples folder

## 4) Optional: Hyperparameter Tuning

Use `tuning.ipynb` to test different learning rates, batch sizes, and related settings.

## Notes

- Make sure `DATA_DIR` in notebooks points to an existing processed dataset folder (for example `processed_fish_256`).

