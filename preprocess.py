import kagglehub
from datasets import Dataset
from PIL import Image
import numpy as np
import os
from pathlib import Path

OUTPUT_DIR = "processed_fish_256"

def preprocess_example(example):
    img = example["image"].convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    example["pixel_values"] = np.array(img)
    return example

if __name__ == "__main__":
    # Download from Kaggle
    path = kagglehub.dataset_download("markdaniellampa/fish-dataset")
    print("Path to dataset files:", path)

    # Load images from folder structure into HuggingFace dataset
    image_paths = list(Path(path).rglob("*.jpg")) + \
                  list(Path(path).rglob("*.png")) + \
                  list(Path(path).rglob("*.jpeg"))

    print(f"Found {len(image_paths)} images")

    # Build dataset from image files
    def generate_examples():
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                label = img_path.parent.name  # folder name = class label
                yield {"image": img, "label": label}
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    dataset = Dataset.from_generator(generate_examples)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {set(dataset['label'])}")

    # Preprocess
    processed_dataset = dataset.map(
        preprocess_example,
        num_proc=max(1, os.cpu_count() - 4)
    )

    processed_dataset.save_to_disk(OUTPUT_DIR)

    processed_dataset = processed_dataset.remove_columns(
    [col for col in processed_dataset.column_names if col != "pixel_values"]
    )
    
    processed_dataset.save_to_disk(OUTPUT_DIR)

    print(f"Saved processed dataset to: {OUTPUT_DIR}")
    print(processed_dataset)
    print("Columns:", processed_dataset.column_names)