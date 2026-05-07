from datasets import load_dataset, concatenate_datasets
from PIL import Image
import numpy as np
import os

OUTPUT_DIR = "processed_fish_256"

def preprocess_example(example):
    img = example["image"].convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)

    # Adds image tensor data but keeps original columns like label
    example["pixel_values"] = np.array(img)

    return example

if __name__ == "__main__":
    dataset = load_dataset("markdaniellampa/fish-dataset")

    # Combine all splits into one dataset for diffusion training
    if isinstance(dataset, dict):
        split_names = list(dataset.keys())
        print("Found splits:", split_names)

        if len(split_names) == 1:
            dataset = dataset[split_names[0]]
        else:
            dataset = concatenate_datasets([dataset[split] for split in split_names])

    processed_dataset = dataset.map(
        preprocess_example,
        num_proc=max(1, os.cpu_count() - 1)
    )

    processed_dataset.save_to_disk(OUTPUT_DIR)

    print(f"Saved processed dataset to: {OUTPUT_DIR}")
    print(processed_dataset)
    print("Columns:", processed_dataset.column_names)