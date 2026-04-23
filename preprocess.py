from datasets import load_dataset
from PIL import Image
import numpy as np
import os

def preprocess_example(example):
    img = example["image"].convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    example["pixel_values"] = np.array(img)
    return example

if __name__ == "__main__":
    dataset = load_dataset("lucabaggi/animal-wildlife")

    processed_dataset = dataset.map(
        preprocess_example,
        num_proc=os.cpu_count() - 1
    )
    processed_dataset.save_to_disk("processed_animals_256")
