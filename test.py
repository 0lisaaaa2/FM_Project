import os
import numpy as np
import pandas as pd

# path to the dataset root (adjust!)
dataset_root = r"D:\lisa-\Universität_2\Master\2. Semester\FM\preprocessed_datasets\bones" 


splits = ["train", "val", "test"]

for split in splits:
    print("=" * 50)
    print(f"Split: {split}")

    split_path = os.path.join(dataset_root, split)
    pictures_path = os.path.join(split_path, "pictures.npz")
    metadata_path = os.path.join(split_path, "metadata.parquet")

    # load data
    data = np.load(pictures_path)
    metadata = pd.read_parquet(metadata_path)

    # total number of images
    num_images_npz = len(data.files)
    num_images_meta = len(metadata)

    print(f"Number of images (pictures.npz): {num_images_npz}")
    print(f"Number of images (metadata):     {num_images_meta}")

    # sanity check
    if num_images_npz != num_images_meta:
        print("⚠ WARNING: mismatch between npz and metadata counts")

    # class distribution
    # change 'label' if your column has a different name
    class_counts = metadata["label"].value_counts().sort_index()

    print("Class distribution:")
    for label, count in class_counts.items():
        print(f"  Class {label}: {count}")

print("=" * 50)
print("Done.")