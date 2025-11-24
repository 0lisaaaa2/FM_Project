import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

"""
Load dataset from Kaggle
"""
#import kagglehub
#path = kagglehub.dataset_download("harmeetsingh13/foreign-body-hernia-detection-in-animals")
#print("Path to dataset files:", path) # C:\Users\lisa-\.cache\kagglehub\datasets\harmeetsingh13\foreign-body-hernia-detection-in-animals\versions\1

"""
Read files and assign labels

data_root: root directory containing (train|validation|test)/(foreign_body_hernia|normal)

returns a dataframe with columns: "file_path" and "label"
"""
def load_images(data_root):
    rows = []

    # iterate through split folder (want to redo split later, so put everything in one df for now)
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(data_root, split)
        if not os.path.isdir(split_path):
            continue
    
        # read through each folder in the root directory
        for folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(split_path):
                continue

            # label assignment based on folder name
            folder_lower = folder.lower()
            if "foreign" in folder_lower:
                label = "foreignbody"
            elif "normal" in folder_lower:
                label = "normal"
            else:
                print(f"Skipping unknown folder: {folder}")
                continue
            
            # read image files
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    rows.append({
                        "file_path": os.path.join(folder_path, fname), # save file path for later steps
                        "label": label
                    })

    df = pd.DataFrame(rows) # dataframe with "file_path" and "label"
    print("Loaded", len(df), "images total.")
    return df

"""
Split data into train, validation, and test sets (70/15/15)

df: dataframe containing file_path and label columns

retunrns three dataframes: train_df, val_df, test_df according to the split
"""

def split_data(df):
    train_list, val_list, test_list = [], [], []

    # split data for each label separately to maintain class distribution
    for label in df["label"].unique():
        sub_df = df[df["label"] == label]
        train, temp = train_test_split(sub_df, test_size=0.30, shuffle=True, random_state=42) # 70% train, 30% temp
        val, test = train_test_split(temp, test_size=0.50, shuffle=True, random_state=42) # 15% val, 15% test

        train_list.append(train) # [train_foreignbody, train_normal] 
        val_list.append(val)
        test_list.append(test)

    # recombine classes into one df each
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    return train_df, val_df, test_df

""" 
Add images ids animal_test_img_1 etc.

df: input dataframe
split_name: "train" "test" or "val" 

returns dataframe with new image ids
"""
def add_image_ids(df, split_name):
    df = df.copy()
    df["img_id"] = [f"animal_{split_name}_img_{i+1}" for i, row in df.iterrows()]
    return df

"""
Convert image to greyscale

path: image path

returns images as greyscale array
"""
def load_and_convert_image(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)

"""
Save metadata.parquet and pictures.npz

df: input dataframe (train_df, val_df, test_df)
output_dir : output directory
split_name: "train", "val", "test"
"""
def save_split(df, output_dir, split_name):
    # create output folder for split if not already exsiting
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # save metadata.parquet
    metadata_df = df[["img_id", "label"]]
    metadata_path = os.path.join(split_dir, "metadata.parquet")
    metadata_df.to_parquet(metadata_path)
    print(f"Saved metadata: {metadata_path}")

    # load all images and create dict
    img_dict = {}
    for _, row in df.iterrows():
        img_array = load_and_convert_image(row.file_path) # convert into greyscale
        img_dict[row.img_id] = img_array # img_id : image_array

    # save pictures.npz
    pictures_path = os.path.join(split_dir, "pictures.npz")
    np.savez(pictures_path, **img_dict)
    print(f"Saved pictures: {pictures_path}")

"""
Main
"""
def main():
    data_root = r"C:\Users\lisa-\.cache\kagglehub\datasets\harmeetsingh13\foreign-body-hernia-detection-in-animals\versions\1\DataSet Hernia Detection"
    output_dir = r"D:\lisa-\Universität_2\Master\2. Semester\FM\preprocessed_datasets\animals"

    df = load_images(data_root)

    train_df, val_df, test_df = split_data(df)

    train_df = add_image_ids(train_df, "train")
    val_df = add_image_ids(val_df, "val")
    test_df = add_image_ids(test_df, "test")

    save_split(train_df, output_dir, "train")
    save_split(val_df, output_dir, "val")
    save_split(test_df, output_dir, "test")

    print("\nAll datasets saved successfully!")

#Loaded 514 images total.
#Train size: 358, Validation size: 78, Test size: 78
if __name__ == "__main__":
    main()