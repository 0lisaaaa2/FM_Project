import pandas as pd
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
from PIL import Image
import io
import numpy as np


""""
Download dataset from HuggingFace in seems like random folder
DO NOT MOVE THIS FOLDER WHATSOEVER! Its something weird with links
"""
def load_data():
    local_dir = snapshot_download(
        repo_id="mthandazo/chest-xray-pneumonia-images",
        repo_type="dataset"
    )
    print("Data saved to:", local_dir)


"""
Load Data into a PANDAS DataFrame
input: dataname ("test", "train" or "valid")
output: dataframe
"""
def data_to_df(dataname):
    dataset = load_dataset("mthandazo/chest-xray-pneumonia-images") #pulls data from wherever the load_data()-function has put the data
    dataset = dataset[dataname]
    df = dataset.to_pandas()
    print("Data loaded into dataframe: ", dataname, "_df_raw")
    return df


"""
Decode image and convert it to greyscale
input: image_bytes_dict (A dictionary automatically created when to_pandas() is used for HuggingFace-DataFrame)
output: image (A PIL Image object)
"""
def decode_and_to_greyscale(image_bytes_dict):
    img_bytes = image_bytes_dict["bytes"] #bytes is the default name created by to_pandas()
    img = Image.open(io.BytesIO(img_bytes))
    return img.convert("L")


"""
Creates an ID column for the dataframe, because apparently it had none..
input: dataframe (the one you want to ad IDs to), dataname ("test", "train" or "valid", according to the dataframe)
"""
def create_ids(df, dataname):
    df["img_id"] = [f"pneum_{dataname}_img_{i}" for i in range(len(df))]
    print("IDs for", dataname, "_df_raw created")


"""
Prepares the dataframe used to create the final output
input: dataname ("test", "train" or "valid")
output: dataframe
"""
def preprocess_df(dataname):
    df = data_to_df(dataname)
    df["image"] = df["image"].apply(decode_and_to_greyscale)
    print("Images loaded into greyscale")
    create_ids(df, dataname)
    print("Preprocessing of ", dataname, "_df_raw done")
    return df


"""
Creates dictionaries with image-IDs and their corresponding image in greyscale
input: dataframe (the raw one to extract the information from)
output: dictionary
"""
def create_img_dict(df):
    dict = {}
    for idx, row in df.iterrows():
        dict[row["img_id"]] = row["image"]
    print(dict)
    return dict


"""
Create DataFrames for image-IDs and their corresponding labels
input: dataframe (the raw one to extract the information from)
output: dataframe
"""
def create_label_df(df_raw):
    columns = ["img_id", "label"]
    df = pd.DataFrame(columns=columns)
    counter = 0
    for idx, row in df_raw.iterrows():
        df.loc[counter] = [row["img_id"], row["label"]]
        counter += 1
    print(df)
    return df


"""
preprocesses the data and saves it to a folder
input: dataname ("test", "train" or "valid")
"""
def preprocess_and_save_data(path, dataname):
    os.mkdir(path + f"/{dataname}")
    df_raw = preprocess_df(dataname)
    df = create_label_df(df_raw)
    df.to_parquet(f"{path}/{dataname}/metadata.parquet")
    dict = create_img_dict(df_raw)
    np.savez(f"{path}/{dataname}/pictures.npz", **dict)


def main():
    load_data() #run only once, after that delete or set comment
    path = "/home/lisa/PycharmProjects/FM_Project"  # path to folder, where data gets saved, please change

    preprocess_and_save_data(path, "test") # test data
    preprocess_and_save_data(path, "valid") # validation data
    preprocess_and_save_data(path, "train") # trainings data


if __name__ == '__main__':
    main()