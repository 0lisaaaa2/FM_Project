import pandas as pd
import os
import numpy as np
from PIL import Image
import io


"""
Gets the JPEG version of an image in greyscale
input: img_path (path of the image)
output: bytes of the JPEG image
"""
def pil_to_jpeg_bytes(img_path):
    img = Image.open(img_path).convert("L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.getvalue()


"""
Creates a DataFrame containing everything needed to get test/train/valid datasets
input: path (path of the data folder)
output: DataFrame containing all needed data
"""
def to_df_raw(path):
    # creates a DataFrame of the metadata
    df_meta = pd.read_csv(f"{path}/meta.txt", sep='\t')
    df_meta.drop(columns=["Unnamed: 0", "basename", "class", "instance", "pose", "scan_tray", "IN_id", "WN_id"], inplace=True) # keeping only "dangorous"
    df_meta["dangerous"] = df_meta["dangerous"].fillna(False).astype(bool) #Set NaN Entries False, bc container is empty
    df_meta.rename(columns={"dangerous": "label"}, inplace=True)
    print(df_meta)

    # creates a DataFrame of the image data
    files = [f for f in os.listdir(f"{path}/Low") if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    df = pd.DataFrame({
        "path": [os.path.join(f"{path}/Low", f) for f in files]
    })
    df["img"] = df["path"].apply(pil_to_jpeg_bytes)
    df.drop(columns=["path"], inplace=True) # keeping only "img"
    print(df)

    # combining meta_df and df
    df_combined = pd.concat([df_meta, df], axis=1)
    print(df_combined)
    df_combined = df_combined.sample(frac=1, axis=0).reset_index(drop=True) # mixing df, bc there is several pics with the same items next to each other
    print(df_combined)
    return df_combined


"""
Creates an ID column for the dataframe, because apparently it had none..
input: dataframe (the one you want to ad IDs to), dataname ("test", "train" or "valid", according to the dataframe)
"""
def create_ids(df, dataname):
    df["img_id"] = [f"baggage_{dataname}_img_{i}" for i in range(len(df))]
    print("IDs for", dataname, "_df_raw created")


"""
Splits the dataframe to 70% trainings, 15% test and 15% validation data
input: dataframe to be split
output: trainings, test and validation dataframes
"""
def split_train_test_valid(df):
    train_df = df.iloc[:-580]
    test_df = df.iloc[-580:-290]
    valid_df = df.iloc[-290:]
    return train_df, test_df, valid_df


"""
Split the dataframe into label with id and image with id
input: dataframe to be split
output: label_df, img_df
"""
def split_label_image(df):
    label_df = df[["img_id", "label"]]
    img_df = df[["img_id", "img"]]
    return label_df, img_df


"""
Creates dictionaries with image-IDs and their corresponding image in greyscale
input: dataframe (the raw one to extract the information from)
output: dictionary
"""
def create_img_dict(df):
    dict = {}
    for idx, row in df.iterrows():
        dict[row["img_id"]] = row["img"]
    return dict


"""
preprocesses the data and saves it to a folder
input: dataname ("test", "train" or "valid")
"""
def preprocess_and_save_data(path, df, dataname):
    os.mkdir(path + f"/{dataname}")
    create_ids(df, dataname)
    label_df = df[[f"img_id", f"label"]]
    label_df.to_parquet(f"{path}/{dataname}/metadata.parquet")
    dict = create_img_dict(df)
    np.savez(f"{path}/{dataname}/pictures.npz", **dict)


def main():
    load_path = "/home/lisa/PycharmProjects/FM_Project/data" #Please enter where you saved data after download
    save_path = "/home/lisa/PycharmProjects/FM_Project" #Please enter where you want to save data after preprocessing

    to_df_raw(load_path)
    train_df, test_df, valid_df = split_train_test_valid(to_df_raw(load_path))
    preprocess_and_save_data(save_path, train_df, dataname="train")
    preprocess_and_save_data(save_path, test_df, dataname="test")
    preprocess_and_save_data(save_path, valid_df, dataname="valid")



if __name__ == '__main__':
    main()