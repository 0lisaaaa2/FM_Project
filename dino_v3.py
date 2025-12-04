import numpy as np
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import logging

logging.basicConfig(level=logging.INFO)

# path to folder containing all datasets
data_path = r"D:\lisa-\Universität_2\Master\2. Semester\FM\preprocessed_datasets" #\datasetname\(test/train/valid)\pictures.npz
# path to dinov3 (uploaded to drive as well)
model_path = r"D:\lisa-\Universität_2\Master\2. Semester\FM\dinov3"
# if not locally, instead load:
# model_path = "facebook/dinov3-vits16-pretrain-lvd1689m"

# load model 
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# load data 
# path = path to pictures.npz
def load_data(path):
    data = np.load(path)
    return data


# preprocesses the data using dino processor 
# arr = numpy array containing image values
def preprocess_data(arr):
    img = Image.fromarray(arr.astype(np.uint8), mode="L") # convert into PIL image
    inputs = processor(images=img, return_tensors="pt")  # handles preprocessing of input: resizing, normalization, convert to tensor -> generates dictrionary
    return inputs 


# generates the embedding using dinov3
# data = images arrays loaded from pictures.npz 
def generate_embeddings(data):
    embeddings = {}

    for img_id in list(data.keys()):  #[:1]: # only first image in split for testing
        arr = data[img_id]

        inputs = preprocess_data(arr).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)  

        emb = outputs.pooler_output

        embeddings[img_id] = emb.cpu().numpy()  # als np.array speichern
        
        #print(f"Shape Embedding for {img_id}: {embeddings[img_id].shape}") #(1, 384)
        #print(f"Embedding for {img_id}: {embeddings[img_id]}")
    
    return embeddings


if __name__ == "__main__":
    for folder_name in os.listdir(data_path):
        #print("Dataset being processed:", folder_name)
        logging.info(f"Dataset being processed: {folder_name}")
        dataset_path = os.path.join(data_path, folder_name)
        
        for split in os.listdir(dataset_path):
            logging.info(f"Current data split being processed: {split}")
            #print("Current data split being processed:", split)
            img_path = os.path.join(dataset_path, split, "pictures.npz")

            data = load_data(img_path)
            embeddings = generate_embeddings(data)
            #print(embeddings)
            np.savez_compressed(os.path.join(dataset_path, split, "embeddings.npz"), **embeddings)