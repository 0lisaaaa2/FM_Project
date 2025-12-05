import argparse
import numpy as np
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from config import dataset_dir, models_dir
import logging

logging.basicConfig(level=logging.INFO)

model_dir = os.path.join(models_dir, 'dinov3')
# load model 
processor = AutoImageProcessor.from_pretrained(models_dir)
model = AutoModel.from_pretrained(model_dir)
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
    inputs = processor(images=img, return_tensors="pt")  # handles preprocessing of input: resizing, normalization, convert to tensor -> generates dictionary
    return inputs 


# generates the embedding using dinov3
# data = images arrays loaded from pictures.npz 
def generate_embeddings(data, test=False):
    embeddings = {}
    i = 0
    for img_id in list(data.keys()):  #[:1]: # only first image in split for testing
        arr = data[img_id]

        inputs = preprocess_data(arr).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)  

        emb = outputs.pooler_output

        embeddings[img_id] = emb.cpu().numpy()  # als np.array speichern
        i += 1
        if test and i >= 2:
            break
        #print(f"Shape Embedding for {img_id}: {embeddings[img_id].shape}") #(1, 384)
        #print(f"Embedding for {img_id}: {embeddings[img_id]}")
    
    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate embeddings for dataset splits.")
    parser.add_argument('--datasetdir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--test', action='store_true', help='If set, only test code with 2 datasets')
    args = parser.parse_args()
        
    for split in os.listdir(args.datasetdir):
        logging.info(f"Current data split being processed: {split}")
        # print("Current data split being processed:", split)
        img_path = os.path.join(args.datasetdir, split, "pictures.npz")

        data = load_data(img_path)
        embeddings = generate_embeddings(data, args.test)
        #print(embeddings)
        np.savez_compressed(os.path.join(args.datasetdir, split, "embeddings.npz"), **embeddings)