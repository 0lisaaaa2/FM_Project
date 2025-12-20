import argparse
import numpy as np
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import logging

logging.basicConfig(level=logging.INFO)

TEST_IMAGES_NUM = 42
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
def generate_embeddings(data, test=False, batch_size=16):
    embeddings = {}
    img_ids = list(data.keys())
    num_images = len(img_ids)
    if test:
        img_ids = img_ids[:TEST_IMAGES_NUM]
        num_images = TEST_IMAGES_NUM

    num_batches = (num_images + batch_size - 1) // batch_size
    logging.info(f"Processing {num_images} images in {num_batches} batches (batch size: {batch_size})")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_img_ids = img_ids[batch_start:batch_start + batch_size]
        batch_imgs = [
            Image.fromarray(data[img_id].astype(np.uint8), mode="L") for img_id in batch_img_ids
        ]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        logging.info(f"Batch {batch_idx+1}/{num_batches}")
        logging.debug("Actual Batch size: %s, img_format %s, %s, %s", *inputs.pixel_values.shape)
        with torch.inference_mode():
            outputs = model(**inputs)
        batch_embs = outputs.pooler_output
        logging.debug("Output size: %s", batch_embs.shape)
        for idx, img_id in enumerate(batch_img_ids):
            embeddings[img_id] = batch_embs[idx].cpu().numpy()
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate embeddings for dataset splits.")
    parser.add_argument('--datasetdir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--modeldir', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--test', action='store_true', help='If set, only test code with 2 datasets')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing images', default=16)
    args = parser.parse_args()
    
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    global model, processor, device    
    # load model 
    processor = AutoImageProcessor.from_pretrained(args.modeldir)
    model = AutoModel.from_pretrained(args.modeldir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for split in os.listdir(args.datasetdir):
        logging.info(f"Current data split being processed: {split}")
        # print("Current data split being processed:", split)
        img_path = os.path.join(args.datasetdir, split, "pictures.npz")

        data = load_data(img_path)
        embeddings = generate_embeddings(data, args.test, args.batch_size)
        #print(embeddings)
        np.savez_compressed(os.path.join(args.datasetdir, split, "dinov3_embeddings.npz"), **embeddings)