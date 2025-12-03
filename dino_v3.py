import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

data_path = r"D:\lisa-\Universität_2\Master\2. Semester\FM\preprocessed_datasets\animals\test\pictures.npz"
model_path = r"D:\lisa-\Universität_2\Master\2. Semester\FM\dinov3"

processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


def load_data(path):
    data = np.load(path)
    return data

def proprocess_data(arr):
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    inputs = processor(images=img, return_tensors="pt")  # handles preprocessing of input: resizing, normalization, convert to tensor -> generates dictrionary
    return inputs 

def generate_embeddings(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = {}

    for img_id in list(data.keys())[:3]: # only first 3 images for now
    arr = data[img_id]

    inputs = preprocess_array(arr).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)  

    emb = outputs.pooler_output

    embeddings[img_id] = emb.cpu().numpy()  # als np.array speichern
    
    print(f"Shape Embedding for {img_id}: {embeddings[img_id].shape}") #(1, 384)
    print(f"Embedding for {img_id}: {embeddings[img_id]}")




if __name__ == "__main__":
    data = load_data(data_path)
    generate_embeddings(data)

