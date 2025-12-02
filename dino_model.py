import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# load dinov3 from huggingface
model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# convert np.array to tensor
def preprocess_array(arr):
    img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB") # greyscale input as rgb
    inputs = processor(images=img, return_tensor="pt")  # handles preprocessing of input: resizing, normalization, convert to tensor -> generates dictrionary
    return inputs

# load data
pictures_path = r"D:\lisa-\Universität_2\Master\2. Semester\FM\animals\test\pictures.npz"          
metadata_path = r"D:\lisa-\Universität_2\Master\2. Semester\FM\animals\test\metadata.parquet"      

data = np.load(pictures_path)
metadata = pd.read_parquet(metadata_path)

# generate embeddings
embeddings = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for idx, row in metadata.head(3).iterrows(): # only use first 3 images for now
    img_id = row["img_id"]
    arr = data[img_id]

    inputs = preprocess_array(arr).to(device)

    with torch.inference_mode:
        outputs = model(**inputs)  

    emb = outputs.pooler_output

    embeddings[img_id] = emb.cpu().numpy()  # als np.array speichern
    
    print(f"Shape Embedding for {img_id}: {embeddings[img_id].shape}")
    print(f"Embedding for {img_id}: {embeddings[img_id]}")

# save embeddings
#output_path = "embeddings.npz"
#np.savez(output_path, **embeddings)
#print(f"Embeddings saved to {output_path}")