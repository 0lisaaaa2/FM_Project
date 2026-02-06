import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

# load DINOv2
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval()

# image transfromation to fit dinos expected RGB input
transform = T.Compose([
    #T.Grayscale(num_output_channels=3),  # dino expects RGB -> need 3 channels
    T.ToTensor(), # convert to floattensor + normalize to [0,1]
    T.Resize(244), # resize image (dino expect 224 x 224)
    T.CenterCrop(224), 
    T.Normalize([0.5], [0.5]) # convert to [-1, 1]
    ])

# convert np.array using steps from above ()
def preprocess_array(arr):
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    return transform(img).unsqueeze(0)

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
    x = preprocess_array(arr).to(device)

    with torch.no_grad():
        emb = model(x)  
    embeddings[img_id] = emb.cpu().numpy()  # als np.array speichern
    print(f"Shape Embedding for {img_id}: {embeddings[img_id].shape}") #(1, 384)
    print(f"Embedding for {img_id}: {embeddings[img_id]}")

# save embeddings
#output_path = "embeddings.npz"
#np.savez(output_path, **embeddings)
#print(f"Embeddings saved to {output_path}")