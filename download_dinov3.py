import os
from huggingface_hub import snapshot_download
from config import models_dir

def download_model(local_dir):
    
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        local_dir=local_dir
    )

# change model_dir
if __name__ == "__main__":
    model_dir = os.path.join(models_dir, 'dinov3')
    download_model(model_dir)
    print("Download completed")