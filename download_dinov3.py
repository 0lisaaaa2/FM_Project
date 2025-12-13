import os
from huggingface_hub import snapshot_download
from config import models_dir

def download_model(local_dir, repo_id):
    
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir
    )

# change model_dir
if __name__ == "__main__":
    # repo = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    # model_dir = os.path.join(models_dir, 'dinov3')
    repo = 'facebook/dinov3-vit7b16-pretrain-lvd1689m'
    model_dir = os.path.join(models_dir, 'dinov3_7b')
    download_model(model_dir, repo)
    print("Download completed")