import os
from huggingface_hub import snapshot_download
from config import model_dir

def download_model(models_dir):

    os.makedirs(models_dir, exist_ok=True)

    snapshot_download(repo_id="google/cxr-foundation",
                    local_dir=models_dir,
                    allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*'])    

if __name__ == "__main__":
    download_model(model_dir)
