import argparse
import logging
import time
from config import models_dir
import os
import io
import png
import tensorflow as tf
import tensorflow_text
import numpy as np

logging.basicConfig(level=logging.INFO)

logging.info("GPUs: %s", tf.config.list_physical_devices('GPU'))
logging.info("Num GPUs Available: %s", len(tf.config.list_physical_devices('GPU')))

if 'elixrc_model' not in locals():
    elixrc_model = tf.saved_model.load(os.path.join(models_dir,'elixr-c-v2-pooled'))
    if elixrc_model is None:
        raise ValueError("Failed to load ELIXR-C model.")
    elixrc_infer = elixrc_model.signatures['serving_default']

if 'qformer_model' not in locals():
    qformer_model = tf.saved_model.load(os.path.join(models_dir,'pax-elixr-b-text'))
    if qformer_model is None:
        raise ValueError("Failed to load ELIXR-C model.")
    qformer_infer = qformer_model.signatures['serving_default']
    
# Helper function for processing image data
def load_dataset(dataset_path):
    """load a npz dataset from the given path."""
    return np.load(dataset_path, allow_pickle=True)
    

def array_to_tfexample(image_array: np.ndarray) -> tf.train.Example:
    """Creates a tf.train.Example from a NumPy array."""
    # Convert the image to float32 and shift the minimum value to zero
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        # For uint8 images, no rescaling is needed
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        # For other data types, scale image to use the full 16-bit range
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val  # Scale to 16-bit range
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    # Ensure the array is 2-D (grayscale image)
    if pixel_array.ndim != 2:
        raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

    # Encode the array as a PNG image
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create a tf.train.Example and assign the features
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example


def calculate_embeddings(data, test=False):
    embeddings_dict = {}
    i = 0
    for key, value in data.items():
        
        logging.debug(f'Processing {key}...')
        serialized_image = array_to_tfexample(value).SerializeToString()
        embeddings = process_image(serialized_image)
        embeddings_dict[key] = embeddings
        logging.debug(f'Embeddings shape for {key}: {embeddings[0].shape}')
        i += 1
        if test and i >= 2:
            break
    return embeddings_dict
    
def process_image(serialized_image):

    # ELIXR-C inference
    elixrc_output = elixrc_infer(input_example=tf.constant([serialized_image]))
    elixrc_embeddings = elixrc_output['feature_maps_0'].numpy()

    # QFormer inference
    elixrb_embeddings = []
    qformer_input = {
        'image_feature': elixrc_embeddings.tolist(),
        'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
        'paddings': np.zeros((1, 1, 128), dtype=np.float32).tolist(),
    }
    output = qformer_infer(**qformer_input)
    elixrb_embeddings.append(output['all_contrastive_img_emb'].numpy())
    return elixrb_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate embeddings for dataset splits.")
    parser.add_argument('--datasetdir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--test', action='store_true', help='If set, only test code with 2 datasets')
    args = parser.parse_args()
    

    for split in os.listdir(args.datasetdir):
        split_path = os.path.join(args.datasetdir, f'{split}/')
        logging.info(f"Processing split: {split_path}")
        data = load_dataset(os.path.join(split_path,'pictures.npz'))
        start_time = time.time()
        embeddings = calculate_embeddings(data, args.test)

        np.savez_compressed(os.path.join(split_path, 'embeddings.npz'), **embeddings)
        elapsed = time.time() - start_time
        logging.info(f"Processing time for {split}: {elapsed:.2f} seconds")
