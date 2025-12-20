import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # just use CPU for TF classifier
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
class EmbeddingClassifierTF:
    def __init__(self, dataset_path, emb_file, normal_label, epochs=10, batch_size=32):
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path
        self.emb_file = emb_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.normal_label = normal_label
        self.cxr_embedding = True if emb_file == 'embeddings.npz' else False
        # paths
        self.train_path = os.path.join(dataset_path, "train")
        self.test_path = os.path.join(dataset_path, "test")
        if os.path.exists(os.path.join(dataset_path, "valid")):
            self.valid_path = os.path.join(dataset_path, "valid")
        elif os.path.exists(os.path.join(dataset_path, "val")):
            self.valid_path = os.path.join(dataset_path, "val")
        else:
            raise ValueError("Validation directory not found in dataset path.")
        self.label_to_int = {}

    @staticmethod
    def evaluate_binary_classifier(y_true, y_pred_probs):
        y_pred = (y_pred_probs >= 0.5).astype(int)
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        return results

    def load_split(self, split_path):
        meta_path = os.path.join(split_path, "metadata.parquet") 
        meta = pd.read_parquet(meta_path)
        # embeddings path
        emb = np.load(os.path.join(split_path, self.emb_file), allow_pickle=True)
        emb_dict = dict(emb)
        
        img_ids = emb_dict.keys()
        X = np.stack([emb_dict[img_id] for img_id in img_ids])
        # Remove extra dimensions if present
        X = X.reshape(X.shape[0], -1)
        unique_labels = sorted(meta['label'].unique())
        if not self.label_to_int:
            self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"Label to integer mapping: {self.label_to_int}")  
        meta['label_bin'] = meta['label'].map(self.label_to_int)
        y = meta.set_index("img_id").loc[img_ids]["label_bin"].values
        return X, y

    def build_tf_classifier(self, input_dim):
        # if cxr embedding add reshape layer
        layers = []

        layers.append(tf.keras.layers.Input(shape=(input_dim,)))
        layers.append(tf.keras.layers.Dense(128, activation='relu'))
        layers.append(tf.keras.layers.Dense(64, activation='relu'))
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model = tf.keras.Sequential(layers)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def run(self):
        # Load data
        X_train, y_train = self.load_split(self.train_path)
        X_val, y_val = self.load_split(self.valid_path)
        X_test, y_test = self.load_split(self.test_path)
        input_dim = X_train.shape[1]

        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")
        # Train model
        model = self.build_tf_classifier(input_dim)
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), verbose=1)

        # Evaluate on train and test
        for split_name, X, y in [("train", X_train, y_train), ("validation", X_val, y_val),("test", X_test, y_test)]:
            probs = model.predict(X, batch_size=self.batch_size).flatten()
            results = self.evaluate_binary_classifier(y, probs)
            print(f"{split_name} results: {results}")

        # # Save predictions for test split
        # probs = model.predict(X_test, batch_size=self.batch_size).flatten()
        # preds = (probs >= 0.5).astype(int)
        # pred_col = f"pred_{self.emb_file.replace('.npz','')}"
        # meta_test.loc[meta_test["img_id"].isin(img_ids_test), pred_col] = preds
        # meta_test.to_parquet(self.test_meta_path)
        # print(f"Saved predictions to {self.test_meta_path} in column '{pred_col}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate embeddings for a single dataset using TensorFlow.")
    parser.add_argument('--datasetdir', type=str, required=True, help="Path to a single dataset directory (e.g. datasets/animals)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--normal_label', type=str, default='normal', help="Batch size")
    args = parser.parse_args()
    print(tf.config.list_physical_devices('GPU'))
    embedding_files = [ 'embeddings.npz', 'dinov3_embeddings.npz']

    for em in embedding_files:
        print(f"Processing embeddings from file: {em}")
        
        clf = EmbeddingClassifierTF(
            dataset_path=args.datasetdir,
            emb_file=em,
            normal_label=args.normal_label,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        clf.run()
        print("-"*50)

    results_df = pd.DataFrame(all_results)

    print("\nResults table:")
    print(results_df)
    results_df.to_csv(args.output_file, index=False)
    print(f"\nSaved results table to {args.output_file}")
    