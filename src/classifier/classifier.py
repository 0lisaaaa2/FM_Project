import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # just use CPU for TF classifier
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

NORMAL_LABELS = {'normal', 'NORMAL', 'np.False_', np.False_}

class EmbeddingClassifierTF:
    def __init__(self, dataset_path, emb_file, results_file, epochs=10, batch_size=32):
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path
        self.dataset_name = os.path.basename(dataset_path.rstrip('/\\'))
        self.classifier_name = 'cxr' if emb_file == 'embeddings.npz' else 'dinov3'
        self.emb_file = emb_file
        self.results_file = results_file
        self.epochs = epochs
        self.batch_size = batch_size
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
            "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
            "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
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

        if not self.label_to_int:
            unique_labels = sorted(meta['label'].unique())
            assert len(unique_labels) == 2, "Expected exactly two unique labels for binary classification."
            self.label_to_int = { label: (0 if label in NORMAL_LABELS else 1) for label in unique_labels}
            print(f"Label to integer mapping: {self.label_to_int}")
            assert len(self.label_to_int) == 2, "No normal label found in the dataset."
        
        meta['label_bin'] = meta['label'].map(self.label_to_int)
        y = meta.set_index("img_id").loc[img_ids]["label_bin"].values
        return X, y

    def build_tf_classifier(self, input_dim):
        # if cxr embedding add reshape layer
        layers = []

        layers.append(tf.keras.layers.Input(shape=(input_dim,)))
        layers.append(tf.keras.layers.Dense(512, activation='relu'))
        layers.append(tf.keras.layers.Dense(256, activation='relu'))
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

        results = {}
        # Evaluate on train and test
        for split_name, X, y in [("train", X_train, y_train), ("validation", X_val, y_val),("test", X_test, y_test)]:
            probs = model.predict(X, batch_size=self.batch_size).flatten()
            results = self.evaluate_binary_classifier(y, probs)
            print(f"{split_name} results: {results}")
            self.write_file(results, split_name)


    def write_file(self, results, split_name):
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as f:
                f.write("dataset,classifier,split,tn,tp,fn,fp,accuracy,precision,recall,f1\n")
            
        with open(self.results_file, 'a') as f:
            f.write(
                (
                    f"{self.dataset_name},{self.classifier_name},{split_name},"
                    f"{results['tn']},{results['tp']},{results['fn']},{results['fp']},"
                    f"{results['accuracy']},{results['precision']},{results['recall']},{results['f1']}\n"
                )
            )

        
        
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
    args = parser.parse_args()
    print(tf.config.list_physical_devices('GPU'))
    embedding_files = [ 'embeddings.npz', 'dinov3_embeddings.npz']

    for em in embedding_files:
        print(f"Processing embeddings from file: {em}")
        
        clf = EmbeddingClassifierTF(
            dataset_path=args.datasetdir,
            emb_file=em,
            results_file= os.path.dirname(os.path.normpath(args.datasetdir)) + '/classifier_results.csv',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        clf.run()
        print("-"*50)

        # insert into results df
        # rows -> dataset + classifier(embedding file) 
        # columns -> metrics, tn, tp, fn, fp, accuracy, precision, recall, f1