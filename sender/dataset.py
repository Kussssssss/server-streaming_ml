import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm

class Dataset:
    def __init__(self, dataset_type="iris"):
        """
        Initialize dataset loader
        
        Args:
            dataset_type (str): Type of dataset to load ("iris" or "california")
        """
        self.dataset_type = dataset_type
        self.data = None
        self.features = None
        self.labels = None
        self.load_dataset()
        
    def load_dataset(self):
        """Load the specified dataset"""
        if self.dataset_type == "iris":
            df = pd.read_csv("../data/iris.csv")
            self.features = df.iloc[:, :-1].values
            self.labels = df.iloc[:, -1].values
            self.feature_names = df.columns[:-1].tolist()
        else:  # california housing
            df = pd.read_csv("../data/california_housing.csv")
            self.features = df.iloc[:, :-1].values
            self.labels = df.iloc[:, -1].values
            self.feature_names = df.columns[:-1].tolist()
            
        # Convert labels to integer for classification
        if self.dataset_type == "california":
            # Convert regression to classification by binning
            bins = [float('-inf'), 1.5, 3.0, 4.5, float('inf')]
            labels = [0, 1, 2, 3]
            self.labels = np.digitize(self.labels, bins=bins[1:]) - 1
    
    # Thêm phương thức shuffle dữ liệu
    def shuffle_data(self):
        """Shuffle the dataset to ensure diverse classes in each batch"""
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        self.features = self.features[indices]
        self.labels = self.labels[indices]
        print(f"Data shuffled. Unique classes: {np.unique(self.labels)}")
    
    def data_generator(self, batch_size):
        """
        Generate batches of data
        
        Args:
            batch_size (int): Size of each batch
            
        Returns:
            list: List of batches, each containing features and labels
        """
        batches = []
        size_per_batch = (len(self.features) // batch_size) * batch_size
        
        for ix in range(0, size_per_batch, batch_size):
            features_batch = self.features[ix:ix+batch_size]
            labels_batch = self.labels[ix:ix+batch_size]
            
            # Kiểm tra số lượng class trong batch
            unique_classes = np.unique(labels_batch)
            if len(unique_classes) < 2:
                print(f"Warning: Batch contains only one class: {unique_classes}. Skipping...")
                continue
                
            batches.append([features_batch, labels_batch])
            
        return batches
    
    def prepare_payload(self, features_batch, labels_batch):
        """
        Prepare payload for sending over socket
        
        Args:
            features_batch (numpy.ndarray): Batch of features
            labels_batch (numpy.ndarray): Batch of labels
            
        Returns:
            bytes: JSON encoded payload
        """
        batch_size = len(features_batch)
        feature_size = features_batch.shape[1]
        
        payload = dict()
        for batch_idx in range(batch_size):
            payload[batch_idx] = dict()
            for feature_idx in range(feature_size):
                payload[batch_idx][f'feature-{feature_idx}'] = float(features_batch[batch_idx][feature_idx])
            payload[batch_idx]['label'] = int(labels_batch[batch_idx])
            
        # Convert the payload to string
        payload = (json.dumps(payload) + "\n").encode()
        return payload
    
    def send_data_to_spark(self, tcp_connection, batch_size, sleep_time=1):
        """
        Send data to Spark over TCP connection
        
        Args:
            tcp_connection (socket.socket): TCP socket connection
            batch_size (int): Size of each batch
            sleep_time (int): Time to sleep between batches
        """
        batches = self.data_generator(batch_size)
        total_batches = len(batches)
        
        if total_batches == 0:
            print("No valid batches to send. Try increasing batch size or shuffling data.")
            return
            
        pbar = tqdm(total=total_batches)
        data_sent = 0
        
        for batch in batches:
            features, labels = batch
            
            # Kiểm tra lại số lượng class trong batch
            unique_classes = np.unique(labels)
            if len(unique_classes) < 2:
                print(f"Warning: Batch contains only one class: {unique_classes}. Skipping...")
                continue
                
            payload = self.prepare_payload(features, labels)
            
            try:
                tcp_connection.send(payload)
            except BrokenPipeError:
                print("Either batch size is too big for the dataset or the connection was closed")
                break
            except Exception as error_message:
                print(f"Exception thrown but was handled: {error_message}")
                break
                
            data_sent += 1
            pbar.update(n=1)
            pbar.set_description(f"Batch: {data_sent}/{total_batches} | Sent: {batch_size} samples | Classes: {len(unique_classes)}")
            time.sleep(sleep_time)
