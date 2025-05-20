from typing import List
import numpy as np
from pyspark.sql.dataframe import DataFrame
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score

class SVM:
    def __init__(self, loss="squared_hinge", penalty="l2"):
        """
        Initialize SVM model
        
        Args:
            loss (str): Loss function ('hinge' or 'squared_hinge')
            penalty (str): Penalty ('l1', 'l2')
        """
        self.model = LinearSVC(loss=loss, penalty=penalty, random_state=42)
        
    def train(self, df: DataFrame) -> List:
        """
        Train the model
        
        Args:
            df (DataFrame): Training data
            
        Returns:
            List: [predictions, accuracy, precision, recall, f1]
        """
        # Extract features and labels
        X = np.array([row.image.toArray() for row in df.collect()])
        y = np.array([row.label for row in df.collect()])
        
        # Kiểm tra số lượng class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Warning: Data contains only one class: {unique_classes}. Cannot train SVM.")
            # Trả về giá trị mặc định để tránh crash
            return [], 0.0, 0.0, 0.0, 0.0
        
        # Train the model
        self.model.fit(X, y)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average="weighted")
        recall = recall_score(y, predictions, average="weighted")
        f1 = f1_score(y, predictions, average="weighted")
        
        return predictions, accuracy, precision, recall, f1
    
    def predict(self, df: DataFrame) -> List:
        """
        Make predictions
        
        Args:
            df (DataFrame): Test data
            
        Returns:
            List: [predictions, accuracy, precision, recall, f1, confusion_matrix]
        """
        # Extract features and labels
        X = np.array([row.image.toArray() for row in df.collect()])
        y = np.array([row.label for row in df.collect()])
        
        # Kiểm tra số lượng class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Warning: Data contains only one class: {unique_classes}. Cannot make predictions.")
            # Trả về giá trị mặc định để tránh crash
            return [], 0.0, 0.0, 0.0, 0.0, None
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average="weighted")
        recall = recall_score(y, predictions, average="weighted")
        f1 = f1_score(y, predictions, average="weighted")
        cm = confusion_matrix(y, predictions)
        
        return predictions, accuracy, precision, recall, f1, cm
