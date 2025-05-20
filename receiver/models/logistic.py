from typing import List
import numpy as np
from pyspark.sql.dataframe import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score

class LogisticRegressionModel:
    def __init__(self, penalty="l2", solver="lbfgs", max_iter=1000):
        """
        Initialize Logistic Regression model
        
        Args:
            penalty (str): Penalty ('l1', 'l2', 'elasticnet', 'none')
            solver (str): Solver algorithm
            max_iter (int): Maximum number of iterations
        """
        self.model = LogisticRegression(
            penalty=penalty, 
            solver=solver, 
            max_iter=max_iter, 
            random_state=42,
            multi_class='auto'
        )
        
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
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average="weighted")
        recall = recall_score(y, predictions, average="weighted")
        f1 = f1_score(y, predictions, average="weighted")
        cm = confusion_matrix(y, predictions)
        
        return predictions, accuracy, precision, recall, f1, cm
