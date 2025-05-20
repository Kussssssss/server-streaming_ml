import json
import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector

from .config import SparkConfig

class DataLoader:
    def __init__(self, 
                 sparkContext: SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig):
        """
        Initialize data loader for Spark streaming
        
        Args:
            sparkContext: Spark context
            sparkStreamingContext: Spark streaming context
            sqlContext: SQL context
            sparkConf: Spark configuration
        """
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )
    
    def parse_stream(self) -> DStream:
        """
        Parse incoming data stream
        
        Returns:
            DStream: Processed data stream
        """
        # Parse JSON from stream
        json_stream = self.stream.map(lambda line: json.loads(line))
        
        # Explode JSON to get individual samples
        json_stream_exploded = json_stream.flatMap(lambda x: x.values())
        
        # Định nghĩa hàm extract_features_and_label ở ngoài để tránh tham chiếu đến self
        def extract_features_and_label(sample):
            # Get all keys and sort them to ensure consistent order
            keys = sorted([k for k in sample.keys() if k.startswith('feature')])
            
            # Extract features in order
            features = [sample[k] for k in keys]
            
            # Extract label
            label = sample['label']
            
            return [DenseVector(features), label]
        
        # Extract features and labels
        features_and_labels = json_stream_exploded.map(extract_features_and_label)
        
        return features_and_labels
