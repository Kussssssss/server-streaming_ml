class SparkConfig:
    """Configuration for Spark streaming"""
    def __init__(self, app_name="ML_Streaming", receivers=2, 
                 host="local", stream_host="localhost", port=6100, batch_interval=2):
        """
        Initialize Spark configuration
        
        Args:
            app_name (str): Name of the Spark application
            receivers (int): Number of receivers
            host (str): Spark host
            stream_host (str): Streaming host
            port (int): Streaming port
            batch_interval (int): Batch interval in seconds
        """
        self.appName = app_name
        self.receivers = receivers
        self.host = host
        self.stream_host = stream_host
        self.port = port
        self.batch_interval = batch_interval
