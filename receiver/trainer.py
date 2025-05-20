import pyspark
import logging
import warnings
import os
from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT
from pyspark import SparkConf

from .config import SparkConfig
from .dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 spark_config: SparkConfig,
                 conf=None,
                 log_level="ERROR"):
        """
        Initialize trainer
        
        Args:
            model: ML model (SVM or LogisticRegression)
            spark_config: Spark configuration
            conf: Optional SparkConf for additional configuration
            log_level: Log level (ERROR, WARN, INFO, DEBUG)
        """
        # Tắt cảnh báo Python
        self._configure_python_logging(log_level)
        
        # Tạo file log4j.properties
        self._create_log4j_properties()
        
        self.model = model
        self.sparkConf = spark_config
        self.log_level = log_level
        
        # Tạo hoặc cập nhật SparkConf
        if conf is None:
            conf = SparkConf()
        
        # Cấu hình SparkConf để tắt cảnh báo
        conf = self._configure_spark_conf(conf)
        
        # Khởi tạo SparkContext với cấu hình đã thiết lập
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]", 
                           f"{self.sparkConf.appName}",
                           conf=conf)
        
        # Đặt log level
        self.sc.setLogLevel(log_level)
        self.sc._jsc.sc().setLogLevel(log_level)
        
        self.ssc = StreamingContext(self.sc, self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, self.sparkConf)
        
        # Initialize metrics for tracking
        self.batch_count = 0
        self.total_accuracy = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0
        
        # Tạo schema cho DataFrame - đưa ra ngoài hàm __train__
        self.schema = StructType([
            StructField("image", VectorUDT(), True),
            StructField("label", IntegerType(), True)
        ])
        
        print(f"Trainer initialized with log level: {log_level}")

    def _configure_python_logging(self, log_level):
        """Configure Python logging and warnings"""
        # Thiết lập mức log cho Python logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.ERROR
        logging.basicConfig(level=numeric_level)
        
        # Tắt tất cả cảnh báo Python
        warnings.filterwarnings("ignore")
    
    def _create_log4j_properties(self):
        """Create log4j.properties file for Spark logging configuration"""
        log4j_properties = f"""
# Set everything to be logged to the console
log4j.rootCategory=ERROR, console
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.target=System.err
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d{{yy/MM/dd HH:mm:ss}} %p %c{{1}}: %m%n

# Settings to quiet third party logs that are too verbose
log4j.logger.org.spark-project.jetty=ERROR
log4j.logger.org.spark-project.jetty.util.component.AbstractLifeCycle=ERROR
log4j.logger.org.apache.spark.repl.SparkIMain$exprTyper=ERROR
log4j.logger.org.apache.spark.repl.SparkILoop$SparkILoopInterpreter=ERROR
log4j.logger.org.apache.parquet=ERROR
log4j.logger.parquet=ERROR

# SPARK-9183: Settings to avoid annoying messages when looking up nonexistent UDFs in SparkSQL
log4j.logger.org.apache.spark.sql.catalyst.analysis.SimpleFunctionRegistry=ERROR
"""
        # Lưu file log4j.properties
        with open("log4j.properties", "w") as f:
            f.write(log4j_properties)
        
        # Thiết lập biến môi trường để Spark sử dụng file log4j.properties
        os.environ["SPARK_CONF_DIR"] = os.getcwd()
    
    def _configure_spark_conf(self, conf):
        """Configure SparkConf to suppress warnings"""
        conf.set("spark.storage.replicationFactor", "1")
        conf.set("spark.ui.showConsoleProgress", "false")
        conf.set("spark.executor.logs.rolling.strategy", "none")
        conf.set("spark.executor.logs.rolling.maxRetainedFiles", "0")
        conf.set("spark.executor.logs.rolling.enableCompression", "false")
        conf.set("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties")
        conf.set("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties")
        return conf

    def train(self):
        """Start training process"""
        stream = self.dataloader.parse_stream()
        
        # Sử dụng biến local để tránh tham chiếu đến self trong closure
        model = self.model
        sqlContext = self.sqlContext
        schema = self.schema
        
        # Tạo một hàm closure không tham chiếu đến self
        def process_rdd(time, rdd):
            if not rdd.isEmpty():
                # Sử dụng biến local thay vì self
                df = sqlContext.createDataFrame(rdd, schema)
                predictions, accuracy, precision, recall, f1 = model.train(df)
                
                # In kết quả
                print("="*50)
                print(f"Batch size: {rdd.count()} samples")
                print(f"Batch accuracy: {accuracy:.4f}")
                print(f"Batch precision: {precision:.4f}")
                print(f"Batch recall: {recall:.4f}")
                print(f"Batch F1 score: {f1:.4f}")
                print("="*50)
            else:
                print("Received empty RDD")
        
        # Sử dụng hàm closure đã định nghĩa
        stream.foreachRDD(process_rdd)

        print(f"Starting Spark streaming with batch interval {self.sparkConf.batch_interval}s")
        print(f"Listening on {self.sparkConf.stream_host}:{self.sparkConf.port}")
        
        self.ssc.start()
        self.ssc.awaitTermination()
