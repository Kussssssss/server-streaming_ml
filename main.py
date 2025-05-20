import os
import sys
import subprocess

devnull = 'NUL'

original_stderr = sys.stderr

null_file = open(devnull, 'w')

sys.stderr = null_file

from receiver.models.svm import SVM
from receiver.models.logistic import LogisticRegressionModel
from receiver.config import SparkConfig
from receiver.trainer import Trainer
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ML model training with Spark streaming')
    parser.add_argument('--model', '-m', help='Model type (svm or logistic)', 
                        required=False, type=str, default='svm')
    parser.add_argument('--host', help='Streaming host', 
                        required=False, type=str, default='localhost')
    parser.add_argument('--port', '-p', help='Streaming port', 
                        required=False, type=int, default=6100)
    parser.add_argument('--batch-interval', '-b', help='Batch interval in seconds', 
                        required=False, type=int, default=2)
    parser.add_argument('--app-name', '-a', help='Spark application name', 
                        required=False, type=str, default='ML_Streaming')
    parser.add_argument('--log-level', '-l', help='Log level (ERROR, WARN, INFO, DEBUG)', 
                        required=False, type=str, default='ERROR')
    parser.add_argument('--show-warnings', '-w', help='Show warnings', 
                        action='store_true', default=False)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.show_warnings:
        sys.stderr = original_stderr
    
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.ui.showConsoleProgress=false pyspark-shell'
    
    spark_config = SparkConfig(
        app_name=args.app_name,
        stream_host=args.host,
        port=args.port,
        batch_interval=args.batch_interval
    )
    
    if args.model.lower() == 'svm':
        model = SVM(loss="squared_hinge", penalty="l2")
        print("Using SVM model")
    else:
        model = LogisticRegressionModel(penalty="l2", solver="lbfgs")
        print("Using Logistic Regression model")
    
    trainer = Trainer(model, spark_config, log_level=args.log_level)
    
    print(f"Starting training with {args.model} model")
    print(f"Log level set to: {args.log_level}")
    print(f"Warnings are {'enabled' if args.show_warnings else 'disabled'}")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        sys.stderr = original_stderr
        null_file.close()
