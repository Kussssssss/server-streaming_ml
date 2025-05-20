import socket
import argparse
import time
from dataset import Dataset

# Default configuration
TCP_IP = "localhost"
TCP_PORT = 6100

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stream data to a Spark Streaming Context')
    parser.add_argument('--dataset', '-d', help='Dataset type (iris or california)', 
                        required=False, type=str, default='iris')
    parser.add_argument('--batch-size', '-b', help='Batch size', 
                        required=False, type=int, default=20)  # Tăng batch size lên 20
    parser.add_argument('--endless', '-e', help='Enable endless stream',
                        required=False, type=bool, default=False)
    parser.add_argument('--sleep', '-s', help='Streaming interval in seconds', 
                        required=False, type=int, default=1)
    parser.add_argument('--host', help='TCP host', 
                        required=False, type=str, default=TCP_IP)
    parser.add_argument('--port', '-p', help='TCP port', 
                        required=False, type=int, default=TCP_PORT)
    
    return parser.parse_args()

def connect_tcp(host, port):
    """
    Create and connect TCP socket
    
    Args:
        host (str): Host address
        port (int): Port number
        
    Returns:
        tuple: Socket connection and address
    """
    # Thay đổi: Thêm retry logic
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            print(f"Waiting for connection on {host}:{port}...")
            connection, address = s.accept()
            print(f"Connected to {address}")
            return connection, address
        except socket.error as e:
            if attempt < max_retries - 1:
                print(f"Socket error: {e}. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Failed to create socket after {max_retries} attempts: {e}")
                raise

def stream_data(dataset_type, batch_size, endless, sleep_time, host, port):
    """
    Stream data to Spark
    
    Args:
        dataset_type (str): Type of dataset to use
        batch_size (int): Size of each batch
        endless (bool): Whether to stream endlessly
        sleep_time (int): Time to sleep between batches
        host (str): Host address
        port (int): Port number
    """
    dataset = Dataset(dataset_type)
    
    # Thêm: Shuffle dữ liệu để đảm bảo mỗi batch có nhiều class
    dataset.shuffle_data()
    
    try:
        tcp_connection, _ = connect_tcp(host, port)
        
        try:
            if endless:
                print(f"Starting endless streaming of {dataset_type} dataset...")
                while True:
                    dataset.send_data_to_spark(tcp_connection, batch_size, sleep_time)
                    print("Restarting stream...")
            else:
                print(f"Starting one-time streaming of {dataset_type} dataset...")
                dataset.send_data_to_spark(tcp_connection, batch_size, sleep_time)
                print("Streaming completed.")
        except KeyboardInterrupt:
            print("Streaming interrupted by user.")
        finally:
            tcp_connection.close()
            print("Connection closed.")
    except Exception as e:
        print(f"Error in stream_data: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    
    dataset_type = args.dataset
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    host = args.host
    port = args.port
    
    stream_data(dataset_type, batch_size, endless, sleep_time, host, port)
