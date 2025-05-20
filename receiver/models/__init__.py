from .svm import SVM
from .logistic import LogisticRegressionModel

def create_model_init():
    """Create __init__.py file for models module"""
    with open("__init__.py", "w") as f:
        f.write("from .svm import SVM\n")
        f.write("from .logistic import LogisticRegressionModel\n")
