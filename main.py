# Main script for running local server and model
from buildModel import *

def main():
    num_classes = 4
    # Create model instance
    model = buildModel(num_classes)
    # Load model
    model.load_weights('RetinalDiseaseCNN.h5')

if __name__ == '__main__':
    main()