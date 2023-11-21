# Main script for running local server and model
from buildModel import *

def main():
    num_classes = 4
    
    # Create model instance
    model = buildModel(num_classes)

    # Load model
    # model.load_weights('RetinalDiseaseCNN.h5')
    model.load_weights('checkpoint/cp.ckpt')

if __name__ == '__main__':
    main()