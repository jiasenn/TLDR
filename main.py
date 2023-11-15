# Main script for running local server and model
from keras.models import load_model

def main():
    # Load model
    model = load_model('RetinalDiseaseCNN.h5')

if __name__ == '__main__':
    main()