import json
import logging as log
import torch
import sys

from constants import *
from torchvision import datasets, transforms
from torchvision.models import resnet18, alexnet

from utils import load_model


def load(filename, use_gpu, model_type):
    log.info("Starting loads ----------------------------------")

    # Create model 
    model = None
    match model_type:
        case 'resnet':
            model = resnet18(num_classes = NUM_CLASSES)
        case 'alexnet':
            model = alexnet(num_classes = NUM_CLASSES)
        case 'custom_resnet':
            model = resnet18(num_classes = NUM_CLASSES)
        case _:
            log.error("ERROR: Invalid model type")
            sys.exit(1)
    
    # Load the model onto GPU if possible otherwise on the cpu
    load_model(model, filename=filename, use_gpu=use_gpu)
    log.info("Model loaded")

    return model


def get_image():
    # Basic transform to be used on images in dataset
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Load the dataset applying the transform to each image
    predict_images = datasets.ImageFolder(
        USER_IMAGES_PATH,
        transform=transform
    )
    log.info("Images for prediction loaded")

    return predict_images


def predict(model):
    # Load an image and set model to evaluation mode
    log.info("Obtaining images for prediction")
    images = get_image()

    model.eval()   
    log.info("Model set to evaluation mode")

    # Predict probabilities for each image
    predictions = [model(image[0].unsqueeze(0)) for image in images]
    log.info("Predictions Complete:")
    log.info(predictions)

    return predictions

def classify(predictions):
    # Classify the top three predicttions
    for i in range(len(predictions)):
        print(f"Image #{i} Prediction: ----------------------------------')")

        prob = torch.nn.functional.softmax(predictions[i], 1)
        max, inds = torch.topk(prob[0], TOP_K)

        # Load the class names as a dictionary of values
        class_names = {}
        with open(CLASSES_PATH) as classes:
            class_names = json.load(classes)

        # Print the top three classes and their % prediction
        ids = [key for key in class_names.keys()]
        print(*(f"{class_names[ids[inds[i]]]}: {max[i]*100:.2f}%" for i in range(TOP_K)), sep='\n', end='\n\n')

def main():
    # Debug Logging Settings
    log.basicConfig(filename='prediction.log', level=log.DEBUG, filemode='w')
    torch.set_printoptions(profile="default")
    log.info('Logging started ----------------------------------')
 
    # Check parameters
    if len(sys.argv) != 2:
        print("ERROR: Please provide a model name as argument")
        sys.exit(1)

    # General Model Settings
    model_type = sys.argv[1]
    filename = MODELS[model_type] # Path to model
    use_gpu = True                # Use GPU
    
    # Load the model and the dataset
    model = load(filename, use_gpu, model_type)

    # Make a prediction with model
    predictions = predict(model)

    # Display the results
    classify(predictions)


if __name__ == "__main__":
    main()