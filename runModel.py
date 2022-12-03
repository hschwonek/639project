import json
import logging as log
import matplotlib.pyplot as plt
import torch

from utils import load_model
from torchvision import datasets, transforms
from torchvision.models import resnet18


def load_model_and_data(model, filename, use_gpu):
    log.info("Starting loads ...")

    # Load the model onto GPU if possible
    load_model(model, filename=filename, use_gpu=use_gpu)
    log.info("Model Loaded")

    # Basic transform to be used on images in dataset
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Load the dataset applying the transform to each image
    dataset = datasets.ImageFolder(
        'C:/Users/hosch/Documents/cs639/plantnet/PlantNet-300K/images',
        transform=transform
    )
    log.info("Dataset Loaded\nLoads Complete")

    return model, dataset


def main():
    # Debug Logging Settings
    log.basicConfig(filename='prediction.log', level=log.DEBUG)
    log.info('Logging started...')

    # General Model Settings
    filename = 'models/resnet18_weights_best_acc.tar'       # Path to model
    use_gpu = True                                          # Use GPU
    
    # create generic resnet18 model with 1081 classes as defined by Pl@ntNet Dataset
    model = resnet18(num_classes=1081)

    # Load the model and the dataset
    model, dataset = load_model_and_data(model, filename, use_gpu)

    # Load an image to for prediction
    image = dataset[1][0]   
    model.eval()            

    # Predict probabilities for every class
    prediction = model(image.unsqueeze(0))
    print(prediction)

    # Find the top three probabilites 
    prob = torch.nn.functional.softmax(prediction, 1)
    max, inds = torch.topk(prob[0], 3)
    print(torch.topk(prob[0], 3))

    # Load the class names as a dictionary of values
    class_names = {}
    with open('data/plantnet300k_species_id_2_name.json') as classes:
        class_names = json.load(classes)

    # Print the top three classes for the prediction
    ids = []
    inds = inds.tolist()
    max = max.tolist()
    for key in class_names.keys():
        ids.append(key)

    for i in range(3):
        index = inds[i]
        print(f"{class_names[ids[index]]}: {max[i]*100:.2f}%")


if __name__ == "__main__":
    main()
