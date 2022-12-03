# Plant Classification

## How to Use the Application
A user can request a prediction on a set of images. These images must go in the `user_images` directory. All images must then be placed in another sub-directory. For example the demo organizes the directory like so:

    user_images
    |	flowers
    |	|	1.jpg
    |	|	2.jpg
    |	|	3.jpg
    |	|	4.jpg
    |	|	5.jpg

Once the images are correctly organized. A user can run the following command to get predictions of the species for each plant:

    python3 .\runModel.py <model_name> 
  
The `"<model_name>"` parameter must be specified for the program to execute. The supported models are listed in the **Supported Models** section.
## Supported Models
We currently support the following models:

 - Resnet18 (Pl@ntNet): `python3 .\runModel.py resnet`
	 - Download Link: https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/files/?p=%2Fresnet18_weights_best_acc.tar
 - Alexnet (Pl@ntNet): `python3 .\runModel.py alexnet`
	 - Download Link: https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/file/?p=%2Falexnet_weights_best_acc.tar
 - #TBD

All models need to be organized in the following way:

    639project
    |	models
    |	|	alexnet_weights_best_acc.tar
    |	|	resnet18_weights_best_acc.tar
    |	|	etc.

