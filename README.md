# deepLearning-pytorch
Implementation of a Neural Network using PyTorch.
# Structure
* Class `ChallengeDataset` in src/data.py implements a Dataset, which basically returns an image and the corresponding label.
* src/model.py contains the Class `ResNet` which is the model. It implements a ResNet Architecture 
  * Uses Convolution Layers
  * Uses `ResBlocks` which are defined in src/ResBlock.py
* src/showImage.py is just a little script that shows how a augmentend picture looks like (not too important)
* src/PyTorchChallengeTest.py contains test cases. 
* 