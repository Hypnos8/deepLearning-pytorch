# deepLearning-pytorch
Implementation of a Neural Network using PyTorch.
# Structure
This implements a ResNet that is trianed on images with solar panels, which have the properties 'cracked' and 'inactive'

Source code files are located in src/
* The project contains an Instance of a PyTorch Dataset (in data.py) that stores the data and augments them (e.g. rotates pictures)
* The actual ResNet Model is defined in model.py
  * It uses ResBlocks that are defined in ResBlock.py
* The code for Training/Evaluation data can be found in trainer.py
* Starting an actual training is done in train.py
  * It uses some helper Methods defined in TrainHeper.py 


## Other Files
* src/showImage.py is just a little script that shows how a augmentend picture looks like (not too important)
* src/PyTorchChallengeTest.py contains test cases.