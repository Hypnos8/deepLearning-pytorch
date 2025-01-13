# deepLearning-pytorch
This implements a ResNet that is trained on images with solar panels, which have the properties 'cracked' and 'inactive'
It was created as part of a exercise of the lecture Deep Learning at the FAU Erlangen-Nürnberg.
# Structure

Source code files are located in src/
* The project contains an Instance of a PyTorch Dataset (in data.py) that stores the data and augments them (e.g. rotates pictures)
  * The data sets consists of a CSV file (containing filenames,cracked, and inactive) and the actual image files
* The actual ResNet Model is defined in model.py
  * It uses ResBlocks that are defined in ResBlock.py
* The code for Training/Evaluation data can be found in trainer.py
* Starting an actual training is done in train.py
  * It uses some helper Methods defined in TrainHeper.py 


## Other Files (not necessarily required)
* src/showImage.py is just a little script that shows how a augmentend picture looks like (not too important)
* src/PyTorchChallengeTest.py contains test cases.

# Reusability

* Model.py has to be adjusted to match your own model
* DataSets
  * data.py has to be adjusted to meet your needs (e.g change transformations), but also on how you want to load your data
  * TrainHelper.py has to be adjusted as well, as it defines where to laod the data from and which labels are considered
