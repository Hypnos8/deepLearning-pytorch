# deepLearning-pytorch
This implements a ResNet that is trained on images with solar panels, which have the properties 'cracked' and 'inactive'
It was created as part of a exercise of the lecture Deep Learning at the FAU Erlangen-Nürnberg.
# Structure

Source code files are located in src/
* The project contains an Instance of a PyTorch Dataset (in data.py) that stores the data and augments them (e.g. rotates pictures)
  * The data sets consists of a CSV file (containing filenames,cracked, and inactive) and the actual image files
* The actual ResNet Model is defined in model.py
  * It uses ResBlocks that are defined in ResBlock.py
* The code for Fitting  the Network can  be found in trainer.py
  * The Fitting process  
    * Consists of multiple iterations of Training and Validation 
    * stops for after a  predefined number of epochs, or when the early stop criterium is met (that is, the loss wasn't improved over the last N epochs)
  * Checkpoints are automatically saved when the current loss is better than all previous ones
* Starting an actual training is done in train.py
  * It uses some helper Methods defined in TrainHeper.py 


## Other Files (not necessarily required)
* src/showImage.py is just a little script that shows how a augmentend picture looks like (not too important)
* src/PyTorchChallengeTest.py contains test cases.

# Usage
The training process can be started by running `train.py`
Checkpoints (as PyTorch Checkpoints) are automatically stored at the folder checkpoints
After each Epoch some metrics (F1, Accuracy, Precision) are printed. They can be used to identify the best run and to store the corresponding checkpoint of the epoche.



## Reusing the project for other training tasks
* Model.py has to be adjusted to match the  desired model
* DataSets
  * data.py has to be adjusted to meet your needs (e.g change transformations), but also on how you want to load your data
  * TrainHelper.py has to be adjusted as well, as it defines where to laod the data from and which labels are considered
