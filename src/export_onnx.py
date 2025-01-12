import torch as t

from trainer import Trainer
import sys
import model
import torchvision as tv

#epoch = int(sys.argv[1])
epoch = 189
#TODO: Enter your model here

resNet = model.ResNet()

crit = t.nn.BCELoss()
trainer = Trainer(resNet, crit, cuda=True)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
