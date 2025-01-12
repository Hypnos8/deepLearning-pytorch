import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
#from tqdm.autonotebook import tqdm


class Trainer:
    """

    """
    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        self.f1_score_val = []

    def save_checkpoint(self, epoch):
        """
        Save current model as PyTorch checkpoint file (ckp)
        The trained model can later be imported and used IN PYTORCH
        :param epoch:
        :return:
        """
        torch.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        """
        Load a checkpoint so it can be used
        :param epoch_n:
        :return:
        """
        ckp = torch.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        """
        Save current model as ONXX
        The trained model can later be imported and used by most Frameworks (PyTorch, TensorFlow etc)

        :param fn: File where the model will be saved
        :return:
        """
        m = self._model.cpu()
        m.eval()
        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        """
        Performs a single training step (usually involing one batch)
        :param x: data
        :param y: Labels of the data
        :return: Caluclated Loss
        """
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()

        # propagate through the network
        pred = self._model(x)

        # calculate the loss_function
        loss = self._crit(pred, y.type(torch.float))

        # compute gradient by backward propagation
        loss.backward()

        # update weights
        self._optim.step()

        # -return the loss
        return loss

    def val_test_step(self, x, y):
        """
        Performs a single validation (that is, no backward propagation/learning!)  step,  usually involing one batch
        :param x:
        :param y:
        :return:
        """
        # propagate through the network and calculate the loss and predictions
        y_pred = self._model(x)
        loss = self._crit(y_pred, y.type(torch.float))

        return loss, y_pred

    def train_epoch(self):
        """
        Train for one epoche (= going once throught the whole training data set)
        :return:
        """
        # set training mode
        self._model.train()

        total_loss = 0.0
        # iterate through the training set
        for batch in self._train_dl:
            data, labels_batch = batch

            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                data = data.to('cuda')
                labels_batch = labels_batch.to('cuda')


            # perform a training step
            loss = self.train_step(data, labels_batch)
            total_loss += loss

        # calculate the average loss for the epoch and return it
        avg_loss = total_loss / len(self._train_dl)

        return avg_loss  # Return average loss per Batch

    def val_test(self):
        """
        Goes through the whole validation data set and prints the metrics
        :return:
        """
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.).
        # To handle those properly, you'd want to call model.eval()
        self._model.eval()

        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with torch.inference_mode():
            total_loss = 0.0
            labels = []
            predictions = []

            # iterate through the validation set
            for batch in self._val_test_dl:
                data, labels_batch = batch
                # transfer the batch to the gpu if given
                if self._cuda:
                    data = data.to('cuda')
                    labels_batch = labels_batch.to('cuda')

                # perform a validation step
                loss, predictions_batch = self.val_test_step(data, labels_batch)  # consumes big amount of vram

                # save the predictions_batch and the labels_batch for each batch
                total_loss += loss
                labels.append(labels_batch.cpu().detach().numpy())
                predictions.append(predictions_batch.cpu().detach().numpy())

        self.__print_metrics(labels, predictions, total_loss)



    def fit(self, epochs=-1):
        """
        Let the model fit (= learn)  the training data
        This will repeatedly run a training and a validation period until a stop criterion is met

        :param epochs:
        :return:
        """
        assert self._early_stopping_patience > 0 or epochs > 0

        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses = []
        validation_losses = []
        epoche_counter = 0

        best_loss = None
        early_stopping_counter = 0
        while True:
            # stop by epoch number
            if epoche_counter >= epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            loss_train = self.train_epoch()
            loss_validate = self.val_test()

            # append the losses to the respective lists
            train_losses.append(loss_train.cpu().detach().numpy())
            validation_losses.append(loss_validate.cpu().detach().numpy())

            epoche_counter += 1
            print(epoche_counter)

            if best_loss is None:
                best_loss = loss_validate

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if loss_validate > best_loss:
                early_stopping_counter += 1
            else:
                best_loss = loss_validate
                print("-- Best loss (until now) at epoche: ", epoche_counter, "with best_loss value ", best_loss)
                early_stopping_counter = 0

                # use the save_checkpoint function to save the model
                self.save_checkpoint(epoche_counter)

            if early_stopping_counter > self._early_stopping_patience:
                print("Early stop")
                break

        print("\nEpoch counter: ", epoche_counter)
        print("Mean F1: ", np.mean(self.f1_score_val))
        print("Best F1 Score: ", np.max(self.f1_score_val), " at epoch ", str(np.argmax(self.f1_score_val)+1))
        epoches = [x for x in range(1, epoche_counter+1, 1)]
        plt.plot(epoches, self.f1_score_val)
        plt.show()

        # return the losses for both training and validation
        return train_losses, validation_losses

    def __print_metrics(self, labels, predictions, total_loss):
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        predictions = np.where(predictions > 0.5, 1, 0)
        avg_loss = total_loss / len(self._val_test_dl)
        accuracy = accuracy_score(labels, predictions, normalize=True)
        precision = precision_score(labels, predictions, average="micro")
        recall = recall_score(labels, predictions, average="micro")
        f1 = f1_score(labels, predictions, average="micro")
        self.f1_score_val.append(f1)

        # return the loss and print the calculated metrics
        print('Accuracy: ', accuracy)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1: ', f1)
        return avg_loss
