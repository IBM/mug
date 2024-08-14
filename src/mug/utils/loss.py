'''
 (C) Copyright IBM Corp. 2024.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
           http://www.apache.org/licenses/LICENSE-2.0
     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
 Project name: Model Urban Growth MUG
'''
import torch
import matplotlib.pyplot as plt

from mug.utils.functions import target_mask, pred_mask, class_mask

class LossComputation:
    def __init__(self, device, batch_size, size, num_classes, weight):
        """
        Initializes the LossComputation class with necessary parameters.

        Args:
            device (torch.device): The device (CPU or GPU) to perform computations on.
            batch_size (int): The size of the batch.
            size (int): The size parameter for the target mask.
            num_classes (int): The number of classes for the class mask.
            weight (list of float): A list of weights for each class.
        """
        self.device = device
        self.batch_size = batch_size
        self.size = size
        self.num_classes = num_classes
        self.weight = weight

    def compute_loss(self, y_hat, y):
        """
        Computes the weighted Mean Squared Error (MSE) loss for the given model outputs and targets.

        Args:
            y_hat (torch.Tensor): Predicted tensor of shape [batch_size, time_steps, height, width].
            y (torch.Tensor): Target tensor of shape [batch_size, time_steps, height, width].

        Returns:
            torch.Tensor: The computed weighted MSE loss.
        """

        # Applying masks
        yy = target_mask(y[:, 0], self.batch_size, self.size).to(self.device)
        yy_class = class_mask(self.num_classes, y=yy).to(self.device)

        # Weighted MSE loss
        loss = sum(
            self.weight[class_indx] * torch.linalg.norm(yy_class[class_indx] * (y[:, 0] - y_hat[:, 0])) ** 2
            for class_indx in range(self.num_classes)
        )

        return loss

class LossPlotter:
    def __init__(self, train_loss_history, val_loss_history):
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

    def plot(self):
        fig, ax = plt.subplots()

        # Plot the training and validation losses
        epochs = range(1, len(self.train_loss_history) + 1)
        ax.plot(epochs, self.train_loss_history, label='Training Loss')
        ax.plot(epochs, self.val_loss_history, label='Validation Loss')

        # Set the title and labels
        ax.set_title('Training and Validation Loss per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        ax.legend()
        plt.show()

class AccPlotter:
    def __init__(self, train_acc_history, val_acc_history):
        self.train_acc_history = train_acc_history
        self.val_acc_history = val_acc_history

    def plot(self):
        fig, ax = plt.subplots()

        # Plot the training and validation losses
        epochs = range(1, len(self.train_acc_history) + 1)
        ax.plot(epochs, self.train_acc_history, label='Training Accuracy')
        ax.plot(epochs, self.val_acc_history, label='Validation Accuracy')

        # Set the title and labels
        ax.set_title('Training and Validation Accuracy per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        ax.legend()
        plt.show()

