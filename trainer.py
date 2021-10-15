import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. train, test
# 2.
class Trainer():
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit


    def data_split(self, src_data, tgt_data,  val_size = 0.2, test_size = 0.2):
        x_data, self.test_x, y_data, self.test_y = train_test_split(src_data, tgt_data, test_size=test_size)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(x_data, y_data, test_size=val_size)

    def train(self, epochs = 100):
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.hist = self.model.fit(self.train_x, self.train_y, epochs=epochs, validation_data=(self.val_x, self.val_y))

    def test(self):
        test_loss, test_acc = self.model.evaluate(self.test_x, self.test_y)
        print("test_loss    :{}".format(test_loss))
        print("test_accuracy:{}".format(test_acc))

    def visualization(self):
        history_dict = self.hist.history

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 8))

        # loss 그래프
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # accuracy 그래프
        plt.subplot(1, 2, 2)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()