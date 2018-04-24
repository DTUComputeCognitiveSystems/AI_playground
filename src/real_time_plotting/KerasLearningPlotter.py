import keras
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)
from IPython.display import clear_output

class KerasLearningPlotter(keras.callbacks.Callback):
    def __init__(self):
        super(KerasLearningPlotter, self).__init__()

    def on_train_begin(self, logs={}):
        # Initialize structures
        self.i = 0
        self.e = 0
        self.x_batch = []
        self.x_epoch = []
        self.losses = []
        self.acc = []
        self.validation = self.params['do_validation']
        if self.validation:
            self.val_losses = []
            self.val_acc = []

        self.fig = plt.figure()
        self.logs = []

    def on_batch_end(self, batch, logs={}):
        # Save current metrics
        self.logs.append(logs)
        self.x_batch.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.i += 1

        clear_output(wait=True)
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        f.suptitle('Epoch ' + str(self.e), fontsize=20)

        # Plot training curves
        ax1.plot(self.x_batch, self.losses, label="loss")
        ax2.plot(self.x_batch, self.acc, label="accuracy")
        ax1.set_xlabel('batch')
        ax2.set_xlabel('batch')

        # Plot validation curves
        if len(self.x_epoch) > 1 and self.validation:
            ax1.plot(self.x_epoch, self.val_losses, label="val_loss", lw=2)
            ax2.plot(self.x_epoch, self.val_acc, label="validation accuracy", lw=2)

        # Show labels
        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        plt.pause(0.1)
        plt.show()

    def on_epoch_end(self, epoch, logs={ }):
        self.x_epoch.append(self.i)
        if self.validation:
            self.val_losses.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_acc'))
        self.e += 1
