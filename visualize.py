import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


class Visualize:
    def __init__(self, numbers, losses, accuracy, features, labels):
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.colors = ['blue', 'cyan', 'green', 'black', 'magenta', 'red', 'yellow']
        self.numbers = numbers
        self.losses = losses
        self.accuracy = accuracy
        self.features = features
        self.labels = labels

    def loss_accuracy_visualize(self):
        plt.plot(self.numbers, self.losses, label='loss')
        plt.plot(self.numbers, self.accuracy, label='accuracy')
        plt.xlabel("train_numbers")
        plt.legend()
        plt.savefig("train_losses_accuracies.svg", dpi=600)

    def features_visualize(self):
        tsne = manifold.TSNE(init='pca')
        labels = self.labels
        labels = labels.reshape(len(labels), 1)
        all = np.hstack((self.features, labels))
        np.random.shuffle(all)
        features = all[:200, :-1]
        labels = all[:200, -1:]
        labels = labels.ravel().astype(np.int32)
        features_tsne = tsne.fit_transform(features)
        features_norm = (features_tsne - features_tsne.min(0)) / (features_tsne.max(0) - features_tsne.min(0))
        plt.cla()
        for i in range(features_norm.shape[0]):
            plt.text(features_norm[i, 0], features_norm[i, 1], color=self.colors[labels[i]],
                     s=self.classes[labels[i]])
        plt.savefig("features.svg", dpi=600)
