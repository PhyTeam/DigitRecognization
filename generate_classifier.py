from sklearn.externals import joblib
from sklearn import datasets
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
from collections import Counter

import mnist_loader
import matplotlib.pyplot as plt
tra, tes, val = mnist_loader.load_data_wrapper()
images_and_labels = tra[:8]
for index, (image, label) in enumerate(images_and_labels):
    plt.subplot(2,4, index + 1)
    plt.axis('off')
    # Convert 2 image
    image = image.reshape((28,28))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Trainning: %i' % int(np.argwhere(label == 1)[0][0]) )
    print np.argwhere(label == 1)[0][0]
#plt.show()

# create classifier: a support vector machine
clf = SVC(gamma=0.001)

# Trainning
X = np.array([x for x, y in tra])
n_samples = len(X)
X = X.reshape((n_samples, -1))
Y = [y for x, y in tra]
Y = np.array([np.argmax(v) for v in Y])
print X.shape

clf.fit(X, Y)
# Now predict the value of digit on the test data
X = np.array([x for x,y in tes])
n_samples = len(X)
X = X.reshape((n_samples, -1))
Y = np.array([y for x,y in tes])
expected = [np.argmax(v) for v in Y]
predicted = clf.predict(X)
metrics.classification_report((expected, predicted))

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
