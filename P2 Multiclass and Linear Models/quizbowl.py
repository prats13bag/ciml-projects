import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from numpy import *
import datasets

if not datasets.Quizbowl.loaded:
    datasets.loadQuizbowl()
print ('\n*****LinearSVR*****')
print('\nRUNNING ON EASY DATA\n')
X = datasets.QuizbowlSmall.X
Y = datasets.QuizbowlSmall.Y

print('training ava')
ava = OneVsRestClassifier(LinearSVR(random_state=0)).fit(X, Y)
print('predicting ava')
avaDevPred = ava.predict(datasets.QuizbowlSmall.Xde)
print('error = {0}'.format(mean(avaDevPred != datasets.QuizbowlSmall.Yde)))
savetxt('predictionsQuizbowlSmall-LinearSVR.txt', avaDevPred)

print('\n\nRUNNING ON HARD DATA\n')    
X = datasets.QuizbowlHardSmall.X
Y = datasets.QuizbowlHardSmall.Y
print('training ava')
ava = OneVsRestClassifier(LinearSVR(random_state=0)).fit(X, Y)
print('predicting ava')
avaDevPred = ava.predict(datasets.QuizbowlHardSmall.Xde)
print('error = {0}'.format(mean(avaDevPred != datasets.QuizbowlHardSmall.Yde)))

savetxt('predictionsQuizbowlHardSmall-LinearSVR.txt', avaDevPred)

print ('\n\n*****MLPClassifier*****')
print('\nRUNNING ON EASY DATA\n')
X = datasets.QuizbowlSmall.X
Y = datasets.QuizbowlSmall.Y

print('training ava')
# ava = OneVsRestClassifier(MultinomialNB(alpha=1)).fit(X, Y)
ava = OneVsRestClassifier(MLPClassifier(alpha=1)).fit(X, Y)
print('predicting ava')
avaDevPred = ava.predict(datasets.QuizbowlSmall.Xde)
print('error = {0}'.format(mean(avaDevPred != datasets.QuizbowlSmall.Yde)))
savetxt('predictionsQuizbowlSmall-MLPClassifier.txt', avaDevPred)

print('\n\nRUNNING ON HARD DATA\n')    
X = datasets.QuizbowlHardSmall.X
Y = datasets.QuizbowlHardSmall.Y
print('training ava')
# ava = OneVsRestClassifier(MultinomialNB(alpha=1)).fit(X, Y)
ava = OneVsRestClassifier(MLPClassifier(alpha=1)).fit(X, Y)
print('predicting ava')
avaDevPred = ava.predict(datasets.QuizbowlHardSmall.Xde)
print('error = {0}'.format(mean(avaDevPred != datasets.QuizbowlHardSmall.Yde)))

savetxt('predictionsQuizbowlHardSmall-MLPClassifier.txt', avaDevPred)