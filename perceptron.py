

import matplotlib.pyplot as plt
import numpy as np

#read data
f = open('dataset_8_liver-disorders.arff')
lines = f.readlines()
content = []
for l in lines:
    content.append(l)
datas = []
for c in content:
    cs = c.split(',')
    datas.append(cs)
datas = np.array(datas[55:400]).astype(float)



class Perceptron(object):
    def __init__(self, eta=0.001, n_iter=500, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 2, 1)

    def get_accuracy(self,target,prediction):
        accuracy = 0
        for item in range(len(target)):
            if target[item] == prediction[item]:
                accuracy += 1
        return accuracy/len(target)

total_accuracy = [ ]
for iter in range(50):
    np.random.shuffle(datas)
    X = np.array(datas)[:, :-1]
    y = np.array(datas)[:, -1]
    X_train, X_test, y_train, y_test = X[:310],X[310:],y[:310],y[310:]
    ppn = Perceptron()
    ppn.fit(X_train,y_train)
    #print(ppn.w_)

    total_accuracy.append(ppn.get_accuracy(y_test,ppn.predict(X_test)))
    print(ppn.get_accuracy(y_test,ppn.predict(X_test)))

print(sum(total_accuracy[:])/50)
plt.scatter(range(1, len(total_accuracy) + 1), total_accuracy[:],
            color='blue', marker='x')
plt.axhline(y =sum(total_accuracy[:])/50,color='b')
plt.show()
#print(X_test)
#print(pnn.predict(X_test))
#for test converge
'''
plt.plot(range(1, len(ppn.errors_) + 1), (ppn.errors_), marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
'''