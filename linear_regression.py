import numpy as np
from matplotlib import pyplot as plt


def generate_wave_set(n_support=1000, n_train=25, std=0.3):
    data = {}
    data['support'] = np.linspace(-np.pi/2, np.pi/2, num=n_support)
    data['values'] = np.sin(data['support']) + 2
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    data['y_train'] = np.sin(data['x_train']) + 2 + np.random.normal(0, std, size=data['x_train'].shape[0])
    return data


data = generate_wave_set(1000, 100)
for i in range(10):
    leftBorder = 10 * i
    rightBorder = 10 * (i + 1)
    testData = data['x_train'][leftBorder:rightBorder]
    trainData = np.append(data['x_train'][0:leftBorder], data['x_train'][rightBorder:100])
    yTrain = np.append(data['y_train'][0:leftBorder], data['y_train'][rightBorder:100])
    X = np.array([np.ones(trainData.shape[0]), trainData]).T
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), yTrain)
    y_hat = np.dot(w, X.T)
    print(len(y_hat))
    yTested = data['y_train'][leftBorder:rightBorder]
    yTrained = y_hat[leftBorder:rightBorder]
    for j in range(10):
        print(yTested[j] - yTrained[j])
    plt.plot(trainData, y_hat, 'r+', alpha=0.8, label='fitted')
    plt.show()

# X = np.array([np.ones(data['x_train'].shape[0]), data['x_train']]).T
#
# w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), data['y_train'])
#
# y_hat = np.dot(w, X.T)
# print(y_hat)
# margin = 0.3
# fig = plt.figure()
# plt.plot(data['support'], data['values'], 'b--', alpha=0.5, label='manifold')
# plt.scatter(data['x_train'], data['y_train'], 40, 'g', 'o', alpha=0.8, label='data')
#
# plt.plot(data['x_train'], y_hat, 'r', alpha=0.8, label='fitted')
#
# plt.xlim(data['x_train'].min() - margin, data['x_train'].max() + margin)
# plt.ylim(data['y_train'].min() - margin, data['y_train'].max() + margin)
# plt.legend(loc='upper right', prop={'size': 10})
# plt.title('Linear regression')
# plt.xlabel('x')
# plt.xlabel('y')
# plt.show()
# fig.savefig('plot.png')
