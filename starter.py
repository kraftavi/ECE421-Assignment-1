import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    # Your implementation here
    predictions = np.dot(x,W) + b
    err = predictions - y
    loss = (np.linalg.norm(err,2))**2
    loss = loss/len(y)
    for i in range(0,len(W)):
        loss += reg/2*(W[i])**2
    return loss[0]

def grad_MSE(W, b, x, y, reg):
    # Your implementation here
    predictions = np.dot(x,W) + b
    err = predictions - y
    grad_bias = np.sum(err)*2/len(y)
    grad_weights = np.zeros(W.shape)
    for i in range(0,len(grad_weights)):
        namesarehard = np.transpose(x)
        namesarehard = namesarehard[i]
        someshit = np.dot(np.transpose(err),namesarehard)
        theWalrus = W[i]
        grad_weights[i] = (2/len(y))*(someshit[0])+theWalrus*(reg/2)/np.linalg.norm(W,2)
    return grad_weights, grad_bias

def acc(W,b,x,y):
    predictions = np.dot(x,W)+b
    accuracy = 0.0
    for i in range(0,len(predictions)):
        if predictions[i][0] >= 0.5 and y[i][0] == 1:
            accuracy += 1.0
        elif predictions[i][0] < 0.5 and y[i][0] == 0:
            accuracy += 1.0
    accuracy = accuracy/len(y)
    return accuracy

def grad_descent(W, b, x, y, val_x, val_y, test_x, test_y, alpha, epochs, reg, error_tol,lossType="CE"):
    # Your implementation here
    if (lossType=="MSE"):
        weights = W
        bias = b
        losses = []
        accs = []
        val_losses = []
        val_accs = []
        test_losses = []
        test_accs = []
        iterations = []
        for i in range(0,epochs):
            print("epoch: ",i)
            loss = MSE(weights,bias,x,y,reg)
            accuracy = acc(weights,bias,x,y)
            print("training loss = ",loss)
            print("training acc = ",accuracy)
            grad_weights, grad_bias = grad_MSE(weights,bias,x,y,reg)
            for j in range(0,len(grad_weights)):
                if abs(alpha*grad_weights[j]) > error_tol:
                    weights[j] = weights[j] - alpha*grad_weights[j]
                if abs(alpha*grad_bias) > error_tol:
                    bias = bias - alpha*grad_bias
                if error_tol >= abs(max(np.amax(grad_weights),grad_bias)):
                    break
            val_loss = MSE(weights,bias,val_x,val_y,reg)
            print("validation loss = ",val_loss)
            val_acc = acc(weights,bias,val_x,val_y)
            print("validation accuracy = ",val_acc)
            iterations.append(i)
            losses.append(loss)
            accs.append(accuracy)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_loss = MSE(weights,bias,test_x,test_y,reg)
            test_acc = acc(weights,bias,test_x,test_y)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            print("test_loss = ",MSE(weights,bias,test_x,test_y,reg))
            print("test_acc = ",acc(weights,bias,test_x,test_y))
        return weights, bias, losses, accs, val_losses, val_accs, test_losses, test_accs

    else: # LossType = CE
        # x = np.transpose(x)
        # val_x = np.transpose(val_x)
        # test_x = np.transpose(test_x)
        # print(x.shape,val_x.shape,test_x.shape)

        trainLossPlot = []
        validLossPlot = []
        testLossPlot = []

        validAccPlot = []
        trainAccPlot = []
        testAccPlot = []

        for e in range(epochs):
            print(e)

            dW, db = gradCE(W, b, x, y, reg)
            trainLossPlot += [crossEntropyLoss(W, b, x, y, reg)]
            validLossPlot += [crossEntropyLoss(W, b, val_x, val_y, reg)]
            testLossPlot += [crossEntropyLoss(W, b, test_x, test_y, reg)]

            trainAccPlot += [acc(W, b, x, y)]
            validAccPlot += [acc(W, b, val_x, val_y)]
            testAccPlot += [acc(W, b, test_x, test_y)]

            W = W - dW * alpha
            b = b - db * alpha

            # for i in range(len(W)):
            #     if (abs(alpha * dW[i]) > error_tol):
            #         W = W - dW * alpha
            # if (abs(alpha * db) > error_tol):
            #     b = b - db * alpha
            # if error_tol >= abs(max(np.amax(dW), db)):
            #     break

        plt.plot(trainLossPlot, 'b--', label="Train")
        plt.plot(validLossPlot, 'r--', label="Valid")
        plt.plot(testLossPlot, 'g--', label="Test")
        plt.title("Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(trainAccPlot, 'b--', label="Train")
        plt.plot(validAccPlot, 'r--', label="Valid")
        plt.plot(testAccPlot, 'g--', label="Test")
        plt.title("Accuracy vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        return W, b, trainLossPlot, trainAccPlot, validLossPlot, validAccPlot, testLossPlot, testAccPlot


def sigmoid (z):
    return 1.0/(1.0+np.exp(-z))

def crossEntropyLoss(W, b, x, y, reg):
    # print(W.shape, b, x.shape,y.shape, reg)
    # x = np.transpose(x)
    N = y.shape[0]
    epsilon = 1e-5
    twonorm = 0
    for n in range(W.shape[0]):  # W.shape = (784,1)
        twonorm += W[n] ** 2
    twonorm = twonorm[0]

    yhat = sigmoid(np.dot(x,W) + b)
    lg = np.log(yhat)
    lgminus = np.log(1 - yhat + epsilon)
    ans = (1 / N * np.sum(-(y * lg + (1 - y) * lgminus))) + twonorm * reg / 2
    return ans
    # Your implementation here

def gradCE(W, b, x, y, reg):
    # x = np.transpose(x)
    # W = np.transpose(W)
    print(x.shape, W.shape)
    yhat = sigmoid(np.dot(x,W) + b)

    dW = np.matmul(np.transpose(x), (yhat - y)) / (np.shape(y)[0]) + 2 * reg * W
    db = np.sum((yhat - y)) / (np.shape(y)[0])
    return dW, db
    # Your implementation here

def buildGraph(loss="MSE"):
	#Initialize weight and bias tensors
	tf.set_random_seed(421)

	# if loss == "MSE":
    #     pass
	# # Your implementation
	
	# elif loss == "CE":
    #     pass
	#Your implementation here

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainData = np.reshape(trainData,(trainData.shape[0],trainData.shape[1]*trainData.shape[2]))
validData = np.reshape(validData,(validData.shape[0],validData.shape[1]*validData.shape[2]))
testData = np.reshape(testData,(testData.shape[0],(testData.shape[1]*testData.shape[2])))

sample_weights = np.zeros(784)
for i in range(0,len(sample_weights)):
    sample_weights[i] = np.random.randint(-10,10)

sample_weights = np.expand_dims(sample_weights,1)

sample_bias = np.random.randint(-10,10)
# print(trainData.shape, validData.shape, testData.shape, sample_weights.shape)

# start = time()
weights, bias, trainingLosses, trainingAccs, valLosses, valAccs, testLoss, testAcc = grad_descent(sample_weights,sample_bias,trainData,trainTarget,validData,validTarget,testData,testTarget,0.005,500,0,1.0/(10**7))
# print(time() - start)

# plt.subplot(2,1,1)
# plt.plot(iterations,trainingLosses)
# plt.plot(iterations,valLosses)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("learning rate = 0.005")

# # plt.subplot(2,1,2)
# # plt.plot(iterations,trainingAccs)
# # plt.plot(iterations,valAccs)
# # plt.legend(["training","validation"])
# # plt.xlabel("epoch")
# # plt.ylabel("accuracy")
# plt.show()

# weights, bias, trainingLosses, trainingAccs, valLosses, valAccs, testLoss, testAcc, iterations = grad_descent(sample_weights,sample_bias,trainData,trainTarget,validData,validTarget,testData,testTarget,0.001,5000,0,1.0/(10**7))

# plt.subplot(2,1,1)
# plt.plot(iterations,trainingLosses)
# plt.plot(iterations,valLosses)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("learning rate = 0.001")

# plt.subplot(2,1,2)
# plt.plot(iterations,trainingAccs)
# plt.plot(iterations,valAccs)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()

# weights, bias, trainingLosses, trainingAccs, valLosses, valAccs, testLoss, testAcc, iterations = grad_descent(sample_weights,sample_bias,trainData,trainTarget,validData,validTarget,testData,testTarget,0.0001,5000,0,1.0/(10**7))

# # plt.subplot(2,1,1)
# plt.plot(iterations,trainingLosses)
# plt.plot(iterations,valLosses)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("learning rate = 0.0001")

# plt.subplot(2,1,2)
# plt.plot(iterations,trainingAccs)
# plt.plot(iterations,valAccs)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()

# weights, bias, trainingLosses, trainingAccs, valLosses, valAccs, testLoss, testAcc, iterations = grad_descent(sample_weights,sample_bias,trainData,trainTarget,validData,validTarget,testData,testTarget,0.005,5000,0.001,1.0/(10**7))

# # plt.subplot(2,1,1)
# plt.plot(iterations,trainingLosses)
# plt.plot(iterations,valLosses)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("reg = 0.001")

# plt.subplot(2,1,2)
# plt.plot(iterations,trainingAccs)
# plt.plot(iterations,valAccs)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()

# weights, bias, trainingLosses, trainingAccs, valLosses, valAccs, testLoss, testAcc, iterations = grad_descent(sample_weights,sample_bias,trainData,trainTarget,validData,validTarget,testData,testTarget,0.005,5000,0.1,1.0/(10**7))

# # plt.subplot(2,1,1)
# plt.plot(iterations,trainingLosses)
# plt.plot(iterations,valLosses)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("reg = 0.1")

# plt.subplot(2,1,2)
# plt.plot(iterations,trainingAccs)
# plt.plot(iterations,valAccs)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.show()

# weights, bias, trainingLosses, trainingAccs, valLosses, valAccs, testLoss, testAcc, iterations = grad_descent(sample_weights,sample_bias,trainData,trainTarget,validData,validTarget,testData,testTarget,0.005,5000,0.5,1.0/(10**7))

# # plt.subplot(2,1,1)
# plt.plot(iterations,trainingLosses)
# plt.plot(iterations,valLosses)
# plt.legend(["training","validation"])
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("reg = 0.5")

# # plt.subplot(2,1,2)
# # plt.plot(iterations,trainingAccs)
# # plt.plot(iterations,valAccs)
# # plt.legend(["training","validation"])
# # plt.xlabel("epoch")
# # plt.ylabel("accuracy")
# plt.show()

def findOptimumWeights(x,y,reg):
    x_T = np.transpose(x)
    print(x_T.shape)
    print(x.shape)
    dotp = np.dot(x_T,x)
    inverse = np.linalg.inv(dotp+reg)
    print(inverse.shape)
    adescriptivename = np.dot(inverse,x_T)
    print(adescriptivename.shape)
    print(y.shape)
    weights = np.dot(np.transpose(adescriptivename),y)
    bias = y - np.dot(np.transpose(weights),x)
    return np.transpose(weights), bias

start = time()
optimumWeights, optimumBias = findOptimumWeights(trainData,trainTarget,0)
print(time() - start)
# print("optimum loss = ",MSE(optimumWeights,optimumBias,trainData,trainTarget,0))
# print("optimum accuracy = ",acc(optimumWeights,optimumBias,trainData,trainTarget))
