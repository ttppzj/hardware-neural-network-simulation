import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(18,self.input.shape[0]) 
        self.weights2   = np.random.rand(3,18)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.weights1,self.input))
        self.output = sigmoid(np.dot(self.weights2,self.layer1))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    def matrixtune1(self):
        #automaticly tune each compont of matrix with a small shift to compare if output make costfunction smaller
        [rows, cols] = self.weights1.shape
        for i in range(rows):
            for j in range(cols):
                temp1=self.weights1[i,j]
                temp2=nn.output
                self.weights1[i,j]=self.weights1[i,j]+delta
                nn.feedforward()
                if cross_entropy(nn.output, y)<cross_entropy(temp2, y):
                     self.weights1[i,j]=temp1
                else:
                    self.weights1[i,j]=self.weights1[i,j]
    def matrixtune2(self): 
        #automaticly tune each compont of matrix with a small shift to compare if output make costfunction smaller
        [rows, cols] = self.weights2.shape
        for i in range(rows):
            for j in range(cols):
                temp1=self.weights2[i,j]
                temp2=nn.output
                self.weights2[i,j]=self.weights2[i,j]+delta
                nn.feedforward()
                if cross_entropy(nn.output, y)<cross_entropy(temp2, y):
                     self.weights2[i,j]=self.weights2[i,j]
                else:
                    self.weights2[i,j]=temp1-2*delta
                    

if __name__ == "__main__":
    X = np.array([[0,0,1,
                  0,1,1,
                  1,0,1,
                  ]]).T
    y = np.array([[1],[0],[0]])
    delta=0.01
    nn = NeuralNetwork(X,y)
    
    for i in range(1500):      
        #nn.backprop()
        nn.matrixtune1()
        nn.matrixtune2()
        

    print(nn.output)
