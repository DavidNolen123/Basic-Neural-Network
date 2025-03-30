
#code from michael nielson
def ___init___(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.(y, x)
                    for  x, y in zinp(sizes[:-1], sizes[1:])]


net = Network[10,10,10]

def sigmoid(thing):
    return 1.0/1.0+np.exp(-thing))

def feedforward(self, a):
    #appliing the function forward to get the thing times weights plus bias, all in sigmoid
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
        return a


#stochastic: (!)
def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    #will use "mini batch"
    if test_data: n_test = len(test_data)
    n = len(training_data)#to check progress
    
    for j in xrange(epochs):#where did EPOCHs come from??
        random
    
