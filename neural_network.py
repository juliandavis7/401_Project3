from funcs import *

class NeuralNetwork:
    
    def __init__(self, layers=None, nodes=None, nnodes=None,
                 activations=[], activationFn="relu", batchSize=50,
                 lr=.001, lr_type="constant", power_t=.5,
                 annealing_rate=.999, max_epoch=200, momentum=.9,
                 tol=0.0001, alpha=.0001, shuffle=False,
                 early_stopping=False, num_epochs_stop=50, verbose=True):
        
        if layers != None:
            self.layers = layers # total number of hidden layers
        else:
            self.layers = len(nodes)
        # an int array of size [0, ..., Layers + 1] Nodes[0] shall represent the input size (typically 50) 
        # Nodes[Layers + 1] shall represent the output size (typically 1) all other Nodes represent the number of 
        # nodes (or width) in the hidden layer i
        self.nodes = nodes
        if nodes != None:
            self.nodes.insert(0, batchSize)
            self.nodes.append(1)
        
        # alternative to nodes where each hidden layer of the nueral network is the same size
        self.nnodes = nnodes
        if nnodes != None:
            self.nodes = []
            self.nodes.append(batchSize)
            for i in range(layers):
                self.nodes.append(nnodes)
            self.nodes.append(1)
        
        # activations[i] values are labels indicating the activation function used in layer i
        self.activations = activations
        self.activationFn = activationFn
        if activationFn != "":
            self.activations = [activationFn] * self.layers
        
        self.batchSize = batchSize
        self.lr = lr
        self.lr_type = lr_type
        self.power_t = power_t
        self.annealing_rate = annealing_rate
        self.max_epoch = max_epoch
        self.mu = momentum
        self.tol = tol
        self.alpha = alpha
        self.shuffle = shuffle
        self.verbose = verbose
        
        if early_stopping == False:
            self.num_epochs_stop = max_epoch
        else:
            self.num_epochs_stop = num_epochs_stop
    
        self.layer_values = [None] * (self.layers + 2)
        self.iters = 0
        self.epochs = 0
                
    def validateHyperParams(self):
        
        if self.layers != (len(self.nodes) - 2):
            raise ValueError("layers must be equal to the number of hidden layers, got %s." % self.layers)
        if self.nnodes != None and self.nnodes <= 0:
            raise ValueError("nnodes must be > 0, got %s." % self.nnodes)
        if self.lr <= 0 or self.lr > 1:
            raise ValueError("lr must be in (0, 1], got %s." % self.lr)
            
        if self.lr_type not in ["constant", "invscaling", "annealing", "adaptive"]:
            raise ValueError("lr_type is not valid" % self.lr_type
                            + "\nAvailable lr types: constant, invscaling, adaptive")
            
        if self.max_epoch <= 0:
            raise ValueError("max_iter must be > 0, got %s." % self.max_epoch)
               
        activation_functions = list(ACTIVATIONS.keys())
        if self.activationFn != "":
            if self.activationFn not in activation_functions:
                raise ValueError("%s is not an activation function" % self.activationFn
                                + "\nAvailable activation functions: relu, leaky_relu, sigmoid, tanh")
    
    def initialize_weights(self, M):
        weights = []
        
        for i in range(self.layers + 1):
            if i == 0:
                input_size = M # special case for w1
            else:
                input_size = self.nodes[i]
            output_size = self.nodes[i + 1]
            
            # Xavier (Glorot) Initialization
            if self.activationFn == "tanh":
                target_variance = 2 / (input_size + output_size)
                w_i = np.random.normal(loc= 0, scale = np.sqrt(target_variance), size=(input_size, output_size))
            # He Initialization
            elif self.activationFn == "relu":
                target_variance = 2 / input_size
                w_i = np.random.normal(loc= 0, scale = np.sqrt(target_variance), size=(input_size, output_size))
            # Random Uniform
            else:
                w_i = np.random.uniform(-1/np.sqrt(input_size), 1/np.sqrt(input_size))
                #w_i = np.random.normal(size=(input_size, output_size))
            w_i = np.round(w_i, 2)
            w_i[input_size - 1:] = 0 # initialize bias to 0
            weights.append(w_i)
        return weights
    
    # returns the weight term for L2 regularization
    def get_weight_term(self):
        weight_term = 0
        for i in range(len(self.weights)):
            weight_term = np.sum(self.weights[i] ** 2)
        return weight_term
        
    def forward_pass(self, X_batch, y_batch):
        
        self.layer_values[0] = X_batch
        
        # calculate hidden layers
        for i in range(self.layers):
            X = self.layer_values[i]
            weights = self.weights[i]
            h_layer = X.dot(weights)
            
            # apply activation function
            activation_fn = ACTIVATIONS[self.activations[i]]
            activation_fn(h_layer)
            self.layer_values[i + 1] = h_layer
            
        
        # calculate predictions
        X = self.layer_values[self.layers] # values in last hidden layer
        weights = self.weights[self.layers]
        y_pred = X.dot(weights)
        y_pred = y_pred.flatten()
        
        # calculate the l2 loss
        l2_loss = 0
        # only need predictions once we have fit the data
        if isinstance(y_batch, np.ndarray):
            l2_loss = squared_loss(y_pred, y_batch) # l2
            weight_term = self.get_weight_term()
            l2_loss += self.alpha * weight_term # l2 regularization
            self.layer_values[self.layers + 1] = l2_loss
        
        return l2_loss, y_pred
    
    
    def backward_pass(self, y_pred, y_batch):
        
        # loss layer
        J = squared_loss_derivative(y_pred, y_batch, self.batchSize)
        J = np.reshape(J, (len(J), 1))
        
        J_weights = [None] * (self.layers + 1)
        
        # output layer jacobian w.r.t. weights
        x_t = self.layer_values[self.layers].T
        J_wi = x_t.dot(J)
        J_weights[self.layers] = J_wi
        
        # update jacobian at output layer
        w_t = self.weights[self.layers].T
        w_t = np.delete(w_t, w_t.shape[1] - 1, 1) # take out the bias
        J = np.dot(J, w_t)
        zeros = [0] * len(J)
        zeros = np.reshape(zeros, (len(J), 1))
        J = np.append(J, zeros, axis=1)
        
        # iterate through hidden layers backwards
        for i in range(self.layers, 0 , -1):
            # update jacobian at activation layer
            d_activation_fn = DERIVATIVES[self.activations[i - 1]]
            d_activation_fn(self.layer_values[i], J)
            
            # hidden layer jacobian w.r.t. weights
            x_t = self.layer_values[i - 1].T
            J_wi = x_t.dot(J)
            J_weights[i - 1] = J_wi
            
            # jacobian w.r.t. inputs
            w_t = self.weights[i - 1].T
            w_t = np.delete(w_t, w_t.shape[1] - 1, 1)
            J = np.dot(J, w_t)
            zeros = [0] * len(J)
            zeros = np.reshape(zeros, (len(J), 1))
            J = np.append(J, zeros, axis=1)
            
            
        # initialize velocity to 0
        if self.epochs == 0 and self.iters == 0:
            self.velocity = []
            for i in range(len(J_weights)):
                n_rows = J_weights[i].shape[0]
                n_cols = J_weights[i].shape[1]
                vel_i = np.zeros((n_rows, n_cols))
                self.velocity.append(vel_i)
        
        for i in range(len(J_weights)):
            self.velocity[i] = self.mu * self.velocity[i] - self.lr * J_weights[i]
            self.weights[i] += self.velocity[i]
      
    
    def fit(self, X_train, y_train):
        
        self.validateHyperParams()
        # convert to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
            
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
            
        # add ones for bias
        ones = [1] * len(X_train)
        ones = np.reshape(ones, (len(X_train), 1))
        X_train = np.append(X_train, ones, axis=1)
        
        # save 10% for validation
        val_rows = round(len(X_train) * .1)
        X_val = X_train[:val_rows, :]
        y_val = y_train[:val_rows]
        
        X_train = X_train[val_rows:, :]
        y_train = y_train[val_rows:]
        
        # initalize weights on first iteration
        M = X_train.shape[1] # M = number of features
        self.weights = self.initialize_weights(M)
        
        best_v_loss = np.inf
        n_epoch_no_change = 0
        while (self.epochs < self.max_epoch and n_epoch_no_change <= self.num_epochs_stop):
            # ONE EPOCH
            last_idx = 0
            if self.shuffle == True: # shuffle data after every epoch, if specified
                np.random.shuffle(X_train)
            while (last_idx < len(X_train)):
                first_idx = self.iters * self.batchSize
                remaining_rows = len(X_train) - first_idx
                last_idx = first_idx + min(self.batchSize, remaining_rows)
                X_batch = X_train[first_idx: last_idx, :]
                y_batch = y_train[first_idx: last_idx]
                loss, y_pred = self.forward_pass(X_batch, y_batch)
                self.backward_pass(y_pred, y_batch)
                self.iters += 1
            
            # trainig and validation loss after one epoch
            t_loss, y_pred = self.forward_pass(X_train, y_train)
            v_loss, y_pred = self.forward_pass(X_val, y_val)
            if self.verbose:
                print("epoch:", self.epochs)
                print("training loss:", t_loss)
                print("validation loss:", v_loss)
            
            self.iters = 0 # start over, next epoch
            self.epochs += 1
            
            # decrease the learning rate by one of three methods, if specified
            if self.lr_type == "invscaling":
                self.lr = self.lr/pow(self.epochs, self.power_t)
            elif self.lr_type == "annealing":
                self.lr = self.lr * self.annealing_rate
            elif self.lr_type == "adaptive":
                if n_epoch_no_change >= 2:
                    self.lr = self.lr/5
                
            # stops when validation loss doesn't improve for num_epochs_stop
            if best_v_loss - v_loss < self.tol:
                n_epoch_no_change += 1
            else:
                n_epoch_no_change = 0
            # update best_v_loss
            if v_loss < best_v_loss:
                best_v_loss = v_loss
            
            
        
    def predict(self, X_test):
        
        # convert to numpy array
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        
        # add ones for bias
        ones = [1] * len(X_test)
        ones = np.reshape(ones, (len(X_test), 1))
        X_test = np.append(X_test, ones, axis=1)
        
        loss, y_pred = self.forward_pass(X_test, None)
        return y_pred
        
