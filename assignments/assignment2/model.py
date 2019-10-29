import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, pp


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization ngth
        """
        self.reg = reg
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size
        # # TODO Create necessary layers
        # raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        

        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")
        self.X = X
                
        self.first_layer = FullyConnectedLayer(self.n_input,self.hidden_layer_size)
        self.first_relu = ReLULayer()

        self.second_layer = FullyConnectedLayer(self.hidden_layer_size,self.n_output)
        self.second_relu = ReLULayer()


        first_layer_f_out = self.first_layer.forward(self.X)
        first_relu_f_out = self.first_relu.forward(first_layer_f_out)

        second_layer_f_out = self.second_layer.forward(first_relu_f_out)
        second_relu_f_out = self.second_relu.forward(second_layer_f_out)

        loss, grad = softmax_with_cross_entropy(second_relu_f_out,y)

        
        second_relu_b_out = self.second_relu.backward(loss)
        second_relu_param = self.second_relu.params()
        self.result['second_relu_param']=second_relu_param
        second_layer_b_out = self.second_layer.backward(second_relu_b_out)
        second_layer_param = self.second_layer.params()
        self.result['second_layer_param']=second_layer_param

        first_relu_b_out = self.first_relu.backward(second_layer_b_out)
        first_relu_param = self.first_relu.params()
        self.result['first_relu_param']=first_relu_param
        first_layer_b_out = self.first_layer.backward(first_relu_b_out)
        first_layer_param = self.first_layer.params()
        self.result['first_layer_param']=first_layer_param


        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # for key in self.result:
        #     self.result[key] += l2_regularization(self.result[key],self.reg)[1]
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        
        pred = np.zeros(X.shape[0], np.int)
        for key in self.result:
            X = X.dot(self.result[key])
        
        
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        
        
        return pred

    def params(self):
        result = {}
        return result
