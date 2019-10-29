import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    s_pred =predictions-np.max(predictions)
    probs = np.exp(s_pred)/np.sum(np.exp(s_pred))
    # pp('soft max =', probs)
    return probs
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    
    log =   np.sum(- np.log(probs[np.arange(probs.shape[0]),target_index.flatten()]))
    pp ('log loss = ',log)
    return log
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    # pp('enter of L2 ',W.shape,reg_strength)
    n_classes = W.shape[1]

    loss = reg_strength * np.sum(W**2)
    loss/=n_classes
    grad = reg_strength*W
    # pp('L2 loss, grad = ',loss,grad)
    return loss, grad


    # TODO: Copy from the previous assignment
    

def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    n_samples = predictions.shape[0]  
    n_featerus = predictions.shape[1] if len(predictions.shape)>1 else 0
    # pp('enter of the function = ',predictions, target_index)
    
    softmax_ = softmax(predictions)
    
    loss = cross_entropy_loss(softmax_,target_index)
    loss = loss/n_samples
    prediction = softmax_.copy()
    prediction[np.arange(n_samples),target_index.ravel()]-=1/n_samples
    
    # pp('=N_SUMPLES=',n_samples)
    # prediction/=n_samples
    # pp('loss , grand (prediction) = ',loss, prediction)
    return loss, prediction


    # TODO: Copy from the previous assignment
    


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)



class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.maximum(0,X)

        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return np.where(self.X>=0,1,0)*d_out


        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        
    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return self.X.dot(self.W.value)+self.B.value

        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        d_input = d_out@self.W.value.T
        d_W = self.X.T@d_out
        d_B = np.sum(d_out,axis=0)

        self.W.grad += d_W
        self.B.grad += d_B

        
        return d_input

        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment


    def params(self):
        return {'W': self.W, 'B': self.B}


def fmt_items(lines,max_lines=0):
    max_width=max([len(line)for line in lines])
    empty =' '*max_width
    lines = [line.ljust(max_width)for line in lines]
    lines += [empty]*(max_lines - len(lines))
    return lines
    
def pp (*list):
    lines = [ str(item).split('\n') for item in list]
    max_lines=max([len(item)for  item in lines])
    lines = [fmt_items(item,max_lines=max_lines)for item in lines]
    lines_t= np.array(lines).T
    print('\n'.join([' '.join(line) for  line in lines_t]))