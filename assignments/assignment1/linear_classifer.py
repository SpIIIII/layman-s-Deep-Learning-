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
    pp('soft max =', probs)
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
    pp ('log = ',log)
    return log
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      prediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    n_samples = predictions.shape[0]  
    n_featerus = predictions.shape[1] if len(predictions.shape)>1 else 0
    pp('enter of the function = ',predictions, target_index)
    
    softmax_ = softmax(predictions)
    
    loss = cross_entropy_loss(softmax_,target_index)
    loss = loss/n_samples
    prediction = softmax_.copy()
    prediction[np.arange(n_samples),target_index.ravel()]-=1/n_samples
    
    pp('=N_SUMPLES=',n_samples)
    # prediction/=n_samples
    pp('loss , grand (prediction) = ',loss ,prediction,'\n')
    return loss, prediction
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    pp('enter of L2 ',W.shape,reg_strength)
    n_classes = W.shape[1]

    loss = reg_strength * np.sum(W**2)
    loss/=n_classes
    grad = reg_strength*W
    pp('L2 loss, grad = ',loss,grad)
    return loss, grad
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss
  
    '''
    predictions = np.dot(X, W)

    loss, grad = softmax_with_cross_entropy(predictions,target_index)


    n_samples = predictions.shape[0]

    # n_features = predictions.shape[1]
    # pp('enter of linear_softmax',X,W,target_index)
    # grad = softmax(predictions)
    # loss = cross_entropy_loss(grad,target_index)


    dW = X.T.dot(grad)
    
    
    pp('loss , grand (prediction), grad by W = ',loss ,grad,dW,'\n')

    return loss, dW
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          0, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")

            # end

            X=X[np.array(batches_indices).ravel()]
            y=y[np.array(batches_indices).ravel()]
            
            loss,dW = linear_softmax(X, self.W, y)
            
            loss_l2,dW_l2 =l2_regularization(self.W,reg)
            loss+=loss_l2
            dW+=dW_l2
            
            loss_history.append(loss)
            pp("Epoch %i, loss: %f" % (epoch, loss))
            self.W += dW*learning_rate

            print('== W ==',np.sum(self.W))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        predicitons = X@self.W
        
        y_pred = np.zeros(X.shape[0], dtype=np.int8)
        y_pred = predicitons.argmax(axis=1)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops

        return y_pred

def softmax_derivative(q):
    x =softmax(q)
    s = x.reshape(-1,1)
    return (np.diagflat(s) - np.dot(s, s.T))

# def exper(soft,target,n_feat:int=None,n_sempl:int=None):
#     pp(soft)
#     soft[np.arange(n_sempl),target]-=1
#     pp(soft)
#     return soft
                
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

            

                
