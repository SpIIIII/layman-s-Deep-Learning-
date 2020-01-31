import numpy as np

def softmax_(predictions):
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
    # TODO implement softmax_
    # Your final implementation shouldn't have any loops


def cross_entropy_loss_(probs, target_index):
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
    return log
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops

def l2_regularization_(W, reg_strength):
    """
      Computes L2 regularization loss on weights and its gradient

      Arguments:
        W, np array - weights
        reg_strength - float value

      Returns:
        loss, single value - l2 regularization loss
        gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    n_classes = W.value.shape[1]

    loss = reg_strength * np.sum(W.value**2)
    loss/=n_classes
    grad = reg_strength*W.value
    # pp('L2 loss, grad = ',loss,grad)
    return loss, grad


    # TODO: Copy from the previous assignment
    

def softmax__with_cross_entropy_(predictions, target_index):
    """
      Computes softmax_ and cross-entropy loss for model predictions,
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
    
    softmax__ = softmax_(predictions)
    
    loss = cross_entropy_loss_(softmax__,target_index)
    loss = loss/n_samples
    prediction = softmax__.copy()
    prediction[np.arange(n_samples),target_index.ravel()] -= 1/n_samples
    
    # pp('=N_SUMPLES=',n_samples)
    # prediction/=n_samples
    # pp('loss , grand (prediction) = ',loss, prediction)
    return loss, prediction


    # TODO: Copy from the previous assignment