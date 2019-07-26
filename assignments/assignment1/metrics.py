def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0


    T = ground_truth[prediction]
    TP = len(T[T])

    F = ground_truth[prediction] == False
    FP = len(F[F]) 

    N = ground_truth[prediction == False]
    FN = len(N[N])
    #print('True Posit = ',TP,'False Posit = ',FP, 'False Negative = ',FN )
    precision = TP /(FP + TP)

    recall = TP / (TP + FN)

    accuracy = len(ground_truth[ground_truth == prediction])/len(ground_truth)
    

    f1 = (2*precision*recall)/(precision+recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = len(ground_truth[ground_truth == prediction])/len(ground_truth)
    # TODO: Implement computing accuracy
    return accuracy
