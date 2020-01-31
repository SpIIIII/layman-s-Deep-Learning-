def multiclass_accuracy(prediction, ground_truth):
    """
        Computes metrics for multiclass classification

        Arguments:
        prediction, np array of int (num_samples) - model predictions
        ground_truth, np array of int (num_samples) - true labels

        Returns:
        accuracy - ratio of accurate predictions to total samples
    """
    print('ACCURACY', prediction[:10], ground_truth[:10])

    # TODO: Implement computing accuracy
    accuracy = ground_truth[ground_truth == prediction].shape[0]/ground_truth.shape[0]

    return accuracy
