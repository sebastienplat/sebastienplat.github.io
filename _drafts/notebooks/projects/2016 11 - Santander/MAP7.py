
import numpy as np
import pandas as pd

def test():
    
    print('hello')
    
    
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

    
def mapk(actual, predicted, checkFile, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    
    results = [apk(a,p,k) for a,p in zip(actual, predicted)]
    results_pd  = pd.DataFrame({'actual': actual,'predicted': predicted,'results': results})
    
    #actual_lengths = np.array([len(x) for x in actual])
    #results_pd  = results_pd.loc[actual_lengths!=0]
    if checkFile != False:
        results_pd.to_csv('model_results_' + checkFile + '.csv',index=False)
    
    return np.mean(results)
    
    
def run_map7(actual, predicted, checkFile=False):
    
    print('run MAP7')
    k=7
    predicted = [x.split() for x in predicted]
    print(mapk(actual, predicted, checkFile, k))
    
    