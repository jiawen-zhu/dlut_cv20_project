import numpy as np
import pdb


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match. Confidences must be descended. （matches is the corresponding
            sorted index)

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################

    dis = np.zeros((features1.shape[0],features2.shape[0]))
    for i in range(features1.shape[0]):
        for j in range(features2.shape[0]):
            dis[i,j] = np.linalg.norm(features1[i] - features2[j])
    matches = np.zeros((features1.shape[0], 2))
    matches[:,0], matches[:,1]  = np.arange(features1.shape[0]), dis.argmin(axis=1)
    dis.sort(axis=1)
    confidences = 1 - dis[:,0]/dis[:,1]
    sort = np.argsort(1 - confidences)
    matches = matches[sort,:]


    # raise NotImplementedError('`match_features` function in ' +
    #     '`student_feature_matching.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches.astype('int64'), confidences
