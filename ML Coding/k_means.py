# k-means

# The K-means algorithm is a method to automatically cluster similar data points together.
#
# Concretely, you are given a training set  {𝑥(1),...,𝑥(𝑚)}
#  , and you want to group the data into a few cohesive “clusters”.
# K-means is an iterative procedure that
# Starts by guessing the initial centroids, and then
# Refines this guess by
# Repeatedly assigning examples to their closest centroids, and then
# Recomputing the centroids based on the assignments.
# In pseudocode, the K-means algorithm is as follows:
#
#   # Initialize centroids
#   # K is the number of clusters
#   centroids = kMeans_init_centroids(X, K)
#
#   for iter in range(iterations):
#       # Cluster assignment step:
#       # Assign each data point to the closest centroid.
#       # idx[i] corresponds to the index of the centroid
#       # assigned to example i
#       idx = find_closest_centroids(X, centroids)
#
#       # Move centroid step:
#       # Compute means based on centroid assignments
#       centroids = compute_centroids(X, idx, K)
# The inner-loop of the algorithm repeatedly carries out two steps:
# Assigning each training example  𝑥(𝑖)
#   to its closest centroid, and
# Recomputing the mean of each centroid using the points assigned to it.
# The  𝐾
#  -means algorithm will always converge to some final set of means for the centroids.
#
# However, the converged solution may not always be ideal and depends on the initial setting of the centroids.
#
# Therefore, in practice the K-means algorithm is usually run a few times with different random initializations.
# One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).


# 1.1 Finding closest centroids
# In the “cluster assignment” phase of the K-means algorithm, the algorithm assigns every training example  𝑥(𝑖)
#   to its closest centroid, given the current positions of centroids.


# Your task is to complete the code in find_closest_centroids.
#

def find_closest_centroids(X, centroids):
    """
    Compute the centroid memberships for every examples

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m, ) closest centroids
    """

    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distance = []
        for j in range(K):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)

    return idx



# 1.2 Computing centroid means
# Given assignments of every point to a centroid, the second phase of the algorithm recomputes, for each centroid, the mean of the points that were assigned to it.

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m, ) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    m, n = X.shape

    centroids = np.zeros((K, n))

    for k in range(K):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)

    return centroids



def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = initial_centroids
    idx = np.zeros(m)
