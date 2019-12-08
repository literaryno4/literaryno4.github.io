---
layout: post
title:  "CS231n Assignments Note1: K-Nearest Neighbor Classifier and Cross Validation"
categories: jekyll update
---

- [K-Nearest Neighbor](#knn)
- [Cross Validation](#cv)
- [Reference](#rf)

<a name='knn'></a>

## K-Nearest Neighbor Classifier(kNN)

kNN is a really simple way in image classification. However, in practice, few of people use it because of the low accuracy and costing too much time when testing. Here is how I make sense of kNN.

We compare the images pixel by pixel and add up all the differeces. Given two images as vectors $ I_1 $, $ I_2 $, one way of comparing them is the **L1 distence**:

$$
d_1 (I_1, I_2) = \sum_{p} \left| I^p_1 - I^p_2 \right|
$$

one other way of computing distance is **L2 distance**:

$$
d_2 (I_1, I_2) = \sqrt{\sum_{p} \left( I^p_1 - I^p_2 \right)^2}
$$

Take a test image as an example, we compute the distance between the test image and each training image. Find the top k nearest images, and the most common lable in the k images is our prediction.

Using <a href="http://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10 dataset</a>, I have done some experiments. Here is  an implement of L2 distance with one loop:

```python
def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i,:]) ,axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```

I have not figure out the implement way without any loops,which is extremely fast. Then I find this [blog](https://www.cnblogs.com/GeekDanny/p/10179251.html), the math make me confused but the code works:

```python
def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        M = np.dot(X, self.X_train.T)
        nrow=M.shape[0]
        ncol=M.shape[1]
        te = np.diag(np.dot(X,X.T))
        tr = np.diag(np.dot(self.X_train,self.X_train.T))
        te= np.reshape(np.repeat(te,ncol),M.shape)
        tr = np.reshape(np.repeat(tr, nrow), M.T.shape)
        sq=-2 * M +te+tr.T
        dists = np.sqrt(sq)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
```

The math behind is what I should try to understand.

<a name='cv'></a>

## Cross Validation

Besides training sets and test sets, we use **validation sets** for **hyperparameter** tuning. It is different from test sets noting that:

> Evaluate on the test set only a single time, at the very end.

A sophisticated technique for hyperparameter tuning call **cross-validation**. The idea is that we would split the training data into equal folds, say 5, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.

In my exercise, I implement it with this ugly but useful code:

```python

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

classifier_cv = KNearestNeighbor()
num_train_vs = y_train_folds[0].shape[0]

for k_chosen in k_choices:
    
    # a list to store num_folds accuracy
    accuracy_list = []
    for i in range(num_folds):
        # Prepare training data and validation set
        #X_train_cp = X_train_folds[:]
        #y_train_cp = y_train_folds[:]
        X_train_vs = X_train_folds[i]
        # X_train_cp.remove(X_train_cp[i])
        #X_train_cp.pop(i)
        X_train_td = np.array(X_train_folds[:i] + X_train_folds[i+1:])
        X_train_td = X_train_td.reshape(X_train_td.shape[0]*X_train_td.shape[1], X_train_td.shape[2])
        #print(X_train_td.shape)
        y_train_vs = y_train_folds[i]
        # y_train_cp.remove(y_train_cp[i])
        #y_train_cp.pop(i)
        y_train_td = np.array(y_train_folds[:i] + y_train_folds[i+1:])
        y_train_td = y_train_td.reshape(y_train_td.shape[0]*y_train_td.shape[1],)# y_train_td.shape[2])
        #print(y_train_td.shape)
        
        # cal the accuracy
        classifier_cv.train(X_train_td, y_train_td)
        dists = classifier_cv.compute_distances_no_loops(X_train_vs)
        y_train_vs_pred = classifier.predict_labels(dists, k=k_chosen)
        num_correct = np.sum(y_train_vs_pred == y_train_vs)
        accuracy = float(num_correct) / num_train_vs
        accuracy_list.append(accuracy)
        #print(accuracy)
        
    # store the accuracy list into dic
    k_to_accuracies[k_chosen] = accuracy_list
        
    

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```
when debuging, I find that removing numpy array elements from list does not work if using `a.remove`. From [this page](https://stackoverflow.com/questions/3157374/how-do-you-remove-a-numpy-array-from-a-list-of-numpy-arrays), I figure out using `a.pop(i)`instead.

<a name='rf'></a>

## Reference

[CS231n](http://cs231n.stanford.edu/)

[Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits](http://cs231n.github.io/classification/)

[Assignment #1: Image Classification, kNN, SVM, Softmax, Neural Network](http://cs231n.github.io/assignments2019/assignment1/)
