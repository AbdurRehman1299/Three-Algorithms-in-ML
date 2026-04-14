import numpy as np
from collections import Counter

# 1. K Nearest Neighbor Algorithm
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # KNN trains simply by saving the images in memory
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

# 2. Naive Bayes Algorithm
class NaiveBayes:
    def fit(self, X, y):
        X_binary = np.where(X > 0.6, 1, 0)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.feature_probs = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X_binary[y == c]

            self.feature_probs[idx, :] = (X_c.sum(axis=0) + 1) / (X_c.shape[0] + 2)
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def predict(self, X):
        X_binary = np.where(X > 0.6, 1, 0)
        return np.array([self._predict(x) for x in X_binary])

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])

            prob_1 = np.log(self.feature_probs[idx, :])
            prob_0 = np.log(1.0 - self.feature_probs[idx, :])

            posterior = prior + np.sum(x * prob_1 + (1 - x) * prob_0)
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

# 3. Decision Tree Algorithm
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0.0
        n = len(y)
        child_entropy = (len(left_idxs) / n) * self._entropy(y[left_idxs]) + (len(right_idxs) / n) * self._entropy(y[right_idxs])
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        return np.argwhere(X_column <= split_thresh).flatten(), np.argwhere(X_column > split_thresh).flatten()

    def _entropy(self, y):
        y = np.array(y, dtype=int)
        ps = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# 4. Evaluation Metrics
def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    precisions, recalls, f1s = [], [], []
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)