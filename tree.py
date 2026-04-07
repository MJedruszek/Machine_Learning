import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left            # Left child node
        self.right = right          # Right child node
        self.value = value          # Class label for leaf nodes; only filled for leaves

            
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    #compute the entropy based on class probabilities of the next nodes
    #low entropy -> good split, high probability of one class
    #high entropy -> bad split, both classes have a high probability
    #using y because y is the name of classes array
    def _entropy(self, y):
        counts = np.bincount(y)           #Count each class -> how many examples of 0/1 after split
        probabilities = counts / len(y)   #Calculate proportions (how many 0/1 are there as a portion of the whole)
        probabilities = probabilities[probabilities > 0]  #Remove zeroes
        return -np.sum(probabilities * np.log2(probabilities))  #Compute entropy
    
    def _information_gain(self, y, y_left, y_right):
        p_left = len(y_left) / len(y)     #Count how many go left (as proportions)
        p_right = len(y_right) / len(y)   #Count how many go right (as proportions)
        
        entropy_before = self._entropy(y)  #Compute entropy before split
        #Compute total entropy after split: probability to go in that direction*entropy of the direction
        entropy_after = p_left * self._entropy(y_left) + p_right * self._entropy(y_right)  
        
        return entropy_before - entropy_after  #How much did we improve?
    
    def _best_split(self, X, y):
        best_gain = -1          #Track the highest information gain found
        best_feature = None     #Track which feature gave best split
        best_threshold = None   #Track what threshold gave best split
        
        n_features = X.shape[1]  #Get the number of features
        
        for feature in range(n_features):  #Try each feature as the split category
            thresholds = np.unique(X[:, feature])  #Get all unique values for this feature
            
            for threshold in thresholds:  #Try each possible split point
                #Create masks (True/False arrays) for splitting
                #threshold: is this feature <= to the number? yes-> left, no -> right
                left_mask = X[:, feature] <= threshold #apply the threshold for this split
                right_mask = ~left_mask #the rest go here
                
                #Get the classes for each side
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                #Skip if one side is empty (this split just takes every sample to the same direction)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                #Calculate how good this split is using previous function
                gain = self._information_gain(y, y_left, y_right)
                
                #Update if this is the best split so far
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        #return the best results
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0] #how many training samples are there
        
        #STOPPING CONDITIONS - Check if we should make a leaf
        
        #Condition 1: Reached maximum depth
        if depth >= self.max_depth:
            leaf_value = np.bincount(y).argmax()  #Most common class in this bin
            return Node(value=leaf_value)
        
        #Condition 2: Too few samples to split (parameter set at the beggining)
        if n_samples < self.min_samples_split:
            leaf_value = np.bincount(y).argmax() #Most common class in this bin
            return Node(value=leaf_value)
        
        #Condition 3: All samples are the same class
        if len(np.unique(y)) == 1: #how many unique classes are there? if 1, then this is a leaf
            leaf_value = y[0]  #All are this class anyway
            return Node(value=leaf_value)
        
        #SPLITTING - Find best split
        feature, threshold, gain = self._best_split(X, y)
        
        #If there will be no gain, don't split here -> it will be also a leaf node
        #This happens if there are no good split possibilities
        if gain <= 0:
            leaf_value = np.bincount(y).argmax() #Most common class in this bin
            return Node(value=leaf_value)
        
        #RECURSION - Build children
        
        #Split the data according to the threshold computed at splitting step
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        #Build left subtree (samples that go left) recursively
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        
        #Build right subtree (samples that go right) recursively
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        #Return decision node pointing to both children as computed above
        #Leaf nodes have been returned at previous steps if necessary
        return Node(feature=feature, threshold=threshold,left=left_child, right=right_child)
    
    #fit -> only function that is public (aside from init and predict)
    #Tree is saved like so:
    #Root node saved in tree class
    #Decisions from root saved as its left and right children
    #Decisions from left and right root children saved as their children
    #And so on
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    #in order to start, give root as node
    def predict_sample(self, sample, node):
        #If we're at a leaf, return its prediction
        if node.value is not None:
            return node.value
        
        #Otherwise, decide which way to go
        if sample[node.feature] <= node.threshold:
            return self.predict_sample(sample, node.left)   #Go left
        else:
            return self.predict_sample(sample, node.right)  #Go right
        
    #Helper function: visualise to help debugging
    #Not ideal, but good enough to draw the tree by hand with its help
    def print_tree(self, node=None, depth=0):
        #beggining node = root
        if node is None:
            node = self.root
        
        if node.value is not None:
            print(f"{'  ' * depth}Leaf: Class {node.value}")
        else:
            print(f"{'  ' * depth}Feature {node.feature} <= {node.threshold:.2f}")
            print(f"{'  ' * depth}Left:")
            self.print_tree(node.left, depth + 1)
            print(f"{'  ' * depth}Right:")
            self.print_tree(node.right, depth + 1)

