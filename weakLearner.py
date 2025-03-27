#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:
            

        """

        self.fidx = None
        self.threshold = None

        #print "   "        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        #nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        try:
            nexamples, nfeatures=X.shape
        except:
            nexamples=1


        best_feature = None
        score = None
        best_score = float("inf")
        Xlidx = None
        Xridx = None
        
        for feature in range(X.shape[1]):
            threshold, score, xlidx, xridx = self.evaluate_numerical_attribute(X[:, feature], Y)
                
            if score < best_score:
                best_feature = feature
                score = threshold
                best_score = score
                Xlidx = xlidx
                Xridx = xridx

        self.fidx = best_feature
        self.threshold = score
        
        #---------End of Your Code-------------------------#
        return score, Xlidx,Xridx
  
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        return X[self.fidx] <= self.threshold            
        
        #---------End of Your Code-------------------------#
  
    def get_split_points(self, f):
        """Get all possible split points for a feature."""
        # return (f[1:] + f[:-1]) / 2
        return [  round( (f[idx]+f[idx+1])/2, 2 ) for idx in range(len(f)-1) ]
      
    def entropy(self, y):
        """Calculate entropy."""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        res = -np.sum(probabilities * np.log2(probabilities))
        if res == -0.0:
            return 0.0
        return res

    def entropy_of_split(self, y_left, y_right):
        """Calculate the weighted entropy of a split."""
        n = len(y_left) + len(y_right)
        e_left = self.entropy(y_left)
        e_right = self.entropy(y_right)
        return (len(y_left) / n) * e_left + (len(y_right) / n) * e_right
    
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        #classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...

        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...

        # YOUR CODE HERE
        split = -1
        best_entropy = float("inf")
        Xlidx=[]
        Xridx=[]

        unique_f = np.unique(f)
        split_points = self.get_split_points(unique_f)


        for threshold in split_points:
            left_indices = feat <= threshold
            right_indices = feat > threshold
            y_left, y_right = Y[left_indices], Y[right_indices]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            current_entropy = self.entropy_of_split(y_left, y_right)
            
            if current_entropy < best_entropy:
                best_entropy = current_entropy
                split = threshold
                Xlidx = left_indices
                Xridx = right_indices

        if best_entropy == -0.0:
            best_entropy = 0.0
        
        #---------End of Your Code-------------------------#
            
        return split,best_entropy,Xlidx,Xridx
        #return split,mingain,Xlidx,Xridx


class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        best_feature = None
        best_threshold = None
        best_score = float("inf")
        bXl = None
        bXr = None

        if self.nrandfeat != +np.inf:
            split_values =  list(set(np.random.choice(nfeatures, self.nrandfeat, replace=False)))
        else:
            split_values = range(nfeatures)

        for feature in split_values:
            threshold, score, xlidx, xridx = self.findBestRandomSplit(X[:, feature], Y)

                
            if score < best_score:
                best_feature = feature
                best_threshold = threshold
                best_score = score
                bXl = xlidx
                bXr = xridx

        self.fidx = best_feature
        self.threshold = best_threshold
        
        #---------End of Your Code-------------------------#
        
        return best_score, bXl, bXr
        # return best_threshold, bXl, bXr
        # return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        # return splitvalue, score, np.array(indxLeft), indxRight

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        classes=np.unique(Y)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...

        best_threshold = -1
        best_entropy = float("inf")
        Xlidx=[]
        Xridx=[]

        try:
            f = list(set(np.random.choice(f, self.nsplits, replace=False)))
        except:
            f = f

        unique_f = sorted(np.unique(f))
        # split_points = np.array(unique_f[:-1] + unique_f[1:]) / 2
        split_points = unique_f

        for threshold in split_points:
            left_indices = feat <= threshold
            right_indices = feat > threshold
            y_left, y_right = Y[left_indices], Y[right_indices]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # ? Method 2
            # left_entropy = self.calculateEntropy(Y, left_indices)
            # right_entropy = self.calculateEntropy(Y, right_indices)
            # n_left = np.sum(left_indices)
            # n_right = np.sum(right_indices)
            # current_entropy = (n_left * left_entropy + n_right * right_entropy) / len(Y)

            # ? Method 1
            current_entropy = self.entropy_of_split(y_left, y_right)
            
            if current_entropy < best_entropy:
                best_entropy = current_entropy
                best_threshold = threshold
                Xlidx = left_indices
                Xridx = right_indices

        if best_entropy == -0.0:
            best_entropy = 0.0
        
        #---------End of Your Code-------------------------#

        return best_threshold,best_entropy,Xlidx,Xridx
        #return splitvalue, minscore, Xlidx, Xridx
 
    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        # pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        # pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        pl = np.unique(lexam, return_counts=True)[1] / float(len(lexam)) + np.spacing(1)
        pr = np.unique(rexam, return_counts=True)[1] / float(len(rexam)) + np.spacing(1)


        hl= -np.sum(pl*np.log2(pl)) 
        hr= -np.sum(pr*np.log2(pr)) 

        sentropy = pleft * hl + pright * hr

        return sentropy



# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...
    """

    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)

        self.a = None
        self.b = None
        self.c = None
        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

    
        nfeatures = X.shape[1]
        random_features = list(set([tuple(np.random.choice(nfeatures, 2, replace=False)) for _ in range(self.nrandfeat)]))
        random_features = [ list(x) for x in random_features ]

        best_features = None
        best_a, best_b, best_c = None, None, None
        best_score = float("inf")
        bXl = None
        bXr = None


        for feature in random_features:

            a, b, c, score, xlidx, xridx = self.findBestRandomSplit(X, Y, feature)
                
            if score < best_score:
                best_features = feature
                best_a, best_b, best_c = a, b, c
                best_score = score
                bXl = xlidx
                bXr = xridx


        self.a = best_a
        self.b = best_b
        self.c = best_c
        self.fidx = best_features

        #---------End of Your Code-------------------------#

        return best_score, bXl, bXr
        
    def findBestRandomSplit(self,X,Y, feature):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X m] nexamples with a m features
            Y: [n X 1] label vector...

        """
        # return splitvalue, score, np.array(indxLeft), indxRight

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        f1 = feature[0]
        f2 = feature[1]
        best_entropy = float("inf")
        Xlidx=None
        Xridx=None
        best_a, best_b, best_c = None, None, None
        current_entropy = None

        random_splits = np.random.uniform(min(X[:, f1]), max(X[:, f2]), size=(self.nsplits, 3))  # Generate n random splits (a, b, c)

        for a, b, c in random_splits:
                # Determine the split based on ax + by + c <= 0
                left_indices = ((a * X[:, f1]) + (b * X[:, f2]) + c) <= 0
                right_indices = ((a * X[:, f1]) + (b * X[:, f2]) + c) > 0
                
                y_left, y_right = Y[left_indices], Y[right_indices]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # ? Method 2
                # left_entropy = self.calculateEntropy(Y, left_indices)
                # right_entropy = self.calculateEntropy(Y, right_indices)
                # n_left = np.sum(left_indices)
                # n_right = np.sum(right_indices)
                # current_entropy = (n_left * left_entropy + n_right * right_entropy) / len(Y)

                # ? Method 1
                current_entropy = self.entropy_of_split(y_left, y_right)
                                
                if current_entropy < best_entropy:
                    best_entropy = current_entropy
                    best_a, best_b, best_c = a, b, c
                    Xlidx, Xridx = left_indices, right_indices

        if best_entropy == -0.0:
            best_entropy = 0.0
        
        #---------End of Your Code-------------------------#

        return best_a, best_b, best_c , best_entropy, Xlidx, Xridx
 
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        return (self.a * X[self.fidx[0]] + self.b * X[self.fidx[1]] + self.c) <= 0

        #---------End of Your Code-------------------------#
        # build a classifier ax+by+c=0


class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...
    """
 
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass
 
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        

        bfidx=-1, # best feature idx
        minscore=+np.inf
        
        
        tdim= np.ones((nexamples,1))# third dimension
        for i in np.arange(self.nsplits):
            # a*x^2+b*y^2+c*x*y+ d*x+e*y+f


            if i%5==0: # select features indeces after every five iterations
                fidx=np.random.randint(0,nfeatures,2) # sample two random features...
                # Randomly sample a, b and c and test for best parameters...
                parameters = np.random.normal(size=(6,1))
                # apply the line equation...
                res = np.dot ( np.hstack( (np.power(X[:,fidx],2),X[:,fidx],np.prod(X[:,fidx],1)[:,np.newaxis], tdim) ) , parameters )

            splits=np.random.normal(size=(2,1))
            
            # set split to -np.inf for 50% of the cases in the splits...
            if np.random.random(1) < 0.5:
                splits[0]=-np.inf

            tres=  np.logical_and(res >= splits[0], res < splits[1])

            # print(tres.shape)
            
            score = self.calculateEntropy(Y,tres)

            if score < minscore:
                
                bfidx=fidx # best feature indeces
                bparameters=parameters # best parameters...
                minscore=score
                bres= tres
                bsplits=splits

        self.parameters=bparameters
        self.score=minscore
        self.splits=bsplits
        self.fidx=bfidx
        
        bXl=np.squeeze(bres)
        bXr=np.logical_not(bXl)

        return minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        fidx=self.fidx
        # res = np.dot ( np.hstack( (np.power(X[:,fidx],2),X[:,fidx],np.prod(X[:,fidx],1)[:,np.newaxis], np.ones((X.shape[0],1))) ) , self.parameters ) 
        # res = np.dot ( np.hstack( (np.power(X[fidx[0]],2),X[fidx[1]],np.prod(X[fidx],1)[:,np.newaxis], np.ones((X.shape[0],1))) ) , self.parameters ) 

        if X.ndim == 1:
            X = X.reshape(1, -1)  # Reshape if X is a single example with two features

        res = np.dot(
            np.hstack((
                np.power(X[:, fidx], 2),  # Quadratic term
                X[:, fidx],               # Linear term
                np.prod(X[:, fidx], axis=1)[:, np.newaxis],  # Cross term
                np.ones((X.shape[0], 1))  # Constant term
            )),
            self.parameters
        )


        return np.logical_and(res >= self.splits[0], res < self.splits[1])
    
    

"""    
wl=WeakLearner()
rwl=RandomWeakLearner()
lwl=LinearWeakLearner()
"""
        
