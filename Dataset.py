import numpy as np


class Dataset():
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.num_samples = len(X)
        if self.num_samples != 0:
            self.num_features=len(X[0])
        else:
            self.num_features=0
    def random_sampling(self,ratio_samples):
        idx=[x for x in range(self.num_samples)]
        random_idx=np.random.choice(idx, int(ratio_samples * self.num_samples), True)
        X=self.X[random_idx]
        y=self.y[random_idx]
        return Dataset(X,y)
    def most_frequent_label(self):
        counts = np.bincount(self.y)
        return np.argmax(counts)
    def split(self,idx,value):
        left_idx=np.array([],dtype=int)
        right_idx=np.array([],dtype=int)
        samples_of_feature_idx= self.X[:, idx]
        for idx,sample in enumerate(samples_of_feature_idx):
            if sample <= value:
                left_idx=np.append(left_idx,idx)

            else:
                right_idx=np.append(right_idx,idx)

        left_dataset=Dataset(self.X[left_idx],self.y[left_idx])
        right_dataset=Dataset(self.X[right_idx],self.y[right_idx])
        return left_dataset, right_dataset

