from ImpurityMeasures import ImpurityMeasures
import numpy as np


class Entropy(ImpurityMeasures):

    def compute_impurity(self,dataset):
        count = np.bincount(dataset.y)
        prob = np.array(list(map(lambda x:(x/dataset.num_samples), count)))
        filtered_prob=[x for x in prob if x!=0]
        entropy = sum([prob*log for prob,log in zip(filtered_prob, np.log(filtered_prob))])
        return -entropy

