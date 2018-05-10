import numpy as np
from scipy.stats import norm

def soft_max(logits, temperature=1,num_classes=None):
    assert len(logits.shape)==2,'Unrecognized logits shape'
    if logits.shape[0]==0:
        return logits
    soft_max = np.minimum(np.finfo(np.float32).max,np.exp(logits/temperature))
    if num_classes is not None and num_classes!=logits.shape[1]:
        soft_max = soft_max.reshape([soft_max.shape[0],-1,num_classes])
    soft_max = soft_max/np.maximum(np.finfo(np.float32).eps,np.minimum(np.finfo(np.float32).max,np.sum(soft_max,axis=-1,keepdims=True)))
    soft_max = soft_max/np.maximum(np.finfo(np.float32).eps,np.sum(soft_max,axis=-1,keepdims=True))#Normalizing again for the case where the actual sum used for normalization overflowed, and the minimum above was used.
    if num_classes is not None and num_classes!=logits.shape[1]:
        soft_max = soft_max.reshape(logits.shape)
    return soft_max
def MutualySortLogitsOfImageVersions(logits,num_classes=None,num_copies=None,descending_order=True):
    assert not (num_copies is None and num_classes is None),'Should pass at least one of them'
    assert len(logits.shape)==2,'Unrecognized logits shape'
    if logits.shape[0]==0:
        return logits
    if num_classes is None:
        num_classes = int(logits.shape[1]/num_copies)
    elif num_copies is None:
        num_copies = int(logits.shape[1]/num_classes)
    else:
        assert num_classes*num_copies==logits.shape[1],'Values don not fit'
    if descending_order:
        logits_reordering = np.argsort(logits[:,:num_classes], axis=1)[:, ::-1].reshape([-1,1,num_classes])
    else:
        logits_reordering = np.argsort(logits[:,:num_classes], axis=1).reshape([-1,1,num_classes])

    logits_reordering = (logits_reordering+np.arange(0,num_classes*num_copies,num_classes).reshape([1,num_copies,1])).reshape([logits.shape[0],-1])
    return logits[np.arange(logits.shape[0]).reshape([-1, 1]), logits_reordering]
def BaselineScores(data,num_classes,soft_max_input=False,top5_baseline=False):
    if not soft_max_input:
        data = soft_max(data[:,:num_classes])
    if top5_baseline:
        return np.sum(np.sort(data[:,:num_classes],axis=1)[:,-5:],axis=1)
    else:
        return np.max(data[:,:num_classes],axis=1)
def AccuracyVsCoveragePlot(y_true,y_score):
    ordered_y_true = np.logical_not(y_true[np.argsort(y_score)])
    coverage = np.cumsum(np.ones([len(y_score)])/len(y_score))
    accuracy = np.cumsum(ordered_y_true)/(1+np.arange(len(y_score)))
    return coverage,accuracy
def PostIdealRejectionAccuracy(coverage,accuracy,classifier_accuracy):
    return accuracy[coverage<=classifier_accuracy][-1]
class loss_tracker():
    def __init__(self,min_win_length=10,min_so_far=None,distribution='Uniform'):
        self.min_win_length = min_win_length
        self.distribution = distribution
        self.memory = []
        self.minimum = min_so_far if min_so_far is not None else np.finfo(np.float32).max
        self.maximum = self.minimum
    def update(self,loss):
        if loss<self.minimum:
            self.minimum = loss
            self.maximum = self.minimum
            self.memory = []
        elif loss>self.maximum:
            self.maximum = loss
            # print('********* New tracker maximum: %.4f'%(self.maximum))
        self.memory.append(loss)
        if self.distribution=='Gaussian':
            if len(self.memory)>=self.min_win_length:
                return norm.cdf(self.minimum,loc=np.mean(self.memory),scale=np.std(self.memory))
            else:
                return None
        elif self.distribution=='Uniform':
            est_lower_boundry = np.maximum(0,self.minimum-(self.maximum-self.minimum)/len(self.memory))
            return (self.minimum-est_lower_boundry)/(self.maximum-est_lower_boundry)
    def get_min(self):
        return self.minimum
    def steps_left(self,min_chance):
        # if len(self.memory)>=self.min_win_length:
        return np.ceil(1/min_chance-1-len(self.memory))
        # else:
        #     return None
