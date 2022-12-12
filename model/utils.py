import numpy as np
import torch

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

def mse(prediction, target):
    return torch.mean(torch.square(target - prediction))

# def split_data(data, train_ratio, valid_ratio, use_ratio=1, randomSeed = None):
#       num_train = len(data)
#       indices = list(range(num_train))

#       random_state = np.random.RandomState(randomSeed)
#       random_state.shuffle(indices)

#       indices = indices[:int(len(indices)*use_ratio)]

#       split = int(np.floor(valid_ratio * len(indices)))
#       train_idx, valid_idx = indices[split:], indices[:split]

#       return data[train_idx], data[valid_idx]
def split_data(data, test_ratio, valid_ratio, use_ratio=1, randomSeed = None):
    total_size = len(data)
    train_ratio = 1 - valid_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", randomSeed)
    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
    train_size, valid_size, test_size
    ))

    train_idx, valid_idx, test_idx = indices[:train_size], indices[-(valid_size + test_size):-test_size], indices[-test_size:]



    return data[train_idx], data[valid_idx], data[test_idx]

def split_data_subset(data, test_ratio, valid_ratio, subset_size=500, randomSeed = None):
    total_size = len(data)
    train_ratio = 1 - valid_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", randomSeed)
    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
    subset_size, valid_size, test_size
    ))
    train_idx = np.random.choice(indices[:train_size], size=subset_size, replace=False)
    valid_idx, test_idx = indices[-(valid_size + test_size):-test_size], indices[-test_size:]



    return data[train_idx], data[valid_idx], data[test_idx]