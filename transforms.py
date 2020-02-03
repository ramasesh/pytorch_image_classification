import numpy as np
import torch


class ToTensor:
    def __call__(self, data):
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    def _to_tensor(self, data):
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image

class Scaler:
    def __init__(self, prob, var):
        """ prob - probability of augmenting (i.e. scaling) input,
            var - variance of normal distribution we sample the scale_factor from """

        self.scale_prob = prob
        self.scale_std = np.sqrt(var)

    def __call__(self, data):
                
        if isinstance(data, tuple):
            """ data may be a tuple with a label in the second index.  """
            to_scale = data[0]
        else:
            to_scale = data

        scaled = self._sample_scale_factor() * to_scale

        if isinstance(data, tuple):
            return (scaled, data[1])
        else:
            return scaled

    def _sample_scale_factor(self):
        x = np.random.uniform()
        if x > self.scale_prob:
            return 1.
        else:
            return 1. + np.random.randn() * self.scale_std

class Reflector:
    """ Transform which randomly and independently reflects coordinates of a data point
    about the origin, i.e. flips the data sign""" 

    def __init__(self, prob):
        """ prob  - the probability that each coordinate will flip """
        assert prob <= 1.
        assert prob >= 0.
        self.reflect_prob = prob

    def __call__(self, data):
        if isinstance(data, tuple):
            """ data may be a tuple with a label in the second index.  """
            to_reflect = data[0]
        else:
            to_reflect = data

        reflected = self._sample_reflection_vector(to_reflect.size()[0]) * to_reflect

        if isinstance(data, tuple):
            return (reflected, data[1])
        else:
            return reflected
    
    def _sample_reflection_vector(self, size):
        reflection_vector = -2.*np.random.binomial(1, self.reflect_prob, size=size) + 1
        return torch.Tensor(reflection_vector)
