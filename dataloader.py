import pathlib
import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

from src import augmentations
from src import transforms

import json

from itertools import chain


class PointDataset(torch.utils.data.Dataset):

    def __init__(self, points, labels, config=None):
        assert isinstance(points, torch.Tensor)
        assert len(points.size()) == 2

        self.data = points
        self.config = config
        self.labels = labels

        self._transforms = []
        self.transforms = self._get_transforms()

    def __getitem__(self, idx):
        point = self.data[idx]
        for transform in self.transforms:
            point = transform(point)

        label = self.labels[idx]
        return point, label

    def __len__(self):
        return self.data.size()[0]

    def _get_transforms(self):
        if self.config['use_random_scale']:
            self._add_random_scale() 
        if self.config['use_random_reflection']:
            self._add_random_reflection()
        return self._transforms

    def _add_random_scale(self):
        scale_prob = self.config['random_scale_prob']
        scale_var = self.config['random_scale_var']
        scaler = transforms.Scaler(scale_prob, scale_var) 
        self._transforms.append(scaler)

    def _add_random_reflection(self):
        reflect_prob = self.config['random_reflection_prob']
        reflector = transforms.Reflector(reflect_prob)
        self._transforms.append(reflector)


class Dataset:
    torchvision_datasets = ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST']
    imagefolder_datasets = ['CINIC10']
    def __init__(self, config):
        self.config = config
        dataset_rootdir = pathlib.Path('~/.torchvision/datasets').expanduser()
        self.dataset_dir = dataset_rootdir / config['dataset']

        self._train_transforms = []
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def get_datasets(self):
        if self.config['dataset'] in self.torchvision_datasets:
            train_dataset = getattr(torchvision.datasets, self.config['dataset'])( self.dataset_dir,
                train=True,
                transform=self.train_transform,
                download=True)
            test_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                self.dataset_dir,
                train=False,
                transform=self.test_transform,
                download=True)
        else:
            train_dataset = torchvision.datasets.ImageFolder(self.directory + '/train',
                                                             transform = self.train_transform)
            test_dataset = torchvision.datasets.ImageFolder(self.directory + '/test',
                                                             transform = self.test_transform)
        return train_dataset, test_dataset

    def _add_random_crop(self):
        transform = torchvision.transforms.RandomCrop(
            self.size, padding=self.config['random_crop_padding'])
        self._train_transforms.append(transform)

    def _add_horizontal_flip(self):
        self._train_transforms.append(
            torchvision.transforms.RandomHorizontalFlip())

    def _add_normalization(self):
        self._train_transforms.append(
            transforms.Normalize(self.mean, self.std))

    def _add_to_tensor(self):
        self._train_transforms.append(transforms.ToTensor())

    def _add_random_erasing(self):
        transform = augmentations.random_erasing.RandomErasing(
            self.config['random_erasing_prob'],
            self.config['random_erasing_area_ratio_range'],
            self.config['random_erasing_min_aspect_ratio'],
            self.config['random_erasing_max_attempt'])
        self._train_transforms.append(transform)

    def _add_cutout(self):
        transform = augmentations.cutout.Cutout(self.config['cutout_size'],
                                                self.config['cutout_prob'],
                                                self.config['cutout_inside'])
        self._train_transforms.append(transform)

    def _add_dual_cutout(self):
        transform = augmentations.cutout.DualCutout(
            self.config['cutout_size'], self.config['cutout_prob'],
            self.config['cutout_inside'])
        self._train_transforms.append(transform)

    def _get_train_transform(self):
        if self.config['use_random_crop']:
            self._add_random_crop()
        if self.config['use_horizontal_flip']:
            self._add_horizontal_flip()
        self._add_normalization()
        if self.config['use_random_erasing']:
            self._add_random_erasing()
        if self.config['use_cutout']:
            self._add_cutout()
        elif self.config['use_dual_cutout']:
            self._add_dual_cutout()
        self._add_to_tensor()
        return torchvision.transforms.Compose(self._train_transforms)

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            transforms.Normalize(self.mean, self.std),
            transforms.ToTensor(),
        ])
        return transform


class CIFAR(Dataset):
    def __init__(self, config):
        self.size = 32
        if config['dataset'] == 'CIFAR10':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2470, 0.2435, 0.2616])
        elif config['dataset'] == 'CIFAR100':
            self.mean = np.array([0.5071, 0.4865, 0.4409])
            self.std = np.array([0.2673, 0.2564, 0.2762])
        super(CIFAR, self).__init__(config)


class MNIST(Dataset):
    def __init__(self, config):
        self.size = 28
        if config['dataset'] == 'MNIST':
            self.mean = np.array([0.1307])
            self.std = np.array([0.3081])
        elif config['dataset'] == 'FashionMNIST':
            self.mean = np.array([0.2860])
            self.std = np.array([0.3530])
        elif config['dataset'] == 'KMNIST':
            self.mean = np.array([0.1904])
            self.std = np.array([0.3475])
        super(MNIST, self).__init__(config)

class CINIC(Dataset):
    def __init__(self, config):
        self.size = 32
        self.mean = np.array([0.47889522, 0.47227842, 0.43047404])
        self.std = np.array([0.24205776, 0.23828046, 0.25874835])
        self.directory = './datasets/CINIC10'
        super(CINIC, self).__init__(config)

class spheres():
    def __init__(self, config):
        self.dimension = config['sphere_dim']
        self.inner_radius = config['sphere_inner_rad']
        self.outer_radius = config['sphere_outer_rad']
        self.trainset_size = config['sphere_trainset_size']
        self.testset_size = config['sphere_testset_size']
  
        self.train_augmentation_config, self.test_augmentation_config =  self._get_augmentation_configs(config)

        self.__generate_datasets__()

    def __generate_datasets__(self):
       
        self.testset = self.generate_dataset(self.testset_size, 
                                             self.dimension, 
                                             self.inner_radius, 
                                             self.outer_radius,
                                             self.test_augmentation_config)
        self.trainset = self.generate_dataset(self.trainset_size, 
                                              self.dimension, 
                                              self.inner_radius, 
                                              self.outer_radius,
                                              self.train_augmentation_config)

    def _get_augmentation_configs(self, config):
        """ NOTE: Does not set defaults here!
        If you set defaults here, the problem is that you will not have a good record of the experiment,
        because the experiment config can be missing these values.  So better to throw an error here if keys 
        are not provided """

        required_keys = {
                        'use_random_scale': ['random_scale_prob', 'random_scale_var'],
                        'use_random_reflection': ['random_reflection_prob']
                        }

        train_augmentation_config = {}
        test_augmentation_config = {}
        for key, dependent_keys in required_keys.items():
            if key not in config.keys():
                raise Exception(f'{key} is required in spheres dataset')
            else:
                train_augmentation_config[key] = config[key]
                if train_augmentation_config[key]:
                    # make sure the dependent keys are there
                    for dependent_key in dependent_keys:
                        if dependent_key not in config.keys():
                            raise Exception(
                                            f'{dependent_key} is required'
                                            f' in spheres dataset when {key} is set to True'
                                            )
                        else:
                            train_augmentation_config[dependent_key] = config[dependent_key]

                test_augmentation_config[key] = False

        return train_augmentation_config, test_augmentation_config

    def generate_dataset(self, n_pts_total, dimension, inner_radius, outer_radius, augmentation_config):

        inner_sphere_data = self.sample_sphere(n_pts_total/2, dimension, inner_radius)
        outer_sphere_data = self.sample_sphere(n_pts_total/2, dimension, outer_radius)
        combined_data = torch.Tensor(np.array(list(chain.from_iterable(zip(inner_sphere_data, outer_sphere_data)))))

        n_labels = int(n_pts_total/2)
        labels = n_labels * [0,1]

        return PointDataset(combined_data, labels, augmentation_config)

    def sample_sphere(self, n_pts, dimension, radius):
        gaussians = np.random.randn(int(n_pts), int(dimension))
        return [self.scale_norm(pt, radius) for pt in gaussians]

    def get_datasets(self):
        return self.trainset, self.testset 
   
    def scale_norm(self, pt, norm):
        """ converts the norm of the point to the desired norm """
        return pt/np.linalg.norm(pt)*norm

class DownsampledDataset():
    # Dataset with a subset of images from the original labeled dataset,
    #   with an equal number of images per class
    def __init__(self, full_dataset, num_pts_per_class, label_locations, random_seed=0):
        # seed is for selecting the elements of each class to take
        self.full_dataset = full_dataset
        self.label_locations = label_locations

        self.__set_selected_indices__(random_seed, num_pts_per_class)

    def __set_selected_indices__(self, random_seed, num_pts_per_class):
        np.random.seed(random_seed)
        self.selected_indices = []
        for key in self.label_locations.keys():
            self.selected_indices.append(np.random.choice(self.label_locations[key],
                                                          size=num_pts_per_class,
                                                          replace=False))
        self.selected_indices = np.array(self.selected_indices, dtype=int)
        self.selected_indices = self.selected_indices.flatten('F') # column-major flattening

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        return self.full_dataset.__getitem__(self.selected_indices[idx])


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_loader(config):
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    use_gpu = config['use_gpu']

    dataset_name = config['dataset']
    assert dataset_name in [
        'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST', 'CINIC10', 'spheres'
    ]

    if dataset_name in ['CIFAR10', 'CIFAR100']:
        dataset = CIFAR(config)
    elif dataset_name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset = MNIST(config)
    elif dataset_name in ['CINIC10']:
        dataset = CINIC(config)
    elif dataset_name in ['spheres']:
        dataset = spheres(config)
        test_batch_size=500
    train_dataset, test_dataset = dataset.get_datasets()

    # handle subsampling
    if 'examples_per_class' in config.keys():
        examples_per_class = config['examples_per_class']
        print(f'Subsampling training dataset to {examples_per_class} training examples per class')

        with open('src/dataset_indices.json') as f:
            label_locations = json.load(f)
        label_locations = label_locations[dataset_name]['train']

        train_dataset = DownsampledDataset(train_dataset, examples_per_class, label_locations, random_seed = config['subsampling_seed'])
        print('Length of new dataset')
        print(len(train_dataset))
    else:
        print('Not subsampling')
    # end handle subsampling

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    if dataset_name not in ['spheres']:
      test_loader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=batch_size,
          num_workers=num_workers,
          shuffle=False,
          pin_memory=use_gpu,
          drop_last=False,
      )
    else:
      test_loader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=test_batch_size,
          num_workers=num_workers,
          shuffle=False,
          pin_memory=use_gpu,
          drop_last=False,
      )

    return train_loader, test_loader
