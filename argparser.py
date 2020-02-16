import json
from collections import OrderedDict
import torch


def _args2config(args, keys, json_keys):
    if json_keys is None:
        json_keys = []

    args = vars(args)

    config = OrderedDict()
    for key in keys:
        value = args[key]
        if value is None:
            continue

        if key in json_keys and isinstance(value, str):
            value = json.loads(value)

        config[key] = value

    return config


def _get_model_config(args):
    keys = [
        'arch',
        'input_shape',
        'n_classes',
        # vgg
        'n_channels',
        'n_layers',
        'use_bn',
        #
        'base_channels',
        'block_type',
        'depth',
        # small_relu
        'n_hidden_units',
        'n_hidden_layers',
        # resnet_preact, se_resnet_preact
        'remove_first_relu',
        'add_last_bn',
        'preact_stage',
        # wrn
        'widening_factor',
        # densenet
        'growth_rate',
        'compression_rate',
        # wrn, densenet
        'drop_rate',
        # pyramidnet 'pyramid_alpha',
        # resnext
        'cardinality',
        # shake_shake
        'shake_forward',
        'shake_backward',
        'shake_image',
        # se_resnet_preact
        'se_reduction',
    ]
    json_keys = ['preact_stage']
    config = _args2config(args, keys, json_keys)
    return config


def _check_optim_config(config):
    optimizer = config['optimizer']
    for key in ['base_lr', 'weight_decay']:
        message = 'Key `{}` must be specified.'.format(key)
        assert key in config.keys(), message
    if optimizer == 'sgd':
        for key in ['momentum', 'nesterov']:
            message = 'When using SGD, key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif optimizer == 'adam':
        for key in ['betas']:
            message = 'When using Adam, key `{}` must be specified.'.format(
                key)
            assert key in config.keys(), message
    elif optimizer == 'lars':
        for key in ['momentum']:
            message = 'When using LARS, key `{}` must be specified.'.format(
                key)
            assert key in config.keys(), message

    scheduler = config['scheduler']
    if scheduler == 'multistep':
        for key in ['milestones', 'lr_decay']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif scheduler == 'cosine':
        for key in ['lr_min']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif scheduler == 'sgdr':
        for key in ['lr_min', 'T0', 'Tmult']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message


def _get_optim_config(args):
    keys = [
        'epochs',
        'batch_size',
        'ghost_batch_size',
        'optimizer',
        'base_lr',
        'weight_decay',
        'no_weight_decay_on_bn',
        'momentum',
        'nesterov',
        'gradient_clip',
        'scheduler',
        'milestones',
        'lr_decay',
        'lr_min',
        'T0',
        'Tmult',
        'betas',
        'lars_eps',
        'lars_thresh',
    ]
    json_keys = ['milestones', 'betas']
    config = _args2config(args, keys, json_keys)

    _check_optim_config(config)

    return config


def _get_data_config(args):
    keys = [
        'dataset',
        'n_classes',
        'examples_per_class',
        'subsampling_seed',
        'num_workers',
        'batch_size',
        'use_random_crop',
        'random_crop_padding',
        'use_horizontal_flip',
        'use_cutout',
        'use_dual_cutout',
        'cutout_size',
        'cutout_prob',
        'cutout_inside',
        'use_random_erasing',
        'dual_cutout_alpha',
        'random_erasing_prob',
        'random_erasing_area_ratio_range',
        'random_erasing_min_aspect_ratio',
        'random_erasing_max_attempt',
        'use_mixup',
        'mixup_alpha',
        'use_ricap',
        'ricap_beta',
        'use_label_smoothing',
        'label_smoothing_epsilon',
        'sphere_dim',
        'sphere_inner_rad',
        'sphere_outer_rad',
        'sphere_trainset_size',
        'sphere_testset_size',
        'use_random_scale',
        'random_scale_prob',
        'random_scale_var',
        'use_random_reflection',
        'random_reflection_prob'
        ]
    json_keys = ['random_erasing_area_ratio_range']
    config = _args2config(args, keys, json_keys)
    config['use_gpu'] = args.device != 'cpu'
    _check_data_config(config)
    return config


def _check_data_config(config):
    if config['use_cutout'] and config['use_dual_cutout']:
        raise ValueError(
            'Only one of `use_cutout` and `use_dual_cutout` can be `True`.')
    if sum([
            config['use_mixup'], config['use_ricap'], config['use_dual_cutout']
    ]) > 1:
        raise ValueError(
            'Only one of `use_mixup`, `use_ricap` and `use_dual_cutout` can be `True`.'
        )
    if config['dataset'] == 'spheres':
        if config['sphere_outer_rad'] <= config['sphere_inner_rad']:
            raise ValueError(
                    'In the spheres dataset, `sphere_outer_rad` must be greater than `sphere_inner_rad`')
        if not isinstance(config['sphere_dim'], int) or config['sphere_dim'] <= 2:
            raise ValueError(
                    'In the spheres dataset, `sphere_dim` must be an integer greater than 2')
        if not isinstance(config['sphere_trainset_size'], int) or config['sphere_trainset_size'] <= 0:
            raise ValueError(
                    'In the spheres dataset, `sphere_trainset_size` must be an integer greater than 0`')
        if not isinstance(config['sphere_testset_size'], int) or config['sphere_testset_size'] <= 0:
            raise ValueError(
                    'In the spheres dataset, `sphere_testset_size` must be an integer greater than 0`')
        if config['sphere_testset_size'] % 2 != 0 or config['sphere_trainset_size'] % 2 != 0:
            raise ValueError(
                    'In the spheres dataset, sizes of the training and test sets must be even')

def _get_run_config(args):
    keys = [
        'outdir',
        'seed',
        'test_first',
        'save_name',
        'count_params',
        'device',
        'fp16',
        'use_amp',
        'tensorboard',
        'tensorboard_train_images',
        'tensorboard_test_images',
        'tensorboard_model_params',
        'test_config'
    ]
    config = _args2config(args, keys, None)
    
    return config


def _get_env_info(args):
    info = OrderedDict({
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
    })

    def _get_device_info(device_id):
        name = torch.cuda.get_device_name(device_id)
        capability = torch.cuda.get_device_capability(device_id)
        capability = '{}.{}'.format(*capability)
        return name, capability

    if args.device != 'cpu':
        for gpu_id in range(torch.cuda.device_count()):
            name, capability = _get_device_info(gpu_id)
            info['gpu{}'.format(gpu_id)] = OrderedDict({
                'name':
                name,
                'capability':
                capability,
            })

    return info


def _cleanup_args(args):
    # architecture
    if args.arch == 'vgg':
        args.base_channels = None
        args.depth = None
    if args.arch != 'vgg':
        args.n_channels = None
        args.n_layers = None
        args.use_bn = None
    if args.arch not in [
            'resnet', 'resnet_preact', 'densenet', 'pyramidnet',
            'se_resnet_preact', 'standard_resnet'
    ]:
        args.block_type = None
    if args.arch not in ['resnet_preact', 'se_resnet_preact']:
        args.remove_first_relu = None
        args.add_last_bn = None
        args.preact_stage = None
    if args.arch != 'wrn':
        args.widening_factor = None
    if args.arch != 'densenet':
        args.growth_rate = None
        args.compression_rate = None
    if args.arch not in ['wrn', 'densenet']:
        args.drop_rate = None
    if args.arch != 'pyramidnet':
        args.pyramid_alpha = None
    if args.arch != 'resnext':
        args.cardinality = None
    if args.arch != 'shake_shake':
        args.shake_forward = None
        args.shake_backward = None
        args.shake_image = None
    if args.arch != 'se_resnet_preact':
        args.se_reduction = None

    # optimizer
    if args.optimizer not in ['sgd', 'lars']:
        args.momentum = None
    if args.optimizer != 'sgd':
        args.nesterov = None
    if args.optimizer != 'adam':
        args.betas = None
    if args.optimizer != 'lars':
        args.lars_eps = None
        args.lars_thresh = None

    # scheduler
    if args.scheduler != 'multistep':
        args.milestones = None
        args.lr_decay = None
    if args.scheduler not in ['cosine', 'sgdr']:
        args.lr_min = None
    if args.scheduler != 'sgdr':
        args.T0 = None
        args.Tmult = None

    # standard data augmentation
    if args.use_random_crop is None:
        if args.dataset in ['CIFAR10', 'CIFAR100', 'FashionMNIST', 'KMNIST']:
            args.use_random_crop = True
        else:
            args.use_random_crop = False
    if not args.use_random_crop:
        args.random_crop_padding = None
    if args.use_horizontal_flip is None:
        if args.dataset in ['CIFAR10', 'CIFAR100', 'FashionMNIST']:
            args.use_horizontal_flip = True
        else:
            args.use_horizontal_flip = False

    # (dual-)cutout
    if not args.use_cutout and not args.use_dual_cutout:
        args.cutout_size = None
        args.cutout_prob = None
        args.cutout_inside = None
    if not args.use_dual_cutout:
        args.dual_cutout_alpha = None

    # random erasing
    if not args.use_random_erasing:
        args.random_erasing_prob = None
        args.random_erasing_area_ratio_range = None
        args.random_erasing_min_aspect_ratio = None
        args.random_erasing_max_attempt = None

    # mixup
    if not args.use_mixup:
        args.mixup_alpha = None

    # RICAP
    if not args.use_ricap:
        args.ricap_beta = None

    # label smoothing
    if not args.use_label_smoothing:
        args.label_smoothing_epsilon = None

    # TensorBoard
    if not args.tensorboard:
        args.tensorboard_train_images = False
        args.tensorboard_test_images = False
        args.tensorboard_model_params = False

    # data
    if args.dataset in ['CIFAR10', 'CINIC10', 'CINIC10Enlarged']:
        args.input_shape = (1, 3, 32, 32)
        args.n_classes = 10
    elif args.dataset == 'CIFAR100':
        args.input_shape = (1, 3, 32, 32)
        args.n_classes = 100
    elif 'MNIST' in args.dataset:
        args.input_shape = (1, 1, 28, 28)
        args.n_classes = 10
    elif args.dataset == 'spheres':
        args.input_shape = (1, args.sphere_dim)
        args.n_classes = 2

    return args


def _set_default_values(args):
    if args.config is not None:
        with open(args.config, 'r') as fin:
            config = json.load(fin)

        d_args = vars(args)
        for config_key, default_config in config.items():
            if config_key == 'env_info':
                continue

            for key, default_value in default_config.items():
                if key not in d_args.keys() or d_args[key] is None:
                    setattr(args, key, default_value)
   
    if args.dataset == 'spheres':
        # sphere dataset defaults
        sphere_defaults = {
                            'sphere_dim': 500,
                            'sphere_inner_rad': 1.,
                            'sphere_outer_rad': 1.3,
                            'sphere_trainset_size': 50000,
                            'sphere_testset_size': 50000,
                            'use_random_scale': False,
                            'use_random_reflection': False,
                            'random_scale_prob': 0.25,
                            'random_scale_var': 0.1,
                            'random_reflection_prob': 0.01
                          }

        d_args = vars(args)

        for key, default_value in sphere_defaults.items():
            if key not in d_args.keys() or d_args[key] is None:
                print("setting sphere {}".format(key))
                setattr(args, key, default_value)
        # end sphere dataset defaults

    return args


def get_config(args):
    if args.arch is None and args.config is None:
        raise RuntimeError(
            'One of args.arch and args.config must be specified')
    if args.config is None:
        args.config = 'src/configs/{}.json'.format(args.arch)

    args = _set_default_values(args)
    args = _cleanup_args(args)
    config = OrderedDict({
        'model_config': _get_model_config(args),
        'optim_config': _get_optim_config(args),
        'data_config': _get_data_config(args),
        'run_config': _get_run_config(args),
        'env_info': _get_env_info(args),
    })
    return config
