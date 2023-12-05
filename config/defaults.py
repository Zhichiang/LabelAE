# Code style refers to https://github.com/facebookresearch/maskrcnn-benchmark
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import

import io
import six
import yaml
import numpy as np
import copy
import logging
from ast import literal_eval

from config.attrdict import AttrDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='default_config')

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C: AttrDict = AttrDict()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# # Get dataset path
# _C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = AttrDict()
_C.INPUT.image_height = 512
_C.INPUT.image_width = 512

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = AttrDict()
_C.DATASETS.name = 'cityscapes'
_C.DATASETS.split = ('train', 'val')
_C.DATASETS.num_works = 4
_C.DATASETS.root = ''

# ---------------------------------------------------------------------------- #
# model options
# ---------------------------------------------------------------------------- #

_C.MODEL = AttrDict()

_C.MODEL.name = 'multi'
_C.MODEL.latent_len = 10
_C.MODEL.select_latent = False
_C.MODEL.select_layer = False
_C.MODEL.aeweight = False
_C.MODEL.workspace = "workspace"
_C.MODEL.chkpt_dir = _C.MODEL.workspace
_C.MODEL.logs_dir = _C.MODEL.workspace
_C.MODEL.use_gpu = True
_C.MODEL.gpu_id = '0'
_C.MODEL.gpu_ids = '3'

_C.MODEL.domainada = False
_C.MODEL.pretrained = False
_C.MODEL.mode = "edge"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = AttrDict()
_C.SOLVER.optimizer = 'Adam'
_C.SOLVER.momentum = 0.9
_C.SOLVER.weight_decay = 0.0005
_C.SOLVER.base_lr = 0.001
_C.SOLVER.base_lr_d = 1e-3
_C.SOLVER.lambda_fd = 1e-3
_C.SOLVER.edge_r = 0.0

_C.SOLVER.lr_scheduler = 'StepLR'
_C.SOLVER.num_epochs = 20
_C.SOLVER.max_iters = 0  # 40000
_C.SOLVER.power = 0.9
_C.SOLVER.lr_decay_step = 20
_C.SOLVER.lr_decay = 0.1

_C.SOLVER.save_chkpt_each = 2500
_C.SOLVER.save_output_images = False
_C.SOLVER.writer_sample_each = 100
_C.SOLVER.val_writer_sample_each = 100
_C.SOLVER.val_calc_each = 1000

_C.SOLVER.on_device = "182"

# Number of images per batch
_C.SOLVER.image_per_batch = 4
_C.SOLVER.val_image_per_batch = 4

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = AttrDict()
# Number of images per batch
_C.TEST.image_per_batch = 8


#####################################################
_RENAMED_KEYS = {}
_DEPRECATED_KEYS = []


def load_cfg(cfg_to_load):
    """Wrapper around yaml.load used for maintaining backward compatibility"""
    file_types = [file, io.IOBase] if six.PY2 else [io.IOBase]  # noqa false positive
    expected_types = tuple(file_types + list(six.string_types))
    assert isinstance(cfg_to_load, expected_types), \
        'Expected one of {}, got {}'.format(expected_types, type(cfg_to_load))
    if isinstance(cfg_to_load, tuple(file_types)):
        cfg_to_load = ''.join(cfg_to_load.readlines())
    return yaml.load(cfg_to_load)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(load_cfg(f))
    _merge_a_into_b(yaml_cfg, _C)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                logger.error('Non-existent config key: {}. Ignored for now.'.format(full_key))
                continue

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _key_is_deprecated(full_key):
    if full_key in _DEPRECATED_KEYS:
        logger.warning(
            'Deprecated config key (ignoring): {}'.format(full_key)
        )
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
        format(full_key, new_key, msg)
    )


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
