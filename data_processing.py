import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import itertools as it

from typing import Dict, List, Any
from src.utils import load_standard_measurements
from src.data_extraction import extract_in_braces, load_config

def load_single_arch(basedir, depth, basechannels):
  dirname = f'{basedir}/resnet_depth_{depth}_basechannels_{basechannels}'

  all_files = glob.glob(f'{dirname}/*/*')
  matching_files = [f for f in all_files if 'log.json' in f]

  architecture_data = {}
  for f in matching_files:
      current_key = extract_run_number(f)
      architecture_data[current_key] = load_standard_measurements(f)

  return architecture_data

def extract_run_number(filename):
  return int(filename.split('/')[-2])

def load_all_archs(basedir, depths, basechannels):
  all_data = {}

  for depth in depths:
    all_data[depth] = {b : load_single_arch(basedir, depth, b) for b in basechannels}
  return all_data


def plot_data(all_data, label_func, title, n_classes=10, savename=None):
  f, ax = plt.subplots(1,3, figsize=(24,8))

  colors = ['b', 'g', 'r', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

  for ind, key in enumerate(all_data.keys()):
    label = label_func(key)
    ax[0].plot(all_data[key]['train_loss'], label=label, ls='dotted', color=colors[ind])
    ax[0].plot(all_data[key]['test_loss'], color=colors[ind])
    ax[1].plot(all_data[key]['train_acc'], label=label, ls='dotted', color=colors[ind])
    ax[1].plot(all_data[key]['test_acc'],  color=colors[ind])
    ax[2].scatter(label, all_data[key]['test_acc'][-1], color=colors[ind])
    ax[2].scatter(label, all_data[key]['test_loss'][-1], marker='+', color=colors[ind])


  ax[0].set_ylim(0, np.log(n_classes) + 0.1)
  ax[1].set_ylim(0, 1)
  ax[0].axhline(np.log(n_classes), ls='dashed', color='black')
  ax[0].set_ylabel('Cross-entropy loss')
  ax[1].set_ylabel('Accuracy')
  ax[2].set_xlabel('Learning Rate')

  f.suptitle(title, fontsize=24)

  for i in range(2):
    ax[i].set_xlabel('Epoch')
    ax[i].grid()

  ax[0].legend(loc='upper center',  fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.05), ncol=7)

  return f

def plot_dict(all_data, title):
  f, ax = plt.subplots(1,3, figsize=(24,8))

  colors = ['b', 'g', 'r', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k']

  for ind, key in enumerate(all_data.keys()):
    if len(key) == 1:
      label = key[0]
    else:
      label = key
    ax[0].plot(all_data[key]['train_loss'], label=label, ls='dotted', color=colors[ind])
    ax[0].plot(all_data[key]['test_loss'], color=colors[ind])
    ax[1].plot(all_data[key]['train_acc'], label=label, ls='dotted', color=colors[ind])
    ax[1].plot(all_data[key]['test_acc'],  color=colors[ind])
    ax[2].scatter(label, all_data[key]['test_acc'][-1], color=colors[ind])
    ax[2].scatter(label, all_data[key]['test_loss'][-1], marker='+', color=colors[ind])


  ax[0].set_ylim(0, np.log(10) + 0.1)
  ax[1].set_ylim(0, 1)
  ax[0].set_ylabel('Cross-entropy loss')
  ax[1].set_ylabel('Accuracy')
  ax[2].set_xlabel('Learning Rate')

  f.suptitle(title, fontsize=24)

  for i in range(2):
    ax[i].set_xlabel('Epoch')
    ax[i].grid()

  ax[0].legend(loc='upper center',  fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.05), ncol=7)

  return f

def plot_full_datadict(full_datadict, attribute, config, plot_func, folder, basetitle):

  grouped_dict = group_datadict_by_attribute(full_datadict, attribute, config)

  for key, data in grouped_dict.items():
    plotted_figure = plot_func(data, f"{attribute} = {key}")
    plotted_figure.savefig(folder + f"{basetitle}_{attribute}_{key}.png")

def plot_optimum_test_acc(full_datadict, attribute):

  config = full_datadict['config']

  named_arguments = list(extract_in_braces(config['save_name']))
  attribute_index = named_arguments.index(attribute)

  extraction_func = lambda x: x['test_acc'][-1]
  extracted = extract_measurement(full_datadict, extraction_func)
  summarized = summarize_measurements(extracted, attribute_index, np.max)

  x = sorted(summarized.keys())
  y = 1 - np.array([summarized[xi] for xi in x])

  f = plot_loglog(x,y,yname='test error')
  return x,y,f


def group_datadict_by_attribute(full_datadict, attribute, config):
  """ full_datadict has keys which are tuples, and we want to group the dictionary by attributes """

  if isinstance(config, str):
    config = load_config(config)

  named_arguments = list(extract_in_braces(config['save_name']))
  attribute_index = named_arguments.index(attribute)

  grouped_dict =  group_dict_by_dim(full_datadict, [attribute_index])

  modification_func = lambda k: tuple([k[i] for i in range(len(k)) if i != attribute_index])
  modified_dicts = {k: modify_keys(v,modification_func) for k,v in grouped_dict.items()}

  return modified_dicts

def group_dict_by_dim(full_data, dimensions_to_group):
  """ takes a dataset with keys who are tuples, forms groups of keys which share the same values in
  'dimensions_to_group', and then forms sub-dictionaries out of this grouping.
  """


  def clean(d):
    # removes keys from d which are not tuples
    cleaned = {}
    for k,v in d.items():
      if isinstance(k, tuple):
        cleaned.update({k:v})
    return cleaned

  full_data = clean(full_data)

  key_grouping = group_by_dimensions(full_data.keys(), dimensions_to_group)
  grouped_dict = split_dict(full_data, key_grouping)


  return grouped_dict

def modify_keys(input_dict, key_transformation_func):
  """ applies key_transformation_func to evey key in input_dict """

  return {key_transformation_func(k): v for k,v in input_dict.items()}

def plot_loglog(x,y, yname):
  f, ax = plt.subplots()
  logx = np.log(x)
  logy = np.log(y)
  ax.scatter(logx,logy)

  m, b = np.polyfit(logx, logy, deg=1)

  ax.set_xlabel('Log Dataset Size')
  ax.set_ylabel(f'Log {yname}')

  ax.plot(logx, m*np.array(logx) + b)
  ax.set_title(f'M = {m}')
  return f

def optimum_array(all_data, depths, basechannels, msmt_type='train_acc'):

  if 'acc' in msmt_type:
    comp_fn = np.max
  elif 'loss' in msmt_type:
    comp_fn = np.min

  optimum_dict = {depth: {bc: comp_fn([all_data[depth][bc][i][msmt_type] for i in range(len(all_data[depth][bc].keys()))]) for bc in basechannels} for depth in depths}
  optimum_list = [[optimum_dict[d][b] for d in depths] for b in basechannels]

  return np.array(optimum_list)

def imshow_with_vals(data, x_axis, y_axis, xlabel, ylabel, title):
  fig, ax = plt.subplots()
  im = ax.imshow(data)

  ax.set_xticks(np.arange(len(x_axis)))
  ax.set_yticks(np.arange(len(y_axis)))
  ax.set_xticklabels(x_axis)
  ax.set_yticklabels(y_axis)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  for i in range(len(y_axis)):
    for j in range(len(x_axis)):
      text = ax.text(j,i,np.round(data[i,j],2), ha="center", va="center", color="w")

  fig.tight_layout()

  return fig

def extract_measurement(full_measurement_dictionary, extraction_function):
  """ applies the extraction function to each value in the dictionary and returns a dictionary
  with the extracted measurements """

  def clean(d):
    # removes keys from d which are not tuples
    cleaned = {}
    for k,v in d.items():
      if isinstance(k, tuple):
        cleaned.update({k:v})
    return cleaned

  full_measurement_dictionary = clean(full_measurement_dictionary)

  return {k: extraction_function(full_measurement_dictionary[k]) for k in full_measurement_dictionary.keys()}

def summarize_measurements(full_measurement_dictionary, dimensions_to_summarize, summarize_function):
  """ e.g.
  full_measurement_dictionary has keys of the form (learning_rate, dataset_size), with the value being some
  test accuracy, and I want to maximize over learning rate, so then I run
    summarize_measurements(full_measurement_dictionary, dimensions_to_summarize=[0], summarize_function = np.max)
  """

  key_groupings = group_by_dimensions(full_measurement_dictionary.keys(), dimensions_to_summarize)

  return {k: summarize_function([full_measurement_dictionary[l] for l in v]) for k, v in key_groupings.items()}

def split_dict(dict_to_split: Dict, splitting: Dict) -> Dict[Any, Dict]:
  """ splits a dictionary """

  return {k: subset_dict(dict_to_split, keys_to_keep=v) for k, v in splitting.items()}

def subset_dict(dict_to_subset: Dict, keys_to_keep: List) -> Dict:
  """ Takes selected items (specified by keys_to_keep) from dict_to_subset
  and forms a new dictionary from them """

  return {k: dict_to_subset[k] for k in keys_to_keep}


def group_by_dimensions(tuples_to_group, dimensions):
  """ Forms groups from a list of tuples (all of the same length), where """

  def wrap(x):
    """ makes x a list if it isn't already """
    if isinstance(x, list):
      return x
    else:
      return [x]

  def unwrap(x):
    """ if x is a length-0 list, returns the element """
    try:
      if len(x) == 1:
        return x[0]
      else:
        return x
    except:
      return x

  def key_func(input_tuple):
    return tuple(input_tuple[i] for i in wrap(dimensions))

  sorted_tups = sorted(tuples_to_group, key=key_func) # sorting must precede grouping
  G = it.groupby(sorted_tups, key_func)

  return {unwrap(x[0]): list(x[1]) for x in G}

def extract_ks_vs(dictionary):
  ks = dictionary.keys()
  return list(ks), [dictionary[k] for k in ks]
