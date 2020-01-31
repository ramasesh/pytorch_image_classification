import re
import json
import itertools as it
from src import utils
from caliban import util

BASE_BUCKET_NAME = 'gs://ramasesh-bucket-1/'

def load_config(filename):
  """ loads a Caliban-style experiment configuration dictionary (or list of dictionaries)
  from the json file """

  with open(filename, 'rb') as f:
      data = json.load(f)

  return data

def load_data_from_dict(config_dict, loading_function):

  location_string = config_dict['save_name']
  all_configurations = enumerate_configurations(config_dict)

  return load_data(location_string, all_configurations, loading_function)

def load_data(location_string, configurations, loading_function):
  """ given a location string, e.g.
  CIFAR100/LRSched_{scheduler}_LRDecay_{decay}/
  which specifies a folder for data, and a list of configurations"""

  all_data = {}
  for configuration in configurations:
    try:
      data_location = populate_loc_string(location_string, configuration)
      data_tuple = dict_vals_tuple(location_string, configuration)
      all_data[data_tuple] = loading_function(data_location)
    except:
      print(f"Could not load data for configuration {configuration}")
  return all_data

def enumerate_configurations(m, debug_lvl=0):
  """ converts a Caliban-style experiment config dictionary into a list of dictionaries,
  one for each configuration. """

  #TODO replace this with caliban.utils.dict_product as soon as that change becomes approved

  def is_disguised_list(s):
    if not isinstance(s, str):
      return False
    else:
      return s[0] == '[' and s[-1] == ']'

  def wrap_v(k, v):
    if not is_disguised_list(k):
      check_if_list = v
    else:
      check_if_list = v[0]
    return v if isinstance(check_if_list, list) else [v]

  cleaned = {k: wrap_v(k, v) for k, v in m.items()}

  ks = cleaned.keys()
  vs = cleaned.values()

  cartesian_product = (dict(zip(ks, x)) for x in it.product(*vs))

  fully_expanded = (expand_key_lists(d) for d in cartesian_product)

  return fully_expanded

def expand_key_lists(m):
  # TODO kill this once the caliban change becomes approved

  expanded_m = {}

  def is_disguised_list(s):
    if not isinstance(s, str):
      return False
    else:
      return s[0] == '[' and s[-1] == ']'

  def expand_disguised_list(s):
    return s.strip('][').split(',')

  for k, v in m.items():
    if is_disguised_list(k):
      for i, ki in enumerate(expand_disguised_list(k)):
        expanded_m.update({ki: v[i]})
    else:
      expanded_m.update({k:v})
  return expanded_m

def populate_loc_string(location_string, configuration):
  """ takes a string with some fields to populate, e.g.
  location_string = "CIFAR100_{lr_decay}_{depth}",
  and a configuration with the values of the parameters in the string, e.g.
  configuration = {'lr_decay': 0.01, 'depth': 10},
  and populates the string, returning
  "CIFAR10_0.01_10"
  """
  return location_string.format(**configuration)

def all_folders(config):
  """ converts a Caliban-style experiment config dictionary or list of dictionaries into a list of folders,
  which can then be loaded, e.g. from a bucket """

  if type(config) is dict:
    all_configurations = enumerate_configurations(config)
    return [populate_loc_string(config['save_name'], configuration) for configuration in all_configurations]
  elif type(config) is list:
    all_foldernames = []
    for config_dict in config:
      all_foldernames.extend(all_folders(config_dict))
    return(all_foldernames)

def product_dict(**kwargs):
  """ given a kwargs dictionary, all of whose items are lists,
  returns a generator which yields dictionaries that are the cartesian product
  of all the lists"""
  keys = list(kwargs.keys())
  vals = list(kwargs.values())

  # convert all vals to list
  for i in range(len(vals)):
    if type(vals[i]) is not list:
      vals[i] = [vals[i]]

  for instance in it.product(*vals):
    yield dict(zip(keys, instance))

def extract_in_braces(string):
  """ extracts all strings in braces {} from the given string
  Note: Cannot handle nested braces """
  found_in_braces = re.finditer(r'\{.*?\}', string)

  select_match = lambda x: x.group(0)
  remove_outer = lambda x: x[1:-1]

  found_in_braces = map(select_match, found_in_braces)
  found_in_braces = map(remove_outer, found_in_braces)

  return list(found_in_braces)

def dict_vals_tuple(location_string, dictionary):

  keys_to_keep = extract_in_braces(location_string)
  sorted_dict_keys = sorted(keys_to_keep)

  return tuple([dictionary[k] for k in sorted_dict_keys])

# TODO Move this somewhere else
# This is specific to how we load data from the buckets, so I don't want it in
# this code

import os

def grab_from_bucket(config, destination_directory, test_command=False, file_extension=None):
  """ allows either config to be a dictionary itself, or a list of config dictionaries """
  # please specify file_extension with a '.'

  #TODO Current problem: if you add file extensions, it copies the files directly into the destination directory without 
  #   the final foldername

  if type(config) is str:
    config = load_config(config)
  all_foldernames = all_folders(config)

  all_folders_list =[BASE_BUCKET_NAME + folder_name for folder_name in all_foldernames]

  if file_extension is not None:
    #TODO
    all_folders_list = [folder_name + file_extension for folder_name in all_folders_list]

  all_folders_string = ' '.join(all_folders_list)

  command_to_run = f'mkdir -p {destination_directory} && gsutil -m cp -r {all_folders_string} {destination_directory}'

  if test_command:
    print(command_to_run)
  else:
    os.system(f'mkdir -p {destination_directory} && gsutil -m cp -r {all_folders_string} {destination_directory}')

def grab_and_process(config_filename, test=False):
  """ run this in project root """

  if type(config_filename) == str:
    config = load_config(config_filename)
  else:
    config = config_filename


  if type(config) == list:
    data = []
    for indiv_config in config:
      indiv_data = grab_and_process(indiv_config,test=test)
      indiv_data['config'] = indiv_config
      data.append(indiv_data)
  else:
    save_loc = base_foldername(config['save_name'])
    grab_from_bucket(config, save_loc, test_command=test)
    if not test:
      data = load_data_from_dict(config, utils.load_standard_from_folder)
    else: 
      data = []

  return data

def base_foldername(foldername):
  if foldername[-1] == '/':
    foldername = foldername[:-1]

  foldername = foldername[::-1]
  last_slash = foldername.index('/')
  foldername = foldername[::-1]

  foldername = foldername[:len(foldername) - last_slash]

  return foldername
