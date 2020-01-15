import re
import json
import itertools

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
    data_location = populate_loc_string(location_string, configuration)
    data_tuple = dict_vals_tuple(configuration)
    print(data_location)
    print(data_tuple)
    all_data[data_tuple] = loading_function(data_location)

  return all_data

def enumerate_configurations(config_dict, debug_lvl=0):
  """ converts a Caliban-style experiment config dictionary into a list of dictionaries,
  one for each configuration. """

  arguments_to_save = extract_in_braces(config_dict['save_name'])
  save_args = {argument: config_dict[argument] for argument in arguments_to_save}

  if debug_lvl > 0:
    print(f"save_args: {save_args}")

  all_configurations = list(product_dict(**save_args))

  if debug_lvl > 0:
    for configuration in all_configurations:
      print(configuration)

  return all_configurations

def populate_loc_string(location_string, configuration):
  """ takes a string with some fields to populate, e.g.
  location_string = "CIFAR100_{lr_decay}_{depth}",
  and a configuration with the values of the parameters in the string, e.g.
  configuration = {'lr_decay': 0.01, 'depth': 10},
  and populates the string, returning
  "CIFAR10_0.01_10"
  """ 
  return location_string.format(**configuration)

def all_folders(config_dict):
  """ converts a Caliban-style experiment config dictionary into a list of folders,
  which can then be loaded, e.g. from a bucket """

  all_configurations = enumerate_configurations(config_dict)
  return [populate_loc_string(config_dict['save_name'], configuration) for configuration in all_configurations]

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

  for instance in itertools.product(*vals):
    yield dict(zip(keys, instance))

def extract_in_braces(string):
  """ extracts all strings in braces {} from the given string
  Note: Cannot handle nested braces """
  found_in_braces = re.finditer(r'\{.*?\}', string)

  select_match = lambda x: x.group(0)
  remove_outer = lambda x: x[1:-1]

  found_in_braces = map(select_match, found_in_braces)
  found_in_braces = map(remove_outer, found_in_braces)

  return found_in_braces


def dict_vals_tuple(dictionary):
  sorted_dict_keys = sorted(dictionary.keys())
  return tuple([dictionary[k] for k in sorted_dict_keys])

# TODO Move this somewhere else
# This is specific to how we load data from the buckets, so I don't want it in
# this code

import os

def grab_from_bucket(config, destination_directory):
  """ allows either config to be a dictionary itself, or a list of config dictionaries """
  #TODO make this exclude certain filenames if you don't want to copy them over
  if type(config) is str:
    config = load_config(config)

  if type(config) is dict:
    all_foldernames = all_folders(config)
  elif type(config) is list:
    all_foldernames = []
    for config_dict in config:
      all_foldernames.extend(all_folders(config_dict))

  base_bucket_name = 'gs://ramasesh-bucket-1/'
  all_folders_list =[base_bucket_name + folder_name for folder_name in all_foldernames] 
  all_folders_string = ' '.join([base_bucket_name + folder_name for folder_name in all_foldernames])

  os.system(f'gsutil -m cp -r {all_folders_string} {destination_directory}')

