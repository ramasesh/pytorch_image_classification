import numpy as np
import glob
import json
import matplotlib.pyplot as plt

from src.utils import load_standard_measurements

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

def plot_data(all_data, depth, basechannels):

  f, ax = plt.subplots(1,2, figsize=(16,8))

  colors = ['m', '#FFA500', 'b', 'g', 'r','k', 'c']

  learning_rates = [3., 1., 0.3, 0.1, 0.03, 0.01]


  current_data = all_data[depth][basechannels]
  for key in sorted(current_data.keys()): # this is just to make the legend come out right
    lr = learning_rates[key]
    ax[0].plot(current_data[key]['train_loss'], label=f'$\eta$={lr}', ls='dotted', color=colors[key])
    ax[0].plot(current_data[key]['test_loss'], color=colors[key])
    ax[1].plot(current_data[key]['train_acc'], label=f'$\eta$={lr}', ls='dotted', color=colors[key])
    ax[1].plot(current_data[key]['test_acc'],  color=colors[key])


  # TODO set this to number of classes
  ax[0].set_ylim(0, np.log(100) + 0.1)
  ax[1].set_ylim(0, 1)
  ax[0].axhline(np.log(100), ls='dashed', color='black')
  ax[0].set_ylabel('Cross-entropy loss')
  ax[1].set_ylabel('Accuracy')

  f.suptitle(f"Depth = {depth}, Base Channels = {basechannels}", fontsize=24)

  for i in range(2):
    ax[i].set_xlabel('Epoch')
    ax[i].grid()
    ax[i].axvline(80, ls='dashed', color='gray')
    ax[i].axvline(120, ls='dashed', color='gray')

  ax[0].legend(loc='upper center',  fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.05), ncol=7)

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



