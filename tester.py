import math
from PIL import Image, ImageOps
import numpy as np
import torch
import conv
import score
import pickle
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import statistics
from matplotlib.ticker import StrMethodFormatter


# dataset constraints: must have images of same size
# images of same format are used to compare performances on original & preprocessed samples (thus canceling out effects of padding) 
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


images = []
targets_values = []
targets_indices = []
f = open('data/validation.txt', 'r')

line = f.readline()

n_of_tested_files = 10000+1

i = 0

while line:

  line = line.split(sep=' ', maxsplit=1)

  file_name = line[0]
  target = line[1][:-1]

  img = Image.open('data/brno/' + file_name)
  img = add_margin(img, math.floor((59-img.size[1]) / 2), math.floor((1357-img.size[0]) / 2), math.ceil((59-img.size[1]) / 2), math.ceil((1357-img.size[0]) / 2), (255, 255, 255)).convert("RGB")
  img = np.array(img)

  images.append(img)

  targets_values.append(target)
  targets_indices.append(i)

  line = f.readline()
  i+=1

  if i >= n_of_tested_files:
      break

f.close()

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
device = torch.device("cuda" if use_cuda else "cpu")

# create testing set
test_batch_size = 1


test_data = data_utils.TensorDataset(torch.Tensor(images).permute(0, 3, 1, 2), torch.Tensor(targets_indices))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)

# set conv processor weights
conv_processor = conv.ConvProcessor().to(device)
conv_filters_file = open('conv-filters/conv_filters_-666.dat', mode='rb')
conv_weights = pickle.load(conv_filters_file)
conv_filters_file.close()

conv_processor.get_weights()
conv_processor.set_weights(conv_weights)


batch_ids = []
processed_cer_rewards = []
processed_wer_rewards = []
processed_lcser_rewards = []
processed_precision = []
processed_recall = []

non_processed_cer_rewards = []
non_processed_wer_rewards = []
non_processed_lcser_rewards = []
non_processed_precision = []
non_processed_recall = []

total_chars = 0

# run tests
for batch_id, (input, target) in enumerate(test_loader):
    input = input.to(device)
    target = target.to(device)

    # evaluate performance of preprocessing action and calculate reward      
    conv_result = conv_processor(input) 
    conv.n_of_channels = 1

    for i in range(test_batch_size):
      cer_reward, wer_reward, lcser_reward, precision, recall, crt_done_bool = score.compute_score(conv_result[i], targets_values[int(target[i].cpu().numpy())])
      
      total_chars += len(targets_values[int(target[i].cpu().numpy())])
      
      batch_ids.append(batch_id)
      processed_cer_rewards.append(cer_reward)
      processed_wer_rewards.append(wer_reward)
      processed_lcser_rewards.append(lcser_reward)
      processed_precision.append(precision)
      processed_recall.append(recall)


    # also evaluate with no preprocessing
    input = input.permute(0,2,3,1)
    input = torch.clamp(input, 0, 255).type(torch.uint8).cpu().numpy()

    conv.n_of_channels = 3

    for i in range(test_batch_size):
      cer_reward, wer_reward, lcser_reward, precision, recall, crt_done_bool = score.compute_score(input[i], targets_values[int(target[i].cpu().numpy())])
      
      non_processed_cer_rewards.append(cer_reward)
      non_processed_wer_rewards.append(wer_reward)
      non_processed_lcser_rewards.append(lcser_reward)
      non_processed_precision.append(precision)
      non_processed_recall.append(recall)


print("Total chars: ", total_chars)
print("Avg. chars: ", total_chars / n_of_tested_files)

print("Avg. processed CER: ", statistics.mean(processed_cer_rewards))
print("Avg. processed WER: ", statistics.mean(processed_wer_rewards))
print("Avg. processed LCSER: ", statistics.mean(processed_lcser_rewards))
print("Processed CER stdev: ", statistics.stdev(processed_cer_rewards))
print("Processed WER stdev: ", statistics.stdev(processed_wer_rewards))
print("Processed LCSER stdev: ", statistics.stdev(processed_lcser_rewards))

print("Avg. NON-processed CER: ", statistics.mean(non_processed_cer_rewards))
print("Avg. NON-processed WER: ", statistics.mean(non_processed_wer_rewards))
print("Avg. NON-processed LCSER: ", statistics.mean(non_processed_lcser_rewards))
print("NON-Processed CER stdev: ", statistics.stdev(non_processed_cer_rewards))
print("NON-Processed WER stdev: ", statistics.stdev(non_processed_wer_rewards))
print("NON-Processed LCSER stdev: ", statistics.stdev(non_processed_lcser_rewards))

print("Avg. processed precision ", statistics.mean(processed_precision))
print("Avg. processed recall ", statistics.mean(processed_recall))
print("Avg. NON-processed precision ", statistics.mean(non_processed_precision))
print("Avg. NON-processed recall ", statistics.mean(non_processed_recall))


##### BEGIN HISTOGRAMS

plt.figure(figsize=(5.6,1.7))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
plt.rcParams.update({'font.size': 9})
plt.grid(True)
plt.xlabel("Character Error Rate (CER)", fontsize=9)
plt.ylabel("Probability Density", fontsize=9)
plt.hist([processed_cer_rewards, non_processed_cer_rewards], 30, density=True, color=['g', 'r'], alpha=1)
plt.tight_layout()
plt.savefig("plots/cer-histogram.png")
plt.clf()

plt.figure(figsize=(5.6,1.7))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
plt.rcParams.update({'font.size': 9})
plt.grid(True)
plt.xlabel("Word Error Rate (WER)", fontsize=9)
plt.ylabel("Probability Density", fontsize=9)
plt.hist([processed_wer_rewards, non_processed_wer_rewards], 30, density=True, color=['g', 'r'], alpha=1)
plt.tight_layout()
plt.savefig("plots/wer-histogram.png")
plt.clf()

plt.figure(figsize=(5.6,1.7))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
plt.rcParams.update({'font.size': 9})
plt.grid(True)
plt.xlabel("Longest Common Subsequence Error (LCSE)", fontsize=9)
plt.ylabel("Probability Density", fontsize=9)
plt.hist([processed_lcser_rewards, non_processed_lcser_rewards], 30, density=True, color=['g', 'r'], alpha=1)
plt.tight_layout()
plt.savefig("plots/lcser-histogram.png")
plt.clf()

##### BEGIN SCATTER PLOTS

coef1 = np.polyfit(batch_ids, processed_cer_rewards, 1)
poly1d_fn1 = np.poly1d(coef1)

coef2 = np.polyfit(batch_ids, non_processed_cer_rewards, 1)
poly1d_fn2 = np.poly1d(coef2)

plt.figure(figsize=(1.8,5))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
plt.rcParams.update({'font.size': 9})
plt.grid(True)
plt.xlabel("Sample Index", fontsize=9)
plt.ylabel("Character Error Rate (CER)", fontsize=9)
plt.plot(batch_ids, processed_cer_rewards, 'go', markersize=1.5, alpha=0.03) 
plt.plot(batch_ids, poly1d_fn1(batch_ids), '--g', markersize=1, alpha=1)
plt.plot(batch_ids, non_processed_cer_rewards, 'ro', markersize=1.5, alpha=0.03) 
plt.plot(batch_ids, poly1d_fn2(batch_ids), '--r', markersize=1, alpha=1)
plt.tight_layout()
plt.savefig("plots/cer-scatter.png")
plt.clf()





coef1 = np.polyfit(batch_ids, processed_wer_rewards, 1)
poly1d_fn1 = np.poly1d(coef1)

coef2 = np.polyfit(batch_ids, non_processed_wer_rewards, 1)
poly1d_fn2 = np.poly1d(coef2)


plt.figure(figsize=(1.8,5))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
plt.rcParams.update({'font.size': 9})
plt.grid(True)
plt.xlabel("Sample Index", fontsize=9)
plt.ylabel("Word Error Rate (WER)", fontsize=9)
plt.plot(batch_ids, processed_wer_rewards, 'go', markersize=1.5, alpha=0.03) 
plt.plot(batch_ids, poly1d_fn1(batch_ids), '--g', markersize=1, alpha=1)
plt.plot(batch_ids, non_processed_wer_rewards, 'ro', markersize=1.5, alpha=0.03) 
plt.plot(batch_ids, poly1d_fn2(batch_ids), '--r', markersize=1, alpha=1)
plt.tight_layout()
plt.savefig("plots/wer-scatter.png")
plt.clf()




coef1 = np.polyfit(batch_ids, processed_lcser_rewards, 1)
poly1d_fn1 = np.poly1d(coef1)

coef2 = np.polyfit(batch_ids, non_processed_lcser_rewards, 1)
poly1d_fn2 = np.poly1d(coef2)


plt.figure(figsize=(1.8,5))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
plt.rcParams.update({'font.size': 9})
plt.grid(True)
plt.xlabel("Sample Index", fontsize=9)
plt.ylabel("Longest Common Subsequence Error (LCSE)", fontsize=9)
plt.plot(batch_ids, processed_lcser_rewards, 'go', markersize=1.5, alpha=0.03) 
plt.plot(batch_ids, poly1d_fn1(batch_ids), '--g', markersize=1, alpha=1)
plt.plot(batch_ids, non_processed_lcser_rewards, 'ro', markersize=1.5, alpha=0.03) 
plt.plot(batch_ids, poly1d_fn2(batch_ids), '--r', markersize=1, alpha=1)
plt.tight_layout()
plt.savefig("plots/lcser-scatter.png")
plt.clf()


non_proc_score = 0
proc_score = 0
for i in range(len(processed_cer_rewards)):
  non_proc_score += non_processed_cer_rewards[i]
  proc_score += processed_cer_rewards[i]
print("Scores of proc vs non-proc: ", proc_score, non_proc_score)
print("Improvement percent: ", (non_proc_score - proc_score) / (non_proc_score)* 100)



