# Takes in the path with the csvs from model outputs
# Computes for each csv the top model per category

import sys
import os
import pandas
import numpy as np

from natsort import natsorted

if len(sys.argv) < 2:
  raise RuntimeError("Please provide the path to the csvs from training/validation.")

csv_path = sys.argv[1]
print(f'CSV Path: {csv_path}')

files = os.listdir(f'{csv_path}/')
files = natsorted(files) # Sorted in order

print("================================================================================")
best_epochs = []

for specific_file_path_name in files:
  _name, ext = os.path.splitext(specific_file_path_name)

  if ext == '.csv':
    print(f'File: {specific_file_path_name}')

    # manually add in the header for now
    csv_frame = pandas.read_csv(f'{csv_path}/{specific_file_path_name}', header='infer')
    print(f'Count: {len(csv_frame)}\n')

    better_loss_sub_frame = csv_frame.loc[csv_frame['loss'] < 1.0]

    if better_loss_sub_frame.size < 1:
      better_loss_sub_frame = csv_frame.loc[csv_frame['loss'] < 1.5]

    min_loss = csv_frame["loss"].min()
    max_iou2 = better_loss_sub_frame["iou2"].max()
    max_iou3 = better_loss_sub_frame["iou3"].max()

    epochs = []
    loss_row = csv_frame.loc[csv_frame["loss"] == min_loss]
    iou2_row = csv_frame.loc[csv_frame["iou2"] == max_iou2]
    iou3_row = csv_frame.loc[csv_frame["iou3"] == max_iou3]

    epochs.append(loss_row["loss"].index[0])
    if iou2_row.size > 0:
      epochs.append(iou2_row["loss"].index[0])

    if iou3_row.size > 0:
      epochs.append(iou3_row["loss"].index[0])

    best_epochs.append(loss_row["loss"].index[0])

    if iou2_row.size > 0:
      best_epochs.append(iou2_row["loss"].index[0])

    if iou3_row.size > 0:
      best_epochs.append(iou3_row["loss"].index[0])

    print(f'Min Loss\n: {loss_row}')
    print(f'Max IOU2\n: {iou2_row}')
    print(f'Max IOU3\n: {iou3_row}')

    print(f'\nEpochs: {epochs}')
    print("================================================================================")

best = np.unique(np.array(best_epochs))
print(f'Best epochs: {best}')
print("Complete!")
