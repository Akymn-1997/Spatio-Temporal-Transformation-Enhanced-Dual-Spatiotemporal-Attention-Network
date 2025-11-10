import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DataLD
from sklearn.preprocessing import MinMaxScaler
import time
import torch.nn.functional as F

from STAN_Functions import STAN,train,create_sequences_and_targets,test
# torch.manual_seed(114514)
# np.random.seed(114514)
i = 23 # prediction interval
print(f'Case of dataset 1 with forecasting interval of {i+1} Hour(s) ahead')
import matplotlib.pyplot as plt
# Data Preprocessing
raw_csv = np.array(pd.read_csv("Wind Spatio-Temporal Dataset2.csv"))

cluster_turb_index = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 52, 53, 61, 62, 63, 64, 65, 66],
    [54, 57, 59, 60, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 112, 126],
    [40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 55, 56, 58, 89, 92, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 150, 151, 152, 153, 154, 155, 157, 158, 159, 160, 162, 163, 165, 167, 171, 172],
    [132, 147, 149, 156, 161, 164, 166, 168, 169, 170, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
]

Affine_parameters = [[46.58126117,15.39979395],[34.16539933,9.59594637],[71.07281113,23.28328517],[34.96746878,8.95020157]]

benchmark_turb_index = [23,70,134,193]

data = raw_csv[4:,2:402:2].astype(float)

data[np.isnan(data)] = 0 # Wind power with zero interpolation

scaler = MinMaxScaler()
scaler_global = MinMaxScaler()

normalized_data = scaler.fit_transform(data)

scaler_global.fit_transform(data.reshape(-1, 1))

benchmark_turb_index_index = [23,70,134,193]

# Definition of the hyperparameters and running device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timelen = 10
turbnum = 200
epochs = 100
batch_size = 512
learning_rate = 1e-3

# Datasets formation
X, y = create_sequences_and_targets(normalized_data, timelen,pred_interval=i)
print("X shape:", X.shape)  # (8750, turbnum, timelen)

print("y shape:", y.shape)  # (8750, 1)

scaler = MinMaxScaler()

# y = scaler.fit_transform(y)

train_ratio = .6
valid_ratio = .2
train_valid = int((y.shape[0])*(train_ratio))
valid_test = int((y.shape[0])*(train_ratio+valid_ratio))

train_X = X[:train_valid]
train_y = y[:train_valid].reshape(-1,1,1)
validate_X = X[train_valid:valid_test]
validate_y = y[train_valid:valid_test].reshape(-1,1,1)
test_X = X[valid_test:]
test_y = y[valid_test:].reshape(-1,1,1)

train_dataset = DataLD.TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
valid_dataset = DataLD.TensorDataset(torch.tensor(validate_X), torch.tensor(validate_y))
test_dataset = DataLD.TensorDataset(torch.tensor(test_X), torch.tensor(test_y))
train_loader = DataLD.DataLoader(train_dataset, batch_size=batch_size, shuffle=True ,drop_last=False)
valid_loader = DataLD.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLD.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = STAN(time_len=timelen,turb_num=turbnum,cluster_index=cluster_turb_index,A_parameters=Affine_parameters,benchmark_index=benchmark_turb_index)

net.to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

best_valid_loss = float('inf')

training_time = 0
for epoch in range(epochs):

  start_time = time.time()
  train_loss, valid_loss = train(net, device, train_loader, optimizer, criterion, -1,valid_loader,)
  end_time = time.time()
  training_time += end_time-start_time

  print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, Best Valid Loss: {best_valid_loss:.6f}",)

  if valid_loss < best_valid_loss:

      best_valid_loss = valid_loss
      best_model = net
      start_time = time.time()
      test_loss, final_predict,ground_truth = test(net, device, test_loader, criterion, interval = i)
      end_time = time.time()
      inference_time = start_time - end_time
      print('inference time:',inference_time)

  print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.6f}")

print('Total training time:',training_time)

final_predict = scaler_global.inverse_transform(np.array(final_predict).reshape(-1,1))
ground_truth = scaler_global.inverse_transform(np.array(ground_truth).reshape(-1,1))

df = pd.DataFrame({
    'final_predict': final_predict.reshape(-1),
    'ground_truth': ground_truth.reshape(-1)
})

df.to_csv(f'STAN{1}_{i+1}Hour.csv', index=False)
