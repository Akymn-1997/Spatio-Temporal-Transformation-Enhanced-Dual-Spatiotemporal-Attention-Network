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
raw_csv = np.array(pd.read_csv("hourly_wtbdata_245days.csv"))

cluster_turb_index = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 44,
     45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 90, 91, 92, 93, 94, 95, 96, 112, 113, 114, 115, 116, 119],
    [15, 16, 17, 18, 19, 20, 39, 40, 41, 42, 43, 57, 58, 59, 60, 61,  62, 63, 64, 65, 66, 67, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
     89, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,  110, 111, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133]
]

Affine_parameters = [[67.55534749,14.99021851],[61.05266495,12.14009152]]

benchmark_turb_index = [49,86]

data = raw_csv[1:,1::2].astype(float)

data[np.isnan(data)] = 0 # Wind power with zero interpolation

scaler = MinMaxScaler()

scaler_global = MinMaxScaler()

normalized_data = scaler.fit_transform(data)

scaler_global.fit_transform(data.reshape(-1, 1))

# Definition of the hyperparameters and running device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timelen = 10
turbnum = 134
epochs = 100
batch_size = 512
learning_rate = 1e-4

# Datasets formation
X, y = create_sequences_and_targets(normalized_data, timelen,pred_interval=i)
print("X shape:", X.shape)  # (8750, turbnum, timelen)

print("y shape:", y.shape)  # (8750, 1)

scaler = MinMaxScaler()

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
      print(len(ground_truth))
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

df.to_csv(f'STAN{2}_{i+1}Hour.csv', index=False)
