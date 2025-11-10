# STAN: A Spatio-Temporal Transformation-based STAN for Hour-Ahead Wind Turbine Cluster Forecasting
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def create_sequences_and_targets(data, timelen,  pred_interval = 0):
    num_samples = data.shape[0] - timelen - pred_interval
    num_features = data.shape[1]

    X = np.zeros((num_samples, num_features, timelen))

    y = np.zeros((num_samples, 1))

    for i in range(num_samples):
        X[i] = data[i:i + timelen].T
        y[i] = sum(data[i + timelen + pred_interval])

    return X, y

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, input_dim, d_model=16, n_heads=4, ff_dim=16,dropout=.3):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, input_dim]
        x = self.embedding(x)         # [B, N, d_model]
        x = self.dropout(x)
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn_weights

class STAN(nn.Module):
    def __init__(self, time_len=10,turb_num=50, d_model=8, n_heads=4, ff_dim=16, num_layers=2,dropout=.0,A_parameters=None,cluster_index=[],benchmark_index=[]):
        super().__init__()
        self.time_len = time_len
        self.turb_num = turb_num
        print(turb_num)
        self.d_model = d_model
        self.cluster_index = cluster_index
        self.benchmark_index = benchmark_index
        self.cluster_num = len(self.benchmark_index)
        self.A_parameters = A_parameters
        self.acti = nn.ReLU()

        self.encoder_layers_spatial_groups = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(
                    input_dim=time_len if i == 0 else d_model,
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                for i in range(num_layers)
            ])
            for g in range(self.cluster_num)
        ])

        self.encoder_layers_temporal_groups = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(
                    input_dim=len(self.cluster_index[g]) if i == 0 else d_model,
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                for i in range(num_layers)
            ])
            for g in range(self.cluster_num)
        ])

        self.encoder_layers_spatial_temporal_groups = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(
                    input_dim=d_model,
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                for i in range(num_layers)
            ])
            for g in range(self.cluster_num)
        ])

        self.lstm = nn.ModuleList([
            nn.LSTM(d_model, d_model, num_layers, dropout=dropout, batch_first=True)
            for g in range(self.cluster_num)
        ])

        self.final_output_layer = nn.ModuleList([
            nn.Linear(self.d_model, d_model)
            for g in range(self.cluster_num)
        ])

        self.res_proj = nn.ModuleList([
            nn.Linear(self.time_len, d_model)
            for g in range(self.cluster_num)
        ])

        self.final_proj_1 = nn.ModuleList([
            nn.Linear(time_len+len(self.cluster_index[g]), len(self.cluster_index[g]))
            for g in range(self.cluster_num)
        ])

        self.final_proj_2 = nn.ModuleList([
            nn.Linear(d_model, 1)
            for g in range(self.cluster_num)
        ])

    def forward(self, input):
        outputs = []

        for i in range(self.cluster_num):
            x_spatial = x_spatial_ = input[:,self.cluster_index[i],:]

            x_temporal = x_spatial_.view(-1, self.time_len, len(self.cluster_index[i]))

            for layer in self.encoder_layers_spatial_groups[i]:
                x_spatial, attn = layer(x_spatial)

            for layer in self.encoder_layers_temporal_groups[i]:
                x_temporal, attn = layer(x_temporal)

            x_temporal,_ = self.lstm[i](x_temporal)

            x = torch.cat((x_temporal, x_spatial), dim=1)

            for layer in self.encoder_layers_spatial_temporal_groups[i]:
                x, attn = layer(x)

            k = self.A_parameters[i][0]
            b = self.A_parameters[i][1]

            # x = self.acti(self.final_output_layer[i](x))

            x = x * torch.tensor(k, device=x.device, dtype=x.dtype) + torch.tensor(b, device=x.device, dtype=x.dtype) + x

            res_connection = self.acti(self.res_proj[i](x_spatial_))

            x = self.final_proj_1[i](x.view(-1, self.d_model, len(self.cluster_index[i]) + self.time_len))

            x = self.acti(x)

            x = res_connection + x.view(-1, len(self.cluster_index[i]), self.d_model)

            output = self.final_proj_2[i](x)

            outputs.append(output[:,-1])

        return torch.stack(outputs, dim=0).sum(dim=0)

def train(model, device, train_loader, optimizer, criterion, center=0,valid_loader=None):
    model.train()
    train_loss = 0.0
    valid_loss = 0.0

    for data, labels in train_loader:
        data, labels = data.float().to(device), labels.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        prediction = output
        loss = criterion(prediction.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    if valid_loader is not None:
        model.eval()
        with torch.no_grad():
            for data, labels in valid_loader:
                data, labels = data.float().to(device), labels.float().to(device)
                output = model(data)
                prediction = output
                loss = criterion(prediction.view(-1), labels.view(-1))
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)

    return train_loss, valid_loss

def test(model, device, test_loader, criterion=None,interval = 0):
    model.eval()
    running_loss = 0.0
    final_predict = []
    ground_truth = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.float().to(device), labels.float().to(device)
            output = model(data)
            prediction = output
            loss = criterion(prediction.view(-1), labels.view(-1))
            running_loss += loss.item()
            final_predict = final_predict + prediction.view(-1).cpu().numpy().tolist()
            ground_truth = ground_truth + labels.view(-1).cpu().numpy().tolist()

    return running_loss / len(test_loader),final_predict,ground_truth
