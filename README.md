# STT-DSTAN-Based Wind Power Forecasting

This repository contains the implementation of a spatiotemporal attention network (STAN) for short-term wind power forecasting.  
The workflow involves clustering wind turbines based on their operational similarity, followed by forecasting models trained on each dataset.

---

## 📁 Project Structure

| File | Description |
|------|--------------|
| `K-means Clustering_Dataset_1.py` | Performs K-means clustering on Dataset 1 to generate turbine clusters. |
| `K-means Clustering_Dataset_2.py` | Performs K-means clustering on Dataset 2. |
| `STAN_Functions.py` | Contains model components, utility functions, and core STAN implementations. |
| `main_STAN_Dataset_1.py` | Main script for training and evaluating the STAN model on Dataset 1. |
| `main_STAN_Dataset_2.py` | Main script for training and evaluating the STAN model on Dataset 2. |

---

