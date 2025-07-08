# Deep Learning-based IndustrialNILM



This repository provides tools for preprocessing and modeling power consumption data for **Non-Intrusive Load Monitoring (NILM)** using **deep learning**. It is tailored for industrial environments where multiple machines of the same type operate concurrently.

## 📦 Modules

### `data_extraction.py`

This module handles data preprocessing:

- Loads datasets in **NILMTK format** (`.h5` / `.hdf5`)
- Aligns aggregate and appliance signals to a common sampling rate
- Extracts only overlapping time intervals
- Outputs a `.csv` file with the following columns:
  - `timestamp`
  - `aggregated_signal`
  - `machine1`
  - `machine2`

> **Note:** `machine1` and `machine2` are two instances of the same machine type, as the dataset includes duplicate machines per category.

### `main.py`

This module manages the modeling pipeline:

- Loads the preprocessed CSV data
- Applies **windowing**:
  - **Sliding windows** for the training set
  - **Non-overlapping windows** for the testing set (validation used as test in this example)
- Initializes the selected deep learning model
- Supports both **training** and **inference**, controlled via a configuration flag
- Implements several architectures inspired by residential NILM literature:
  - CNN
  - CRNN
  - TCN
  - WaveNet
  - BERT
  - LSTM
- Evaluation metrics:
  - **Mean Absolute Error (MAE)**
  - **Signal Aggregated Error**

## 📁 Dataset

This tool is designed to work with the following dataset:

**[Industrial Machines Dataset for Electrical Load Disaggregation](https://ieee-dataport.org/open-access/industrial-machines-dataset-electy:
   
Requirements.txt is the environment for processing the data, with the libraries required by NILMTK.
Architecture for WAVENET (https://github.com/picagrad/WaveNILM) and BERT (https://github.com/Yueeeeeeee/BERT4NILM) integrate original codes and they refer to https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682543 and https://dl.acm.org/doi/10.1145/3427771.3429390 respectively; TCN derives from https://ieeexplore.ieee.org/document/8911216, LSTM from https://arxiv.org/abs/1507.06594, CNN from https://dl.acm.org/doi/abs/10.5555/3504035.3504353 and CRNN from https://ieeexplore.ieee.org/document/9831435. 
