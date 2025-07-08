import argparse
from matplotlib import pyplot as plt
from nilmtk_master.nilmtk.dataset import DataSet
import numpy as np
import pandas as pd
import os

def process_data(IMDELD_path, build, period, power_type, channels, save_dir, device):
    refit = DataSet(IMDELD_path)
    elec = refit.buildings[build].elec
    good = elec.meters[0].good_sections(full_results=False)

    def load_signal(channel, power_type, period):
        power_ = next(elec.meters[channel].load(sample_period=period))
        power_signal = power_['power'][power_type]
        return power_signal

    new = pd.DataFrame([])
    app = pd.DataFrame([])
    app2 = pd.DataFrame([])

    for i in range(len(good)):
        refit.set_window(start=good[i].start.asm8, end=good[i].end.asm8)
        elec = refit.buildings[build].elec

        try:
            mains_ACTIVE = load_signal(channel=0, power_type=power_type, period=period).fillna(method='bfill')
            mains_ACTIVE[mains_ACTIVE < 0] = 0
        except KeyError:
            continue

        try:
            fan1 = load_signal(channel=channels[0], power_type=power_type, period=period).fillna(0)
            fan1[fan1 < 0] = 0
        except KeyError:
            continue
        try:
            fan2 = load_signal(channel=channels[1], power_type=power_type, period=period).fillna(0)
            fan2[fan2 < 0] = 0
        except KeyError:
            continue

        if len(mains_ACTIVE) > len(fan1):
            mains_ACTIVE = mains_ACTIVE[fan1.index[0]:fan1.index[-1]]

        plt.plot(mains_ACTIVE)
        plt.plot(fan1)
        plt.plot(fan2)
        plt.show()

        new = pd.concat([new, mains_ACTIVE], axis=0)
        app = pd.concat([app, fan1], axis=0)
        app2 = pd.concat([app2, fan2], axis=0)

    final = pd.concat([new,app,app2],axis=1).fillna(method='ffill')
    final.to_csv(save_dir + device + '.csv', sep=';')



IMDELD_path = "IMDELD.hdf5"
build = 1
period = 10
power_type = 'active'
save_dir = 'processed_data/'
device = 'milling'

os.mkdir(save_dir)

if device == 'fan':
  channels = [6,7]
if device == 'pelletizer':
    channels = [2,3]
if device == 'double':
    channels = [4,5]
if device == 'milling':
    channels = [9,10]
process_data(IMDELD_path, build, period, power_type, channels, save_dir, device)


