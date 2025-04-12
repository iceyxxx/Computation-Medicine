import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

diagnosis_type = {}
def sliceData(ecg_signal):
    window_size = 2000
    min_stride, max_stride = 1500, 2000  # Range of random window

    segments = []
    start_idx = np.random.randint(0, window_size)
    while start_idx + window_size < len(ecg_signal):
        segment = ecg_signal[start_idx : start_idx + window_size, :]
        segments.append(segment)

        stride = np.random.randint(min_stride, max_stride)
        start_idx += stride

    segments = np.stack(segments, axis = 0)

    return segments

def fetchECG(file_name):
# load ECG Data
    record = wfdb.rdrecord(file_name)
    label = 2

    with open(file_name + '.hea', 'r') as f:
        for line in f.readlines():
            if(line.startswith("# Reason for admission: ")):
                diag = line.strip().split("# Reason for admission: ")[1]
                if diag not in diagnosis_type.keys():
                    diagnosis_type[diag] = 1
                else:
                    diagnosis_type[diag] += 1
                if diag == 'Myocardial infarction':
                    label = 1
                elif diag == 'Healthy control':
                    label = 0

    ecg_signals = record.p_signal
    fs = record.fs
    # print(ecg_signals.shape)

    X_train = sliceData(ecg_signals)
    Y_train = np.repeat(label, X_train.shape[0])
    
    return X_train, Y_train

    


data_dir = 'D:\\OxfordHomework\\CM\\ECGData'
X_data = []
Y_data = []
for patient in os.listdir(data_dir):
    if patient[:7] != "patient":
        continue
    ecg_list = set()
    for ecg_name in os.listdir(data_dir + '\\' + patient):
        ecg_list.add(ecg_name.split('.')[0])
    for ecg_name in ecg_list:
        tmpX, tmpY = fetchECG(data_dir + '\\' + patient + '\\' + ecg_name)
        X_data.append(tmpX)
        Y_data.append(tmpY)

# print(diagnosis_type)

X_data = np.concatenate(X_data, axis = 0)
Y_data = np.concatenate(Y_data, axis = 0)

print(X_data.shape)
print(Y_data.shape)

save_dir = 'D:\\OxfordHomework\\CM\\'
np.save(save_dir + 'X_data_L.npy', X_data)
np.save(save_dir + 'Y_data_L.npy', Y_data)

