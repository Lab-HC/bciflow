from bciflow.datasets.BCICIV2b import bciciv2b

dataset = bciciv2b(subject=1,path='data/BCICIV2b/')

print("EEG signals shape:", dataset["X"].shape)
print("Labels:", dataset["y"])
print("Class dictionary:", dataset["y_dict"])
print("Events:", dataset["events"])
print("Channel names:", dataset["ch_names"])
print("Sampling frequency (Hz):", dataset["sfreq"])
print("Start time (s):", dataset["tmin"])

