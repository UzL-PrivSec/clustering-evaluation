import os
from string import ascii_uppercase
import urllib.request
import zipfile
import numpy as np
import pickle


def main():
    prepare_datasets()


def prepare_datasets():

    dataset_path = "./data/dts/"

    print("Preparing datasets")

    print("Dataset 'UCI Letters'")
    uci_letters_path = os.path.join(dataset_path, "uciLetters")
    if not os.path.exists(uci_letters_path):
        prepare_uci_letters(uci_letters_path)
    else:
        print("\tAlready exists!")

    print("Dataset 'UCI Gas Emissions'")
    uci_gas_emissions_path = os.path.join(dataset_path, "uciGasEmissions")
    if not os.path.exists(uci_gas_emissions_path):
        prepare_uci_gas_emissions(uci_gas_emissions_path)
    else:
        print("\tAlready exists!")


def prepare_uci_letters(path):

    os.makedirs(path, exist_ok=False)

    print("\tDownloading...")
    zip_path = os.path.join(path, "uci_letters.zip")
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
        zip_path,
    )

    print("\tExtracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path)

    print("\tPreprocessing...")
    mapping = {l: i for i, l in enumerate(ascii_uppercase)}

    with open(os.path.join(path, "letter-recognition.data"), "r") as file_:
        lines = file_.readlines()

    data = np.zeros((20000, 16))
    labels = np.zeros((20000))

    for i, line in enumerate(lines):

        label, *features, last_feature = line.split(",")

        data[i] = [int(f) for f in features] + [int(last_feature.strip())]
        labels[i] = mapping[label]

    with open(os.path.join(path, "uciLetters.pkl"), "wb") as file_:
        pickle.dump((data, labels), file_)

    print("\tCleaning up...")
    for filename in os.listdir(path):
        if filename != "uciLetters.pkl":
            os.remove(os.path.join(path, filename))


def prepare_uci_gas_emissions(path):

    os.makedirs(path, exist_ok=False)

    print("\tDownloading...")
    zip_path = os.path.join(path, "uci_letters.zip")
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip",
        zip_path,
    )

    print("\tExtracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path)

    print("\tPreprocessing...")
    lines = []
    for filename in os.listdir(path):
        if not filename.endswith(".csv"):
            continue
        with open(os.path.join(path, filename), "r") as _file:
            lines.extend(_file.readlines()[1:])  # ignore first line

    num_features = len(lines[0].split(","))

    data = np.zeros((len(lines), num_features))
    labels = np.zeros(len(lines))

    for i, line in enumerate(lines):
        *features, last_feature = line.split(",")
        data[i] = np.array(features + [last_feature.strip()]).astype(float)

    with open(os.path.join(path, "uciGasEmissions.pkl"), "wb") as file_:
        pickle.dump((data, labels), file_)

    print("\tCleaning up...")
    for filename in os.listdir(path):
        if filename != "uciGasEmissions.pkl":
            os.remove(os.path.join(path, filename))


if __name__ == "__main__":
    main()
