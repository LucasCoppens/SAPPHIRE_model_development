import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from Bio import SeqIO

from architectures import sensitivity, specificity, build_fully_connected_NN, build_CNN, build_LSTM, build_CNN_LSTM, build_LSTM_CNN_LSTM 

genbank_file = "data/NC_002516.gbk"
input_file_TSS = "data/NC_002516_TSS_coordinates.csv"
input_file_background = "data/NC_002516_background_coordinates.csv"


def read_genome():
    return str(next(SeqIO.parse(genbank_file, 'genbank')).seq)

def one_hot_encode_bases(seq):
    result = []
    for base in seq:
        if base == "A":
            result.append(np.array([1,0,0,0]))
        elif base == "C":
            result.append(np.array([0,1,0,0]))
        elif base == "G":
            result.append(np.array([0,0,1,0]))
        elif base == "T":
            result.append(np.array([0,0,0,1]))
        else:
            raise Exception('Input base sequence not only consistent of ACGT')

    return np.array(result)

def evaluate_model(model, x, y):
    evaluation = model.evaluate(x, y, verbose=0)
    for metric, score in zip(model.metrics_names, evaluation):
        if not metric == "loss":
            print(metric,": ", round(score,4)*100, "%")

def plot_metrics(sens, spec, val_sens, val_spec):
    plt.figure(figsize=(10,4))
    plt.title("Learning curves")
    plt.ylim(0.65,1)
    plt.plot(sens, label="Train sensitivity", marker="", color="#FF0000")
    plt.plot(spec, label="Train specificity", marker="", color="#FFFF00")
    plt.plot(val_sens, label="Validation sensitivity", marker="", color="#00FF00")
    plt.plot(val_spec, label="Validation specificity", marker="", color="#0000FF")
    plt.legend()
    plt.show()

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    revcomp = ''
    for nc in seq[::-1]:
        revcomp+=complement[nc]
    return revcomp

def get_proms_sequences(file_path):
    PRE_TSS = 45
    genome = read_genome()
    seqs = []
    for line in open(file_path, 'r'):
        if not len(line.split(",")[0]) == 0:
            loc = int(line.split(",")[0])
            strand = line.split(",")[1].replace("\n", "")
            if strand == "+":
                seqs.append(genome[loc - PRE_TSS:loc])
            elif strand == "-":
                seqs.append(reverse_complement(genome[loc:loc + PRE_TSS]))
    return seqs

def get_encoded_data():
    x_data=[]
    y_data=[]

    for seq in get_proms_sequences(input_file_TSS):
        x_data.append(one_hot_encode_bases(seq))
        y_data.append(1)

    nb_positive_examples = len(np.array(x_data))

    for seq in get_proms_sequences(input_file_background):
        x_data.append(one_hot_encode_bases(seq))
        y_data.append(0)

    nb_negative_examples = len(np.array(x_data)) - nb_positive_examples

    X = np.array(x_data)
    Y = np.array(y_data)

    class_weights = {1: nb_negative_examples / 1000, 0:nb_positive_examples / 1000}

    return X, Y, class_weights

def main():
    X, Y, class_weights=get_encoded_data()

    x_data, x_test, y_data, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 3)
    x_train, x_validate, y_train, y_validate = train_test_split(x_data, y_data, test_size = 0.1, random_state = 2)

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_validate = x_validate.astype(np.float32)
    y_validate = y_validate.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    model = build_CNN(len(x_data[0]))

    file_path_callback = "models/temp_callback.h5"
    checkpoint = ModelCheckpoint(file_path_callback, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    callbacks = [checkpoint]
    history = model.fit(x_train, y_train,
                        shuffle=True,
                        validation_data=(x_validate, y_validate),
                        # epochs=250,
                        epochs=5,
                        batch_size=2000,
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=True)

    model = load_model(file_path_callback,
                       custom_objects={"sensitivity": sensitivity, "specificity": specificity})
    model_location = "models/model_TSS.h5"
    model.save(model_location)
    print("\n--> Model saved to: ", model_location,"\n")

    plot_metrics(history.history["sensitivity"], history.history["specificity"], history.history["val_sensitivity"], history.history["val_specificity"])

    print("Trained model performance:")
    print("")
    print("Training set:")
    evaluate_model(model, x_train, y_train)

    print("")
    print("Validation set:")
    evaluate_model(model, x_validate, y_validate)

    print("")
    print("Test set:")
    evaluate_model(model, x_test, y_test)


main()
