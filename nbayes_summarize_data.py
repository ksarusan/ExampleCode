import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter


def summarize_probabilities(train_data):
    train_labels = train_data.iloc[:, -1]
    labels = list(Counter(train_labels))
    p_labels = [len(train_labels[train_labels == label]) / len(train_labels) for label in labels] #priors for each label
    train_data_2 = train_data[train_data["label"] == 2].iloc[:, :-1]
    train_data_4 = train_data[train_data["label"] == 4].iloc[:, :-1]
    summary_2 = pd.DataFrame(index=range(1,11), columns=train_data_2.columns)
    summary_4 = pd.DataFrame(index=range(1,11), columns=train_data_4.columns)

    # this for loop is using each feature independently and can deal with missing values
    for idx in range(1,11):
        summary_2.loc[idx,"clump"] = len(train_data_2[train_data_2["clump"] == idx])
        summary_2.loc[idx, "uniformity"] = len(train_data_2[train_data_2["uniformity"] == idx])
        summary_2.loc[idx, "marginal"] = len(train_data_2[train_data_2["marginal"] == idx])
        summary_2.loc[idx, "mitoses"] = len(train_data_2[train_data_2["mitoses"] == idx])

        summary_4.loc[idx, "clump"] = len(train_data_4[train_data_4["clump"] == idx])
        summary_4.loc[idx, "uniformity"] = len(train_data_4[train_data_4["uniformity"] == idx])
        summary_4.loc[idx, "marginal"] = len(train_data_4[train_data_4["marginal"] == idx])
        summary_4.loc[idx, "mitoses"] = len(train_data_4[train_data_4["mitoses"] == idx])

    # normalize to get the likelihoods
    summary_2 = summary_2 / summary_2.sum()
    summary_4 = summary_4 / summary_4.sum()

    return labels, p_labels,summary_2,summary_4

def write_summary_to_file(file_name, summary):
    try:
        f_out = open(file_name, 'w')
    except:
        print(f"Output file {file_name} cannot be created")
        sys.exit(1)

    f_out.write("Value\tclump\tuniformity\tmarginal\tmitoses\n")

    for i, row in summary.iterrows():
        f_out.write(f"{i}\t{row[0]:.3f}\t{row[1]:.3f}\t{row[2]:.3f}\t{row[3]:.3f}\n")

    f_out.close()

def classify_point(x,labels, prior,likelihood_2, likelihood_4):
    # x can only have length 4, as there are 4 features
    posterior = np.zeros((2,))
    posterior[0] = prior[0] * likelihood_2.loc[x[0],"clump"] * likelihood_2.loc[x[1],"uniformity"] * likelihood_2.loc[x[2],"marginal"] * likelihood_2.loc[x[3],"mitoses"]
    posterior[1] = prior[1] * likelihood_4.loc[x[0], "clump"] * likelihood_4.loc[x[1], "uniformity"] * likelihood_4.loc[x[2], "marginal"] * likelihood_4.loc[x[3], "mitoses"]
    return labels[np.argmax(posterior)],posterior


if __name__ == "__main__":
    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Naive Bayes probability summary for different values."
    )

    parser.add_argument(
        "--traindir",
        required=True,
        help="Path to train directory containing file tumor_info.txt"
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where output_summary_class_2.txt & output_summary_class_4.txt will be created"
    )

    args = parser.parse_args()

    # Set the paths
    train_dir = args.traindir
    out_dir = args.outdir

    os.makedirs(args.outdir, exist_ok=True)

    # Read the files
    train_data = pd.read_csv(f"{train_dir}/tumor_info.txt", sep="\t", header=None)
    train_data.columns = ["clump", "uniformity","marginal","mitoses","label"]

    #summarize the data and store them in the respective files
    labels, p_labels, summary_2, summary_4 = summarize_probabilities(train_data)

    file_name_1 = f"{out_dir}/output_summary_class_2.txt"
    file_name_2 = f"{out_dir}/output_summary_class_4.txt"

    write_summary_to_file(file_name_1, summary_2)
    write_summary_to_file(file_name_2, summary_4)


    # find the class for point [ clump = 5, uniformity = 2, marginal = 3, mitoses = 1 ]
    test_point = np.array([5,2,3,1])
    likelihood_2 = pd.read_csv(f"{out_dir}/output_summary_class_2.txt",sep = "\t", index_col=0)
    likelihood_4 = pd.read_csv(f"{out_dir}/output_summary_class_4.txt",sep = "\t", index_col=0)
    pred_label,posterior = classify_point(test_point,labels,p_labels,likelihood_2,likelihood_4)

    print(f"The \"posteriors\" for the classes {labels} are {posterior}")
    print(f"The predicted class for the test point [ clump = 5, uniformity = 2, marginal = 3, mitoses = 1 ] is:\n{pred_label}")