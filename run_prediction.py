import argparse

import pandas as pd
import torch
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import numpy as np
from MyUtils.MyData import CelegansData, HumanData
from MyUtils.helpers import get_device
from MyUtils.losses import relu_evidence
import torch.nn.functional as F
import matplotlib.pyplot as plt

def uncertainty_testing(model, dataLoader, output_file_path, args):

    device = get_device()

    model = model.to(device)

    epoch_labels = []
    probability_s = []
    epoch_pred_labels = []
    uncertainties = []

    for i, (batch_drug_indexVec, batch_protein_indexVec, labels) in enumerate(dataLoader):
        batch_drug_indexVec, batch_protein_indexVec, labels = batch_drug_indexVec.to(
            device), batch_protein_indexVec.to(device), labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(batch_drug_indexVec.long(), batch_protein_indexVec.long())

            _, preds = torch.max(outputs, 1)
            evidence = relu_evidence(outputs)
            beta_alpha = evidence + 1

            # Probability of the positive class
            p = F.softmax(beta_alpha, 1).cpu().data.numpy()

            # In the current epoch, the array composed of the prediction uncertainty u of each data point (u=2/beta+alpha)
            u = (args.num_classes / torch.sum(beta_alpha, dim=1, keepdim=True)).view(-1)

            # Splice the data filtered by each batch (for subsequent calculation of auc, prc, etc.)
            epoch_labels += list(map(lambda x: x.item(), labels))
            probability_s += list(map(lambda x: x[1].item(), p))
            epoch_pred_labels += list(map(lambda x: np.argmax(x).item(), p))
            uncertainties += list(map(lambda x: x.item(), u))

    # Calculate and print acc, auc, prc, etc.
    acc = np.mean(np.array(epoch_labels) == np.array(epoch_pred_labels))
    print('ACC:', acc)
    auc = roc_auc_score(epoch_labels, probability_s)
    print('AUC:', auc)
    prc = average_precision_score(epoch_labels, probability_s)
    print('PRC:', prc)

    # Create a DataFrame
    data = {'label': epoch_labels, 'probability': probability_s, 'pred_label': epoch_pred_labels, 'uncertainty': uncertainties}
    df = pd.DataFrame(data)

    # Save the epoch_labels, epoch_labels and epoch_pred_labels to .csv file
    df.to_csv(output_file_path, index=False)

    return True


def run_uncertainty_distibution_expirement(args):
    if args.dataset == "Human":
        test_data = HumanData(url='data/{}ByStr'.format(args.dataset), mode='test')
        model = torch.load('results/Human_model.pth')
    elif args.dataset == "Celegans":
        test_data = CelegansData(url='data/{}ByStr'.format(args.dataset), mode='test')
        model = torch.load('results/Celegans_model.pth')
    else:
        raise ValueError("Invalid dataset name")

    test_dataloader = DataLoader(test_data, batch_size=12, shuffle=True)

    print("-"*20,"Testing_On_{}_Dataset".format(args.dataset),"-"*20)
    uncertainty_testing(model, test_dataloader, "results/{}_output.csv".format(args.dataset), args)


if __name__ == '__main__':
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="Human", help="name of dataset", choices=['Human', 'Celegans'])
    parser.add_argument("--num_classes", default=2, type=int, help="num_classes.")

    args = parser.parse_args()

    run_uncertainty_distibution_expirement(args)
