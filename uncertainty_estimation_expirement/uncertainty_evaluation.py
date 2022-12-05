import argparse
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader
import numpy as np
from MyUtils.MyData import CelegansData, HumanData
from MyUtils.helpers import get_device
from MyUtils.losses import relu_evidence
import torch.nn.functional as F


def acc_with_threshold(preds, labels, uncertainty, threshold):
    # Filter the data of an epoch according to threshold filter and return:
    # 1. the correct number of datapoints after filtering and 2. the number of filtered datapoints
    under_threshold_index = uncertainty < threshold
    preds_filter = preds[under_threshold_index]
    labels_filter = labels[under_threshold_index]
    filter_nums = len(labels_filter)
    match = torch.eq(preds_filter, labels_filter).float()
    acc_nums = torch.sum(match)
    # print('match={},acc={},'.format(match, acc))
    return acc_nums, filter_nums

def filter_with_threshold(preds, labels, uncertainty, threshold):
    # Filter the [preds] and [labels] arrays of a batch according to threshold,
    # and return the filtered array
    under_threshold_index = uncertainty < threshold
    preds_filter = preds[under_threshold_index]
    labels_filter = labels[under_threshold_index]
    return preds_filter, labels_filter

def uncertainty_threshold_testing(model, dataLoader, args):

    device = get_device()

    model = model.to(device)
    table = PrettyTable(['threshold', 'Number of filtered datapoints',  'Proportion','ACC'])

    # Verify the impact of different thresholds on the prediction performance of the model,
    # traverse the test data set once under each threshold and record the prediction results
    for threshold in np.arange(0.1, 1, 0.05):
        total_acc_nums = 0
        total_filter_nums = 0
        model.eval()  # Set model to evaluate mode

        epoch_labels = []
        probability_s = []
        epoch_pred_labels = []
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

                # In the current epoch, the array composed of the prediction uncertainty u of each data point (u=2/beta+alpha)
                u = (args.num_classes / torch.sum(beta_alpha, dim=1, keepdim=True)).view(-1)
                acc_nums, filter_nums = acc_with_threshold(preds, labels, u, threshold)
                # Calculate how many datapoints of the current batch are predicted correctly
                total_acc_nums += acc_nums
                total_filter_nums += filter_nums

                # Filter the predicted evidence array according to threshold,
                # and the real label array of its corresponding position
                beta_alpha_filter, labels_filter = filter_with_threshold(beta_alpha, labels, u, threshold)


                # Calculate the prediction probability p on each category according to the Beta distribution
                p = F.softmax(beta_alpha_filter, 1).cpu().data.numpy()
                labels_filter = labels_filter.cpu().data.numpy()

                # Splice the data filtered by each batch (for subsequent calculation of auc, prc, etc.)
                epoch_labels += list(map(lambda x: x, labels_filter))
                probability_s += list(map(lambda x: x[1], p))
                epoch_pred_labels += list(map(lambda x: np.argmax(x), p))

        filter_acc = total_acc_nums / total_filter_nums

        # Calculate the model prediction performance under the current threshold
        table.add_row([threshold,total_filter_nums,total_filter_nums / len(dataLoader.dataset),filter_acc.item() ])
        print(table)

    return True

def run_uncertainty_prediction_expirement():
    test_Human_data = HumanData(url='../data/HumanByStr', mode='test')
    test_Human_dataloader = DataLoader(test_Human_data, batch_size=12, shuffle=True)
    # #
    test_Celegans_data = CelegansData(url='../data/CelegansByStr', mode='test')
    test_Celegans_dataloader = DataLoader(test_Celegans_data, batch_size=12, shuffle=True)

    Human_model = torch.load('../results/Human_model.pth')
    Celegans_model = torch.load('../results/Celegans_model.pth')
    print("-"*20,"Uncertainty_Testing_On_Celegans_Dataset","-"*20)
    uncertainty_threshold_testing(Celegans_model,test_Celegans_dataloader,args)
    print("-"*20,"Uncertainty_Testing_On_Human_Dataset","-"*20)
    uncertainty_threshold_testing(Human_model,test_Human_dataloader,args)


if __name__ == '__main__':
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_classes", default=2, type=int, help="num_classes."
    )
    args = parser.parse_args()
    run_uncertainty_prediction_expirement()
