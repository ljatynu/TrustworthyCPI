import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from MyUtils.Model import TCPI_Model
from MyUtils.MyData import HumanData,CelegansData
from MyUtils.helpers import get_device, one_hot_embedding
from MyUtils.losses import beta_loss

if __name__ == '__main__':
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument(
        "--num_classes", default=2, type=int, help="num_classes."
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Desired number of epochs."
    )
    args = parser.parse_args()
    num_epochs = args.epochs

    criterion = beta_loss


    model = TCPI_Model()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = get_device()

    model = model.to(device)

    best_acc = 0.0

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}


    train_data = HumanData(url='data/HumanByStr', mode='train')
    train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True)

    test_data = HumanData(url='data/HumanByStr', mode='test')
    test_dataloader = DataLoader(test_data, batch_size=12, shuffle=True)

    # train_data = CelegansData(url='data/CelegansByStr', mode='1to1_train')
    # train_dataloader = DataLoader(train_data, batch_size=12, shuffle=True)
    #
    # test_data = CelegansData(url='data/CelegansByStr', mode='test')
    # test_dataloader = DataLoader(test_data, batch_size=12, shuffle=True)

    dataLoaders = {
        "train": train_dataloader,
        "val": test_dataloader,
    }

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch, args.epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0

            for i, (batch_drug_indexVec, batch_protein_indexVec, labels) in enumerate(dataLoaders[phase]):
                if i % 100 == 0:
                    print("[epoch:{}]  [batch:{}/{}]".format(epoch, i, len(dataLoaders[phase])))
                batch_drug_indexVec, batch_protein_indexVec, labels = batch_drug_indexVec.to(
                    device), batch_protein_indexVec.to(device), labels.to(device)

                # zero the parameter gradients++
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    y = one_hot_embedding(labels.long(), args.num_classes)
                    y = y.to(device)
                    outputs = model(batch_drug_indexVec.long(), batch_protein_indexVec.long())
                    _, preds = torch.max(outputs, 1)
                    # Beta loss with the KL regularized term
                    loss = criterion(
                        outputs, y.float(), epoch+1, args.num_classes, 3, device
                    )

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    # statistics
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataLoaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataLoaders[phase].dataset)
            print(
                "{} loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_loss, epoch_acc
                )
            )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, 'results/Human_model.pth')
