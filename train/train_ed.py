import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    epoch,
    train_loss,
    train_acc,
    mse=None,
    train_f1=None,
    train_auc=None,
):
    model.train()

    curr_loss = 0
    t_pred = 0
    all_targets = []
    all_preds = []
    all_probs = []

    print("Training Size: ", len(train_loader))
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images).squeeze()
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        curr_loss += loss.sum().item()
        _, preds = torch.max(output, 1)
        t_pred += torch.sum(preds == targets.data).item()

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[:, 1])

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(images),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    epoch_loss = curr_loss / len(train_loader.dataset)
    epoch_acc = t_pred / len(train_loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    epoch_auc = roc_auc_score(all_targets, all_probs)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    train_f1.append(epoch_f1)
    train_auc.append(epoch_auc)

    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 Score: {:.4f}, AUC: {:.4f}\n".format(
            epoch_loss,
            t_pred,
            len(train_loader.dataset),
            100.0 * t_pred / len(train_loader.dataset),
            epoch_f1,
            epoch_auc,
        )
    )

    return train_loss, train_acc, epoch_loss, train_f1, train_auc


def valid(
    model, device, test_loader, criterion, epoch, valid_loss, valid_acc, mse=None, val_f1=None, val_auc=None
):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_preds = []
    all_probs = []

    print("Valid Size: ", len(test_loader))

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            output = model(images).squeeze()

            loss = criterion(output, targets)

            test_loss += loss.sum().item()

            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == targets.data).item()

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(output, dim=1).cpu().numpy()[:, 1])

            if batch_idx % 10 == 0:
                print(
                    "Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(images),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss.item(),
                    )
                )

    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_acc = correct / len(test_loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    epoch_auc = roc_auc_score(all_targets, all_probs)

    valid_loss.append(epoch_loss)
    valid_acc.append(epoch_acc)
    val_f1.append(epoch_f1)
    val_auc.append(epoch_auc)

    print(
        "Valid Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 Score: {:.4f}, AUC: {:.4f}\n".format(
            epoch_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            epoch_f1,
            epoch_auc,
        )
    )

    return valid_loss, valid_acc, val_f1, val_auc
