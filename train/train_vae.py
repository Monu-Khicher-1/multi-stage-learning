import torch
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    epoch,
    train_loss,
    train_acc,
    mse,
    train_f1,
    train_auc
):
    model.train()
    curr_loss = 0
    t_pred = 0
    all_targets = []
    all_preds = []
    all_probs = []

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output, recons = model(images)
        loss_m = criterion(output, targets)
        vae = mse(recons, images)
        loss = loss_m + vae  # +model.encoder.kl

        # print("Out shape: ",output)

        loss.backward()
        optimizer.step()

        curr_loss += loss.sum().item()
        _, preds = torch.max(output, 1)
        t_pred += torch.sum(preds == targets.data).item()

        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[:, 1])  # Assuming binary classification

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} vae_Loss {:.6f}".format(
                    epoch,
                    batch_idx * len(images),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss_m.item(),
                    vae.item(),
                )
            )

            # train_loss.append(loss.sum().item() / len(images))
            # train_acc.append(preds.sum().item() / len(images))
    
    epoch_loss = curr_loss / len(train_loader.dataset)
    epoch_acc = t_pred / len(train_loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    epoch_auc = roc_auc_score(all_targets, all_probs)  # Assuming binary classification

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


def valid(model, device, test_loader, criterion, epoch, valid_loss, valid_acc, mse, valid_f1, valid_auc):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            output, recons = model(images)
            loss_m = criterion(output, targets)
            vae = mse(recons, images)
            loss = loss_m + vae  # +model.encoder.kl

            test_loss += loss.sum().item()  # sum up batch loss

            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == targets.data)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[:, 1])  # Assuming binary classification

            if batch_idx % 10 == 0:
                print(
                    "Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} vae_Loss {:.6f}".format(
                        epoch,
                        batch_idx * len(images),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss_m.item(),
                        vae.item(),
                    )
                )

                # valid_loss.append(loss.sum().item() / len(images))
                # valid_acc.append(preds.sum().item() / len(images))

    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_acc = correct / len(test_loader.dataset)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    epoch_auc = roc_auc_score(all_targets, all_probs)  # Assuming binary classification

    valid_loss.append(epoch_loss)
    valid_acc.append(epoch_acc.item())
    valid_f1.append(epoch_f1)
    valid_auc.append(epoch_auc)

    print(
        "\nValid Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 Score: {:.4f}, AUC: {:.4f}\n".format(
            epoch_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            epoch_f1,
            epoch_auc,
        )
    )

    return valid_loss, valid_acc, valid_f1, valid_auc
