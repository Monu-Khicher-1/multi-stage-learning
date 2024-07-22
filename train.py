import sys, os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from time import perf_counter
import pickle
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model.config import load_config
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE
from dataset.loader import load_data, load_checkpoint
import optparse

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# best model:
# 1. train on masked eye dataset for 20 epoch
# 2. train on simple data for 15 epoch
# 3. train on masked eye dataset for 15 epoch
# weight = 1.85 : 1
# Augmentation: loader_2





def load_pretrained(pretrained_model_filename):
    assert os.path.isfile(
        pretrained_model_filename
    ), "Saved model file does not exist. Exiting."

    model, optimizer, start_epoch, min_loss = load_checkpoint(
        model, optimizer, filename=pretrained_model_filename
    )

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return model, optimizer, start_epoch, min_loss


def load_checkpoint(model, filename=None):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def plot_metrics(train_acc, valid_acc, train_auc, val_auc, train_f1, val_f1, train_loss, val_loss):
    epochs = range(1, len(train_acc)+1 )
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, valid_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_auc, label='Training AUC')
    plt.plot(epochs, val_auc, label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1, label='Training F1 Score')
    plt.plot(epochs, val_f1, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()


    

    plt.tight_layout()
    plt.savefig("performance.png")
    plt.close()
        

    plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss.png") 


def save_roc_curve(all_targets, all_probs, filename='roc_curve.png'):
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


def train_model(
    dir_path, mod, num_epochs, pretrained_model_filename, test_model, batch_size
):
    print("Loading data...")
    dataloaders, dataset_sizes = load_data(dir_path, batch_size)
    print("Done.")

    if mod == "ed":
        from train.train_ed import train, valid
        model = GenConViTED(config)
    else:
        from train.train_vae import train, valid
        model = GenConViTVAE(config)

    if pretrained_model_filename:
        print("Loading checkpoints.. hii")
        model = load_checkpoint(model,pretrained_model_filename)

    optimizer = optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
        momentum = 0.9,
        weight_decay=float(config["weight_decay"]),
    )
    class_weights = torch.tensor([1.85, 1.0], device=device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    mse = nn.MSELoss()
    min_val_loss = int(config["min_val_loss"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    

    model.to(device)
    torch.manual_seed(1)
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    train_auc, train_f1, val_auc, val_f1 = [] , [] , [], []

    since = time.time()
    best_acc = 0.0

    for epoch in range(0, num_epochs):
        train_loss, train_acc, epoch_loss, train_f1, train_auc = train(
            model,
            device,
            dataloaders["train"],
            criterion,
            optimizer,
            epoch,
            train_loss,
            train_acc,
            mse,
            train_f1,
           train_auc
        )
        valid_loss, valid_acc, val_f1, val_auc = valid(
            model,
            device,
            dataloaders["validation"],
            criterion,
            epoch,
            valid_loss,
            valid_acc,
            mse,
            val_f1,
            val_auc
        )
        scheduler.step()

        if valid_acc[-1] > best_acc:
            best_acc = valid_acc[-1]
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "min_loss": epoch_loss,
            }
            weight = os.path.join("weight", f"best_model_{mod}.pth")
            torch.save(state, weight)
            print(f"Best model saved with Acc: {best_acc:.4f}")

    time_elapsed = time.time() - since

    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    ) 

    print("\nSaving model...\n")

    file_path = os.path.join(
        "weight", 
        f'genconvit_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}',
    )

    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_loss, train_acc, valid_loss, valid_acc], f)

    state = {
        "epoch": num_epochs + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "min_loss": epoch_loss,
    }

    weight = f"{file_path}.pth"
    torch.save(state, weight)

    print("Done.")

    plot_metrics(train_acc, valid_acc, train_auc, val_auc, train_f1, val_f1, train_loss, valid_loss)


    if test_model:
        test(model, dataloaders, dataset_sizes, mod, weight, valid_loss, valid_acc, val_f1, val_auc)

    # plot_metrics(train_acc, valid_acc, train_auc, val_auc, train_f1, val_f1)


def test(model, dataloaders, dataset_sizes, mod, weight, valid_loss, valid_acc, val_f1, val_auc):
    print("\nRunning test...\n")
    model.eval()
    checkpoint = torch.load(weight, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    _ = model.eval()

    all_targets = []
    all_preds = []
    all_probs = []

    correct=0

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if mod == "ed":
                output = model(inputs).to(device).float()
            else:
                output = model(inputs)[0].to(device).float()

            _, preds = torch.max(output, 1)
            correct+=torch.sum(preds==labels.data).item()
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[:, 1])  # Assuming binary classification

    test_f1 = f1_score(all_targets, all_preds, average='macro')
    test_auc = roc_auc_score(all_targets, all_probs)

    test_acc = correct / len(dataloaders["test"].dataset)

    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    print(f'Test ACC:{test_acc:.4f}')

    save_roc_curve(all_targets, all_probs)


def gen_parser():
    parser = optparse.OptionParser("Train GenConViT model.")
    parser.add_option(
        "-e",
        "--epoch",
        type=int,
        dest="epoch",
        help="Number of epochs used for training the GenConvNextViT model.",
    )
    parser.add_option("-v", "--version", dest="version", help="Version 0.1.")
    parser.add_option("-d", "--dir", dest="dir", help="Training data path.")
    parser.add_option(
        "-m",
        "--model",
        dest="model",
        help="model ed or model vae, model variant: genconvit (A) ed or genconvit (B) vae.",
    )
    parser.add_option(
        "-p",
        "--pretrained",
        dest="pretrained",
        help="Saved model file name. If you want to continue from the previous trained model.",
    )
    parser.add_option("-t", "--test", dest="test", help="run test on test dataset.")
    parser.add_option("-b", "--batch_size", dest="batch_size", help="batch size.")

    (options, _) = parser.parse_args()

    dir_path = options.dir
    epoch = options.epoch
    mod = "ed" if options.model == "ed" else "vae"
    test_model = "y" if options.test else None
    pretrained_model_filename = options.pretrained if options.pretrained else None
    batch_size = options.batch_size if options.batch_size else config["batch_size"]

    return dir_path, mod, epoch, pretrained_model_filename, test_model, int(batch_size)


def main():
    start_time = perf_counter()
    path, mod, epoch, pretrained_model_filename, test_model, batch_size = gen_parser()
    train_model(path, mod, epoch, pretrained_model_filename, test_model, batch_size)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
