import os
import shutil
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model.config import load_config  
from model.genconvit_ed import GenConViTED  
from model.genconvit_vae import GenConViTVAE  
from dataset.test_loader import load_data  
import shutil
import cv2
import numpy as np

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()  
        output[:, target_class].backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(2), input_tensor.size(3)))
        cam -= cam.min()
        cam /= cam.max()
        return cam


def apply_colormap_on_image(org_im, activation, colormap_name):
    colormap = plt.get_cmap(colormap_name)
    heatmap = colormap(activation)
    heatmap = np.delete(heatmap, 3, 2)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    overlayed_img = cv2.addWeighted(org_im, 0.5, heatmap, 0.5, 0)
    return overlayed_img


def load_checkpoint(model, filename=None):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def load_pretrained(pretrained_model_filename, model):
    assert os.path.isfile(pretrained_model_filename), "Saved model file does not exist. Exiting."
    model = load_checkpoint(model, filename=pretrained_model_filename)
    return model


def save_roc_curve(all_targets, all_probs, filename='roc_curve_ed.png'):
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


def test_model(model, dataloaders, mod, weight):
    print("\nRunning test...\n")
    model.eval()
    checkpoint = torch.load(weight, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    all_targets, all_preds, all_probs = [], [], []
    correct = 0

    misclassified_dir = "misclassified"
    classified_dir = "correct_classified"
    os.makedirs(misclassified_dir, exist_ok=True)
    misclassified_file = os.path.join(misclassified_dir, "misclassified.txt")

    # target_layer = model.backbone.stem[0] 
    target_layer = model.backbone.stages[3].downsample[1]
    # target_layer = model.backbone.stages[3].blocks[2].conv_dw
    # target_layer = model.backbone.stages[2].blocks[8].conv_dw
    grad_cam = GradCAM(model, target_layer)

    output_dir = "Heatmaps"

    with open(misclassified_file, "w") as mf:
        mf.write("Original Path,Current Path\n")

        for inputs, labels, paths in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True  # Ensure inputs require gradients
            output = model(inputs)[0] if mod == "vae" else model(inputs)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == labels.data).item()
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()[:, 1])

            # ==================================================================
            # Code for analysing results
            #===================================================================

            # for i in range(len(labels)):
            #     input_img = inputs[i].detach().cpu().numpy().transpose(1, 2, 0)
            #     cam = grad_cam.generate_cam(inputs[i].unsqueeze(0), target_class=labels[i].item())

            #     input_tensor = inputs[i].unsqueeze(0)

            #     input_image = input_tensor.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
            #     input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
            #     input_image = np.uint8(255 * input_image)
            #     heatmap_image = apply_colormap_on_image(input_image, cam, 'jet')

            #     if(preds[i]==1 and labels[i]==1):
            #         output_dir_1 = os.path.join(output_dir,"truePositive")
            #     elif(preds[i]==0 and labels[i]==0):
            #         output_dir_1 = os.path.join(output_dir,"trueNegative")
            #     elif(preds[i]==1 and labels[i]==0):
            #         output_dir_1 = os.path.join(output_dir,"falsePositive")
            #     elif(preds[i]==0 and labels[i]==1):
            #         output_dir_1 = os.path.join(output_dir,"falsenegative")

            #     output_path = os.path.join(output_dir_1, f"{os.path.basename(paths[i].split('.')[0])}_cam.jpg")
            #     os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok=True)
            #     cv2.imwrite(output_path, heatmap_image)

                # if preds[i] != labels[i]:
                #     original_path = paths[i]  
                #     if labels[i]:
                #         current_path = os.path.join(misclassified_dir, "Real")
                #     else:
                #         current_path = os.path.join(misclassified_dir, "Fake")

                #     os.makedirs(current_path, exist_ok=True)
                #     shutil.copy(original_path, current_path)
                #     mf.write(f"{original_path}\n")
                # else:
                #     original_path = paths[i]  
                #     if labels[i]:
                #         current_path = os.path.join(classified_dir, "real")
                #     else:
                #         current_path = os.path.join(classified_dir, "fake")

                #     os.makedirs(current_path, exist_ok=True)
                #     shutil.copy(original_path, current_path)
                #     mf.write(f"{original_path}\n")

    test_f1 = f1_score(all_targets, all_preds, average='macro')
    test_auc = roc_auc_score(all_targets, all_probs)
    test_acc = correct / len(dataloaders["test"].dataset)

    save_roc_curve(all_targets, all_probs)

    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    print(f'Test ACC: {test_acc:.4f}')


def main():
    import optparse
    parser = optparse.OptionParser("Test GenConViT model.")
    parser.add_option("-d", "--dir", dest="dir", help="Data path.")
    parser.add_option("-m", "--model", dest="model", help="Model variant: ed or vae.")
    parser.add_option("-w", "--weight", dest="weight", help="Saved model weight file path.")
    parser.add_option("-b", "--batch_size", dest="batch_size", help="Batch size.")

    options, _ = parser.parse_args()

    dir_path = options.dir
    mod = options.model
    weight = options.weight
    batch_size = int(options.batch_size) if options.batch_size else int(config["batch_size"])

    dataloaders, _ = load_data(dir_path, batch_size)

    if mod == "ed":
        model = GenConViTED(config)
    else:
        model = GenConViTVAE(config)

    model = load_pretrained(weight, model)
    model.to(device)

    test_model(model, dataloaders, mod, weight)


if __name__ == "__main__":
    main()
