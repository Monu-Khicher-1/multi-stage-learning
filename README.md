# Multi Stage Learning for Deepfake Detection

## Introduction

## Weights

[link](https://drive.google.com/drive/folders/1F6YgS3ayNfuHjjOwNMvWFpjY6IONsuhy?usp=drive_link)

## Setup

- Prepare dataset: Split data in train, validation, and testing. These fake and real folders contain fake and real images, respectively.

The directory structure is as follows:

```
   Data --
         |
         |---train--
         |         |---real --|
         |         |          |-- image1.png
         |         |          |-- image2.png
         |         |                :
         |         |                :
         |         |          |-- imagen.png
         |         |
         |         |---fake --|
         |                    |-- image1.png
         |                    |-- image2.png
         |                          :
         |                          :
         |                    |-- imagem.png
         |
         |
         |---valid
         |
         |---test
```

- Generate cropped faces dataset: Now, we have to generate cropped faces dataset which can be genrated by following command
  `$ python3 process_data.py --input_folder Data --output_folder cropped_data`

- Generate Masked Eye Dataset:
  `$ python3 masking.py --input_folder cropped_data --output_folder Masked_data`
  `$ cp -r cropped_data/validation Masked_data `
  `$ cp -r cropped_data/test Masked_data `

- Run the training script:

```bash
python train.py
    -d <training-data-path>
    -m <model-variant>
    -e <num-epochs>
    -p <pretrained-model-file>
    -b <batch-size>
    -t
```

`<training-data-path>`: Path to the training data.<br/>
`<model-variant>`: Specify the model variant (`ed` for Autoencoder or `vae` for Variational Autoencoder).<br/>
`<num-epochs>`: Number of epochs for training.<br/>
`<pretrained-model-file>` (optional): Specify the filename of a pre-trained model to continue training.<br/>
`-b` (optional): Batch size for training. The default is 32.<br/>
`-t` (optional): Run the test on the test dataset after training.

The model weights and metrics are saved in the `weight` folder.

- Multi-stage training:

```bash
python --d Masked_data --m ed --e 20 -b 32 -t y
```

```bash
python --d simple_data --m ed --e 15 -b 32 -t y -p weight/best_model_ed.pth
```

```bash
python --d Masked_data --m ed --e 15 -b 32 -t y -p weight/best_model_ed.pth
```

Testing and Heatmaps :

```bash
python test.py --dir Processed_Data --model ed --weight weight/best_model_ed.pth --batch_size 32
```
