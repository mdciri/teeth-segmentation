# Teeth Segmentation

Teeth segmentation using Pytorch and [Monai](https://monai.io/). The data used belong to the [Tufts Dental Database](http://tdd.ece.tufts.edu/).

# Data split
The Thfts dataset is composed by 1000 images in total which were randomly divided as 70-20-10% in training, validation, and test set. The split is done by:

    pyhton data_split.py

which save the dataset lists in the `data.json` file. It stores also the class names and the class weights for the segmentation.

# Model

Simple UNet network created using `monai.netowrks`.

# Training
    pthon train.py -md "<model_directory>" -d "mps" -g 0 -bs 16 -lr 1.e-4 -ne 100

This scripts runs the training scrits. It:

- saves the best model at the specified `"<model_directory>"`, 
- uses the device `"mps"` (but you can use `"cuda"` or `"cpu"`)
- GPU at position `0`,
- batch size `16`, 
- learning rate `1.e-4`,
- `100` number of maximum epochs.

# Evaluation


# GUI
