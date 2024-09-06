## RetinalNET: Deep learning paper replication.

This repository contains the implementation of a Retinal OCT analysis system using a pretrained ResNet50 model. The system is based on utilizing transfer learning with ResNet50 for efficient analysis of Retinal Optical Coherence Tomography (OCT) images.

https://github.com/user-attachments/assets/c9f7531a-875f-4199-b68c-09cbc0a3a8f4


### Important Notes
This project demonstrates a basic implementation of the OpticNet paper for educational purposes. It's recommended to train for more epochs (e.g., 50) for potentially better results. In my testing, 20 epochs took approximately 2 hours achieveing around 90% accuracy on both testing and training sets. (Trained on NVIDIA GTX 1650 GPU).Training for longer will require more computational resources and time.

This is a replication of the core functionalities described in the Optic-Net paper, but may not include all the optimizations or complexities of the original research.

## How to Use This Project

### Installation
Clone this repository:
```
https://github.com/ft-sreedeep/RetinalNET.git
```
### Install Requirements
```
pip install -r requirements.txt
```
### Download Dataset 

This project uses the Retinal OCT dataset. You can download it from [Kaggle](https://www.kaggle.com/paultimothymooney/kermany2018) or use your own dataset following a similar structure.

### Usage
Prepare your data:
- Organize your OCT images in a folder structure similar to the Kaggle dataset.
- Update the `data_dir` variable in the script to point to your dataset location.

### Train the model
```
python train.py
```
### Customization

- You can modify the `num_epochs`, `batch_size`, and other hyperparameters in the script to suit your needs.
- To use a different pretrained model, modify the model initialization in the script.

## Results
After training for 20 epochs, the model achieved approximately 95%+ accuracy on both the training and testing sets. Your results may vary depending on the specific dataset and training duration.

## Citations
Cheng, J., Zhu, L., Yang, Y., Luo, S., Huang, J., Wang, D., ... & Sun, D. (2018). Optic nerve head segmentation using deep learning in spectral-domain optical coherence tomography images. IEEE Transactions on Medical Imaging, 37(11), 2600-2608. [DOI: 10.1109/TMI.2018.2838682](https://arxiv.org/pdf/1910.05672)
