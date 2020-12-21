As the name suggests this is a face mask detection project for medical as well as non-medical masks.

Problem statement:- You may have seen this types of projects but they all can only detect medical masks or full black masks. In this project what i have done is instead of converting images to gray scale i have passed RGB images to train the model so that it would detect masks of different colours.

This repository contains python files and link to dataset. Dataset contains images of people with different types of masks and images of people without mask.

First download the dataset here:- https://www.kaggle.com/shrirajchauhan/face-mask-medical-and-nonmedical

Then run the "mask_nomask_training.ipynb" to create, train, and save the model as a ".h5" file.

Then run the "mask_no_mask_detection.py" to detect masks.
