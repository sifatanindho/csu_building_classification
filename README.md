# CSU Building Classification
#### CSU's 1st ML competition, CSU Building Classification, supported by CCVC (Club Computer Vision at CSU).

2nd place submission to the Club Computer Vision at CSU (CCVC) 1st ML Competition


## Competition overview:

### Overview
We collected N number of buildings in CSU for you! In this competition, you need to build a model can classify the building well. Competitors will build models that classify CSU buildings. Please check more details about the building image data, metric, submission, and rules in the description section.

### Description
Our club is aiming the activate computer vision community. For that, we are holding a computer vision and machine learning competition related to Colorado State University (CSU). All data is collected in CSU, the data can be used any future research within an approval of Computer Vision Lab in CSU. In addition, we propose to utilize the techniques from the competition to engineering applications after the competition.

If you are interesting on our club activities or data, please contact: computervisionclub.csu@gmail.com

### Evaluation
Submissions are evaluated on F1 - score between the predicted building class index and the target class index. The set of sample test images are provided, but the evaluation will be done in sample test images + hidden images.

### Submission File
For each ImageID in the test set, you must predict a building class index for the Label variable. For example, The file should contain a header and have the following format:

ImageID,Label
00000000,2
00011093,24
00001340,0
etc.

## My approach

I started by defining a custom dataset class, CSUBuildingDataset, to handle data loading and preprocessing for the CSU building classification task. This class allowed me to efficiently load and preprocess the image data, applying a series of transformations including resizing, random rotation, flipping, color jittering, and normalization.

Next, I defined a custom model class, BuildingClassifier, which inherited from pl.LightningModule. This model used a ResNet-18 backbone with a custom classification head, and I trained it using a custom training loop with PyTorch Lightning. I also defined a checkpointing callback to save the model with the lowest validation loss.

During training, I used a batch size of 8 and 16 workers for data loading. I also experimented with different hyperparameters, including the learning rate and number of epochs. While training, I monitored the model's performance on the validation set and adjusted the hyperparameters as needed. Ultimately, my model achieved a strong performance on the CSU building classification task.
