# Automatic-Colouriation-of-Wildlife-GrayScale-Images

## Abstract
This Repository is a part of our submission for the Project of the Image processing and Computer Vision Course. 
The main objective of this project is to develop automated colourising tools to colourise black and white or Grayscale images of Wildlife. This project derives its importance from the fact that a large number of cameras originally used in wildlife surveys produced grayscale images. Colourisation of images is of utmost importance in this domain since colour could be a differentiating factor between structurally similar animals but of different species such as Snakes, Frogs, Birds and so on. Additionally, in places with an insufficient light source, the colour images obtained may not be able to satisfactorily provide results to study the wildlife. In such a case, these models could use previously obtained images as examples and develop a finer version of the new image. To colourise images, we have chosen to preprocess and train our models in the CIE-LAB space. The models developed in the course of this project have been detailed below. 

As a part of this project the following models were built to analyse and predict Life expectancy: 
1. XGBoost Regressor 
2. Convolutional Neural Networks 
3. Auto Encoders

## Preprocessing 
The dataset for this project is the Animals-151 dataset. Due to storage and compute constraints, we've chosen to work with a subset of 60 different animals stored in each directory with anywhere between 30 to 60 images per animal. In the initial stage of preprocessing, we collected the dataset and modified the directory structure to ensure uniform indexing of the training and testing images across models. Following this, we loaded our exemplar images in the CIE-LAB space for the easier development of models. The grayscale testing images were obtained by using the rgb2gray function in the OpenCV package. Once the LAB components of the coloured exemplar images were obtained, the colours were quantised using KMeans. Each pixel in an image in the dataset can take one of 11000 different colours. However, we cannot effectively model such a large colour space so we restrict each exemplar image to a colour space of 256 for the sake of this project. 

## Loading Data 
Due to the size of the dataset for this project and the development of models on google colaboratory, we accessed our data using a Google Drive folder. To replicate the results of this project, we request you to mount your drive while running the model and unzip the dataset in the main Drive folder. 

### The zipped version of the dataset for the project can be found under the data folder

## Modelling 
All three models were developed with the agenda of predicting the colour space of the inputted grey scale image. Below, we detail the approaches and results for the models based on Histogram Comparison for three global test images. We obtain the Chi-Square value and the Level of intersection between the histogram of the original image and the predicted image. 

### Each model is segregated into its folder under the models' folder. The models are developed in notebooks and the data will have to be manually loaded for the same during the execution.  

### XGBoost Regressor 
The Gradient Boosting model was fed the AB space of the coloured exemplar image in the form of colour maps along with the features extracted using HOG and DAISY feature extraction. In the below table, we detail the Chi-Square value and Intersection value of the histograms obtained from the original image and predicted image. 

 |Image                      | Chi-Square | Intersection |
 |---------------------------| -----------| -------------|
 | Ailuropoda Melanoleuca    | 0.88       | 3.40         |
 | Ceratotherium Simum       | 0.85       | 3.69         |
 | Dasypus Novemcinctus      | 0.93       | 2.65         |
 
### Convolutional Neural Network 
For the convolutional neural network model, we pass the LAB images as input. We did not have to perform explicit feature extraction since CNNs are equipped to perform feature engineering on their own without human intervention. For the sake of this project, we use the resnet18 architecture with RELU layers and Batch normalisation to handle the gradients problem. 

 |Image                      | Chi-Square | Intersection |
 |---------------------------| -----------| -------------|
 | Ailuropoda Melanoleuca    | 0.88       | 3.40         |
 | Ceratotherium Simum       | 0.85       | 3.69         |
 | Dasypus Novemcinctus      | 0.93       | 2.65         |


### Auto Encoders
For the AutoEncoder model, we use the lab functions again without feature engineering for the same reason as the CNN model. The Encoder in the autoencoder is built out of a sequential model with about 10 convolutional layers with the relu activation function while the decoder is made out of 10 convolutional layers with upsampling layers in the middle. 

 |Image                      | Chi-Square | Intersection |
 |---------------------------| -----------| -------------|
 | Ailuropoda Melanoleuca    | 2.04       | 1.81         |
 | Ceratotherium Simum       | 0.6        | 2.64         |
 | Dasypus Novemcinctus      | 2.4        | 0.38         |

## Conclusion 
Judging the benefits of our method our existing solutions is a bit troublesome due to two reasons. First, most auto-colourisation methods are judged by a group of participants and arent assessed metrically. Second, from our extensive literature survey, we haven't found a precedent for the usage of auto-colourisation in the field of wildlife images. However, visually we see that the CNN and AutoEncoder models generate more visually similar images. 
