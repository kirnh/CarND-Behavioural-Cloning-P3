#**Behavioral Cloning** 

---

The goals / steps of this project are the following:

  - Use the simulator to collect data of good driving behavior
  - Build, a convolution neural network in Keras that predicts steering angles from images
  - Train and validate the model with a training and validation set
  - Test that the model successfully drives around track one without leaving the road
  - Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

  * model.py containing the script to create and train the model
  * drive.py for driving the car in autonomous mode
  * model.h5 containing a trained convolution neural network 
  * writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model resembles the architecture used by NVIDIA in its end-to-end learning for self driving car paper.

It has 5 convolutional layers and 3 fully connected layers in the following sequence as seen in my model.py file from line 130 to line 139.

  - Convolutional layer with depth 24 and filter of size 5x5
  - Convolutional layer with depth 36 and filter of size 5x5
  - Convolutional layer with depth 48 and filter of size 5x5
  - Convolutional layer with depth 64 and filter of size 3x3
  - Convolutional layer with depth 64 and filter of size 3x3
  - Flatten layer
  - Fully connected layer of width 100
  - Fully connected layer of width 50
  - Fully connected layer of width 10
  - Fully connected layer of width 1

Also, before introducing the image to the first layer, it is passed through an augmentation stage where it is cropped, resized and normalized after applying random flipping and random brightness augmentation (model.py line 45). 

The input image is of shape 64x64x3.

####2. Attempts to reduce overfitting in the model

The number of training epochs is selected appropriately to ensure that there is no overfitting (early cutoff method of preventing overfitting).  

Also, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 143-145). And the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 141).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the sample data provided by Udacity initially before adding a lap of center lane driving in the reverse direction and some more data of curve driving.  

Further details about how the training data was created are discussed below in later sections.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with well known architectures and tune them if necessary untill satisfactory learning is achieved. 

My first step was to use a convolution neural network model similar to NVIDIA's model as discussed in its End to End learning for SDC paper. I thought this model might be appropriate because of the similarity in the use case which is to enable self driving capabilities using only neural networks and image data from the car's cameras. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on both the training set and the validation set. However, when tested on the track the model seemed to predict a constant low steering angle.

This implied that the model was stuck at a local minima. This took me a while however to figure out. I realized that the reason for this was in the distribution of data across steering angles. Zero steering angle data was significantly more in number that the rest of the angle values. The problem I figured was not in the model architecture but with how my data was distributed across the target variable (steering angle). Also, since my model was giving low mean squared errors on both the training set and the validation set, I decided to stay with this same model and work around with data augmentation to better the performance. 

To overcome the local minima, I had to train my model on data where the angles were more evenly distributed. To do this, I included all three camera images and introduced a correction coefficient that corrected the steering angle value before randomly selecting between either 'center', 'left' or 'right' camera image each time during data generation (model.py line 49). This reduced the probability of getting absurdly more zero steering angle data while training due to which the model was getting stuck in a local minima during training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Since the loss was still reducing, I increased the number of training epochs and decided with an optimum value after trying a few values.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 130-139) is the same convolutional network that was selected initially from NVIDIA's paper and its architecture has already been discussed in the above sections. 

####3. Creation of the Training Set & Training Process

I first tried to train the model using only Udacity's sample data set but ended up adding some of my own data to increase the performance.

To capture good driving behavior, I first recorded one lap on track one using center lane driving in the reverse direction and some more data of curve road driving which I believed complimented the existing data. 

From this I ended up with 13773 datapoints where each datapoint had a steering angle corresponding to 'center' camera image path. It also had a corresponding 'left' camera image path and a 'right' camera image path.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

To generate batches of data on the fly, I used a python generator the yielded a batch of 32 images each time it was called. Each time, the generator sliced a 32 sized chunk of datapoints from the dataset and returned 32 images and the corresponding steering angle measurements (model.py line 90). To counteract the fact that the dataset had a zero angle bias, I randomly selected between 'center', 'left' or 'right' camera images each time which reduced the probability of getting a low angle or zero angle data. I made use of 'left' and 'right' camera images with a correction coefficient that simulated recovery data perfectly. (After a few experiments, I decided that a correction value of 0.2 was doing a good job). Also, to increase the robustness of my model, I randomly flipped images horizontally and added random brightness augmentation. 

The generator was used to generate training data and validation data on the fly from the validation set and training set obtained by splitting the initial 13773 datapoints containing image paths and center camera image steering angle. Also due to the set-up, I was sure that every batch was gonna be randomly distributed which would benefit learning.

I used the training data generator to train my model while the validation data generator helped me understand if there was any overfitting or underfitting. The ideal number of epochs was 7 as evidenced by the converging mean squared errors. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Credits
[This](https://medium.com/@subodh.malgonde/teaching-a-car-to-mimic-your-driving-behaviour-c1f0ae543686) blog post helped me streamline my approach at balancing the data by introducing random selection.
