# **Behavioral Cloning** 

---
[image0]: ./documentation/track.gif "testing"
**Behavioral Cloning Project**  

#####Author : Aparajith Sridharan
#####date : 23-03-2021  

![text][image0]

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image1]: ./documentation/PreProcessing.jpg "Pre Process"
[image2]: ./documentation/Model.jpg "Model"
[image3]: ./documentation/ROI.JPG "ROI"
[image4]: ./documentation/20kimgs.jpg "Traindata"
[image5]: ./documentation/noaugmentation_randomizestraight.jpg "randomize"
[image6]: ./documentation/augment.jpg "AugmentedData"
[image7]: ./documentation/straight.gif "Straight"
[image8]: ./documentation/recover.gif "recover1"
[image9]: ./documentation/recovery.gif "recover2"
[image10]: ./documentation/recovery3.gif "recover2"
[image11]: ./documentation/Ep_1_5.jpg "ep1-5"
[image12]: ./documentation/Ep_5_10.jpg "ep5-10"
[video0]: ./track_1.mp4 "trk1"
[video1]: ./ScreenRecording_track1.mp4 "scrtrk1"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `transforms.py` contains the necessary image transformation methods such as affine and rotation of images
* `preprocessing.py` contains the requisite image preprocessing methods
* `readData.py` contains the functions that help to organize the training data
* `utils.py` contains the utility functions to display some pictures from the training set
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results
* `track_1.mp4` recording of the output on track 1 

#### 2. Submission includes functional code

##### Simulator and Unity3D version 
The old simulator no longer worked on my machine due to the obsolete version of Unity used to create them. 
Hence, through forking the repo from Udacity, I ported the Unity3D project manually to work on Windows x64 and Linux x64 machines. 
Kindly refer to my other repository and branch : https://github.com/Aparajith-S/self-driving-car-sim/tree/port_to_unity_2020_3  
The project is ported to Unity3D 2020.3.0f1. 

using the provided drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Dependencies
Before discussing the architecture, the model was trained on a NVIDIA GTX-1650Ti mobile gpu and hence, the following versions of the libraries and cuDNN libraries were used. 

#### Python Dependencies
| Library/Tool   | version used |  
|:---:|:---:|
| python         |3.6.13|
|tensorboard           |    2.4.1|
|tensorboard-plugin-wit|    1.8.0|
|tensorflow-estimator  |    2.4.0|
|tensorflow-gpu        |    2.4.1|
|Keras |2.4.3|
|eventlet       |0.23.0|
|ffmpeg         |   1.4|   
|flask          | 1.1.2| 
|flask-socketio | 3.0.1| 
|imageio        | 2.9.0|
|matplotlib     |3.3.4|
|moviepy        |1.0.3|
|python-socketio|3.0.0|

#### NVIDIA Dependencies
The aforementioned Keras and Tensorflow-gpu uses the cuda packages:  

|NVIDIA packages| version| download link| 
|:-----:|:-----:|:---:|
|CUDA Development Toolkit | v11.0 | https://developer.nvidia.com/cuda-11.0-download-archive |
|cuDNN libs, headers and binaries| v8.1.1  |https://developer.nvidia.com/rdp/cudnn-download#a-collapse811-111|

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

##### Initial Architecture:

The Initial idea was to use Transfer learning using VGG-13 or 16 or a pre-trained Lenet model that I had handy from a previous project. 
However, this idea was dropped as I wanted to train a network from scratch as an exercise.

The model includes RELU layers to introduce non-linearity (code line 34 in `model.py`) , and the data is normalized in the model using the Preprocess function before feeding to Keras model. see `Preprocess(...)` in `preprocessing.py` . 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also augmented and introduced gaussian noise to straight steer values i.e. steer angle = 0 degrees.   

For details about how the training data was obtained, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model. I first tried the already pretrained lenet model that I used for a previous project to test if the training pipeline 
and the `drive.py` prediction pipeline were functional. 

My first step was to use the aforementioned convolution neural network model. I thought this model might be appropriate because of the similar features in terms of curves(edges) that the lenet took from traffic signs. 
Later, I referred to a paper which was published by NVIDIA on using an [empirical 5-layer CNN](https://arxiv.org/pdf/1604.07316v1.pdf) with a similar setup.    

##### Batch normalization and Dropouts
I added Batch normalization layers for each layer. In order to gauge how well the model was working, I did not use Dropouts immediately. After noticing that there was a huge difference between training and validation losses, 
I decided to introduce dropouts only for the dense layers.      

##### Training and Validation sets  
25 percent of the training set was used as the validation set. Additionally, shuffle was set to true to shuffle the training set and validation set.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road with very good handling characteristics.

#### 2. Final Model Architecture

The final model architecture (`model.py` function `networkModel(...)`) consisted of a convolution neural network with the following :

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 240x70x3 YUV colorspace, normalized image   	| 
| Convolution 5x5     	| 2x2 stride, 24 filters, valid padding, outputs 118x33x24 	|
| RELU					|-												|
| BatchNormalization |-|
| Convolution 5x5	    | 2x2 stride, 36 filters, valid padding, outputs 60x16x36   |
| RELU					|-												|
| BatchNormalization |-|
| Convolution 3x3     	| 1x1 stride, 48 filters, valid padding, outputs 28x6x48 	|
| RELU					|-												|
| BatchNormalization |-|
| Convolution 3x3	    | 1x1 stride, 64 filters, valid padding, outputs 26x4x64   |
| RELU					|-												|
| BatchNormalization |-|
| Convolution 3x3	    | 1x1 stride, 64 filters, valid padding, outputs 24x2x64   |
| RELU					|-												|
| BatchNormalization |-|
| Fully connected		| sizes - input: 1152, output: 100              |
| RELU		            | -             								|
| Dropout               |   50% dropout while training. 0% otherwise    |
| BatchNormalization |-|
| Fully connected		| sizes - input: 100, output: 50                |
| RELU		            |  -            								|
| Dropout               |   50% dropout while training. 0% otherwise    |
| BatchNormalization    |   -                   |
| Fully connected		| sizes - input: 50, output: 10                 |
| RELU		            |   -           								|
| Dropout               |   20% dropout while training. 0% otherwise    |
| BatchNormalization    |   -                   |
| Fully connected		| sizes - input: 10, output: 1                  |


Here is a visualization of the architecture

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image7]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center of the road in the event of an unexpected drive close to the curb.  
These images show what a recovery looks like starting from either sides of the road involving heavy swerving to prevent a run-off. 

![alt text][image8]  
![alt text][image9]  

Then I repeated this process on track two in order to get more data points.  
![alt text][image10]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image13]
![alt text][image14]

After the collection process, I had 37566 number of data points. The following describes the challenges faced and the methods employed to overcome the challenges.

##### Preprocessing
The image is converted into YUV color space as Both HLS, YUV were tried and YUV yielded better outcomes.
The steps employed are as follows: 
- Convert RGB ---> YUV
- Crop a ROI
- Do a ordinary Histogram equalization on Y channel (CLAHE was tried but it yielded poor results)
- Normalize the image
  The following shows the image and it's ROI superimposed.
  ![alt text][image3]
- Augment Images.
    - Flipping :
    Flipping of images was not done in post processing. 
    Instead, while recording, A U-turn was done and the trajectory was recorded again. 
    This is much better as the data is original and not post processed.
    - Rotation : 
    Tilting of images +/-5 degrees in angle was done
    - Affine transform : 
    slight skew was tried between points `[5, 5], [20, 5], [5, 20]` and a random value in form of secondary points `[rand1, 5], [rand2, rand1], [5, rand2]`
      where rand1 was chosen as 5 +/- 0.5 and rand2 as 20 +/- 0.5 based on a gaussian distribution.
      
![alt text][image1]

The above picture is a representation of how the processing steps were carried out.

##### Exploratory Data Analysis 
The step was to run the simulator to see how well the car was driving around track one. The performance was not great as the vehicle failed to make proper turns at hard turns and veered off the track.
The histogram of the training set was plotted to see how the distribution looked like. 

![alt text][image4]

It is clear from the above histogram that most of the data is from straight driving. This introduces a bias in the model to always drive straight.
Hence, it was decided to pick only a fraction of this straight driving characteristics and then introduce a gaussian noise in the steering angle of mean as 0.0 degrees and a stddev of 0.05.  
Thus, resulting in the following distribution.

![alt text][image5]

Still, the distribution does not contain comparable amount of data in the entire steering range. so, to help the model learn these features. data is augmented by adding tilt and slight affine transformations. 

![alt text][image6]

The trained model however, sometimes makes harsh turns making the car oscillate. Thus, the stability was not great though the vehicle stayed on course. 
Hence, the maximum steering was limited to `[-0.9,0.9]`  

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the following loss curves
Since, the GPU(NVIDIA GTX 1650 Ti) used here is limited in resources (4Gb of RAM) owing to the large training set, 
I employed an incremental learning pipeline for the model.

| Epochs 1 - 5| Epochs 6 - 10|
|:---:|:---:| 
|![alt text][image11]|![alt text][image12]|  

An adam optimizer with a learn rate of 0.01 was used.  
This proved to make the model learn features better with the batch size of training and validation at 32.
Train-validation split was done at 75-25.
Train and validation set were shuffled before training. 

### 4. Results
`track_1.mp4` contains the video from the pictures from roof mounted center camera images.
`ScreenRecording_track1.mp4` contains the same recorded parallely from unity3d application.  
![video][video0]  
![video][video1]  


### 5. Additional References
1. https://jvgemert.github.io/pub/kayhanCVPR20translationInvarianceCNN.pdf
2. https://stackoverflow.com/questions/40952163/are-modern-cnn-convolutional-neural-network-as-detectnet-rotate-invariant/40953261#40953261
