# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/rotation.jpg "Before and after rotation"
[image3]: ./examples/normalization.jpg "Before and after normalization"
[image4]: ./examples/image4.jpg "Traffic Sign 1"
[image5]: ./examples/image5.jpg "Traffic Sign 2"
[image6]: ./examples/image6.jpg "Traffic Sign 3"
[image7]: ./examples/image7.jpg "Traffic Sign 4"
[image8]: ./examples/image9.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? **34799**
* The size of the validation set is ? **4410**
* The size of test set is ? **12630**
* The shape of a traffic sign image is ? **(32, 32, 3)**
* The number of unique classes/labels in the data set is ? **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the labels are distributed across Training, Validation, and Test sets. 

![alt text][image1]

* The trend looks fairly similar across Training, Validation and Test image sets
* We see that first labels are having more images. We will keep this in mind when we run our model for test images.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle and augment training images by rotating each image by -10 degrees and by zooming by ratio 1.25. The rotated images were appended to the original X_test and y_test list.

Here is an example of a traffic sign image before and after rotation. 

![alt text][image2]

Some statistics: (Note: the code is running augmentation code twice, so the final count of training examples will be twice from what we started with)
..........
Number of Training examples before augmenting = 34799
Number of Training examples after augmenting = 69598

..........
Number of Training examples before augmenting = 69598
Number of Training examples after augmenting = 139196

The reason I decided to increase the training examples is to increase the efficiency of the model as it will have more images to train on.   

As a second step, I normalized the image data so as to reduce the number of shades and to increase the performance of the model.

Here is an example of an original image and an normalized image:

![alt text][image3]

**Please Note**: The image in jupyter notebook might be different, as we are doing random shuffling. The image displayed might get changed in each run.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model architecture is located in code cell #8 of Ipython Jupyter notebook.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, 'VALID' padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120  									|
| RELU					|												|
| Dropout               | Keep probability 0.70                         |
| Fully connected		| Outputs 84  									|
| RELU					|												|
| Dropout               | Keep probability 0.70                         |
| Fully connected		| Outputs 43 logits								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in code cell 10 of [jupyter notebook](Traffic_Sign_Classifier.ipynb). 

To train the model, following hyperparameters were used:
* Number of epochs: 40 (Experimental way)
* Batch size = 128
* Learning rate = 0.001
* Optimizer - Adam algorithm
* Dropout = 0.7 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in code cell #11 of [jupyter notebook](Traffic_Sign_Classifier.ipynb).

This solution is based on modified LeNet-5 architecture which is introduced in Udacity classroom lessons.

My final model results were:
* training set accuracy of ? **0.99905**
* validation set accuracy of ? **0.949**
* test set accuracy of ? **0.931**


* To start with I implemented the LeNet-5 architecture that was mentioned in class.

* Since the training set accuracy was below 90%, the model was adjusted by including dropout and by adding a regularizer. This was done to reduce overfitting. 

* Preprocessing steps: Did augmentation (by rotating image by -10 degrees and appending to original test list) and normalizaton. This reduced overfitting and increased efficiency of model. 

* Epoch was tuned to 40 after few trials, and beta was chosen as 0.1

* With a training accuracy of **0.99905**, validation set accuracy of **0.949**,and test set accuracy of  **0.931** provide evidence that the modified LeNet-5 architecture model is working well.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. Images are resized. 

![alt text][image4] This image might be difficult to classify if model cannot distinctly deduce 'No Passing' and 'No Passing for vehicles over 3.5 metric tons'. 

![alt text][image5] This image might be difficult to classify as the training data for 'Speed limit (20km/h)' is lesser when compared to higher speed limits.

![alt text][image6] This image might be difficult to classify as the image can get misrepresented due to training data having lesser count for this label.

![alt text][image7] This image might be difficult to classify because arrows can prove misleading to choose right only or go straight only. 

![alt text][image8] This image might be difficult to classify as it can be confused for pedestrians crossing.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Passing      		| No Passing                                 	| 
| Speed limit (20km/h)  | No passing for vehicles over 3.5 metric tons	|
| Roadwork				| Roadwork										|
| Go straight or right 	| Go straight or right			 				|
| Children crossing		| Children crossing    							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of  **93.1%**.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is 40% sure that this is a sign for 'No passing' an 25% sureity for 'No passing for vehicles over 3.5 metric tons'.


| Probability         	|     Prediction	        					   | 
|:---------------------:|:---------------------------------------------:   | 
| .41775       			| 9 : No passing                                   | 
| .25477   				| 10 : No passing for vehicles over 3.5 metric tons|
| .2376					| 19 : Dangerous curve to the left 			       |
| .05479	      		| 11 : Right-of-way at the next intersection       |
| .00958				| 29 : Bicycles crossing      	 	 	 		   |


For the second image, the model is having a 77% sureity that it is an image of 'No passing for vehicles over 3.5 metric tons' and 11% for 'Speed limit (20km/h)'.

| Probability         	|     Prediction	        					   | 
|:---------------------:|:---------------------------------------------:   | 
| .77679       			| 10 : No passing for vehicles over 3.5 metric tons| 
| .11369   				| 0 :  Speed limit (20km/h)                        |
| .05886				| 5 : Speed limit (80km/h)                         |
| .0181	   		     	| 14 : 	Stop	                                   |
| .0049			        | 1 : Speed limit (30km/h)		 	 	 		   |


For the third image, the model is having a 88% sureity that it is an image of Label 25: 'Road work'.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .88477       			| 25 : Road work                                | 
| .01859   				| 1 : Speed limit (30km/h)                      |
| .01047				| 26 : Vehicles over 3.5 metric tons prohibited |
| .00963	   			| 11 : Right-of-way at the next intersection    |
| .00783			    | 37 : Go straight or left              	    |


For the fourth image, the model is having a 100% sureity that it is an image of 'Go straight or right'.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.       		      	| 36 : Go straight or right                     | 
| .0   				    | 9 :  No passing                               |
| .0					| 41 : End of no passing                        |
| .0	   		     	| 35 : Ahead only     			                |
| .0			        | 20 : Dangerous curve to the right	 	 		|


For the fifth and final image, the model is having a 100% sureity that it is an image of 'Children crossing'.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.       			    | 28 : Children crossing                        | 
| .0   				    | 3 : Speed limit (60km/h)                      |
| .0					| 23 : Slippery road                            |
| .0	   		     	| 9 : No passing                 				|
| .0			        | 39 : Keep left	                 	 		|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
