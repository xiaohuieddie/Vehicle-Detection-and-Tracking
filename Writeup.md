
## Writeup 


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/Multiscale_Window.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/bboxes_and_heat.jpg
[image6]: ./output_images/labels_map.jpg
[image7]: ./output_images/output_bboxes.jpg
[video1]: ./output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the Jupyter notebook file called `SingleImage_Pipeline`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `GRAY` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and and color spaces. In order to make the test accuracy as high as possible, I used all the features from all the three channel from hog images. The exact values for each parameters are shown below:

| Parameters        |  Values | 
|:-------------:|:-------------:| 
| color_space      | LUV        | 
| orient      | 8      |
| pix_per_cell     | (8,8)      |
| cell_per_block      | (2,2)       |
| spatial_size      | (16,16)        |
| hist_bins      | 32        |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG features from all color spaces, the color features and the color histogram features. Moreover, I did data augmentation to flip all the images from data set before training. The final data set and features are shown below:

| Item        | Number   | 
|:-------------:|:-------------:| 
| Car Sample      | 17584        | 
| Not-Car Sample      | 17936      |
| Features     | 5568      |
| Test Accuracy      | 99.1%       |
| True positive percentage      | 99.28%        |

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the multiscale window search. The size of windows highly depends on its location in the image. All the windows can be demostrated as below:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on multi scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  Moreover, I stored the heatmaps from previous frames and then averaged them to make the bounding boxes more stable.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult problems I faced are the detection for the white car and how to ignore the false positive. In order to increase the test accuracy, I augmented the dataset by flipping the dataset. As a result of that, the white car can be easily detected. In terms of the problem of false positive, especially in the video implementation, I recorded the past few frames and the according heatmaps so that I will set a threshold to test if the detected boxes are confident enough to be true positive. 

My pipline are more likely to fail among the area where the y pixel starts from 400 to 600 because the car in that zone will be relatively smaller than regular size and the resized image will be so blurred. In order to make the model more robust, more dataset would help to increase the tendacy to detecting the "small" car, and experimenting with different scale value of searching windows will also contribute to the robustness of the model.



