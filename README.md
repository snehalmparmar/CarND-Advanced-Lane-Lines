# Advanced Lane Lines
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The purpose of this project is to develope a software to create a pipeline to find lane boundaries on the road. For detailed description of the project check [Writeup](https://github.com/snehalmparmar/CarND-Advanced-Lane-Lines/blob/master/Writeup.md) here.

![Advanced lane lines](https://github.com/snehalmparmar/CarND-Advanced-Lane-Lines/blob/master/videos_output/project_video.gif)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  The video in `test_videos` is for testing pipeline on video. The processed images and video are placed in 
`output_images` and `videos_output` respectively.

## Overview

To work on this project consider following steps to setup on your computer:

1: Install [Anaconda](https://www.anaconda.com/distribution/) and [OpenCV4](https://sourceforge.net/projects/opencvlibrary/files/4.1.1/opencv-4.1.1-vc14_vc15.exe/download) on you computer.

2: Setup new `conda` [project_environment_setup](https://github.com/snehalmparmar/CarND-LaneLines-P1/blob/master/project_environment_setup.yml) using this yml script

3: Activate `conda` environmet each time you want to work with this or any other openCV project.

4: Run the Jupyter Notebook and check the `test_videos_output` for results.

---

## Installation Procedure

**Dowanload** [Anaconda](https://www.anaconda.com/distribution/) and [OpenCV4](https://sourceforge.net/projects/opencvlibrary/files/4.1.1/opencv-4.1.1-vc14_vc15.exe/download) based on your operating system `Windows`, `Mac`, `Linux`

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (Link)][lin64] | [64-bit (Link)][mac64] | [64-bit (Link)][win64]
| 32-bit | [32-bit (Link)][lin32] |  | [32-bit (Link)][win32]

[win64]: https://www.anaconda.com/distribution/#windows
[win32]: https://www.anaconda.com/distribution/#windows
[mac64]: https://www.anaconda.com/distribution/#macos
[lin64]: https://www.anaconda.com/distribution/#linux
[lin32]: https://www.anaconda.com/distribution/#linux

**Install** Anaconda on your computer.

**Unzip** OpenCV inside Anaconda folder. Makesure the OpenCV folder name is 'opencv', if not rename it to 'opencv' to match with the [project_environment_setup](https://github.com/snehalmparmar/CarND-LaneLines-P1/blob/master/project_environment_setup.yml)

**SetUp and Activate** `conda` environment using following steps:

  Step1: Place the [project_environment_setup](https://github.com/snehalmparmar/CarND-LaneLines-P1/blob/master/project_environment_setup.yml) yml script inside the directory where Anaconda folder is available. Don't place it inside Anaconda folder.
  
  Step2: Open Anaconda prompt and execute following commands.
  
    conda env create -f project_environment_setup.yml
    
    conda activate carnd
    
    python -m ipykernel install --user --name carnd --display-name "Python 3.7 (carnd)"
    
    jupyter notebook
  
  Step3: If you have GIT installed in your computer, clone this project. Or download in your local drive. The command4 mentioned above will open jupyter notebook, navigate to the project you have cloned/dowanloaded and open `P1.ipynb`
  
**Run** `P1.ipynb` and check outputs in image folders and video output directory.

**Deactivate** `carnd` environment using following command
  
    conda deactivate carnd
    
**Questions & Feedback**

Contact me about this project and self driving car in general. I would be happy to connect with you.

* Linkedin: [Snehal Parmar](https://www.linkedin.com/in/snehal-parmar-239a9112/)
* Email: [snehalparmar.nvs@gmail.com](mailto:snehalparmar.nvs@gmail.com)
  

