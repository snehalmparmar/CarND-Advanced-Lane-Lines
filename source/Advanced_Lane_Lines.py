import os
import glob
import pickle
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

#from pipeline import Line
from pipelineDefinitions import get_points_for_calibration
from pipelineDefinitions import abs_sobel_thresh
from pipelineDefinitions import mag_thresh
from pipelineDefinitions import dir_threshold
from pipelineDefinitions import color_threshold
from pipelineDefinitions import color_gradient_threshold
from pipelineDefinitions import perspective_transform
from pipelineDefinitions import sliding_windows_polyfit
from pipelineDefinitions import visualize_searching
from pipelineDefinitions import compute_curvature_radius
from pipelineDefinitions import draw_lane
from pipelineDefinitions import draw_data
from pipelineDefinitions import save_transformed_image

#Parameters for calibrations
cameraMatrix = []
distortionCoeff = []
calibrationFile = "wide_dist_pickle.p"

#Parameters for gradient threshold

class Line:
	def __init__(self, max_lines=5):
		# Was the line detected in the last iteration?
		self.detected = False

		# Number of failed detections
		self.failures = 0

		# Max number of last lines
		self.max_lines = max_lines

		#Polynomial coefficients for the most recent fit
		self.recent_fit = []

		#Polynomial coefficients averaged over the last n iterations
		self.best_fit = None

		#Radius of curvature of the lines
		self.radius_of_curvature = None

		#Distance in meters of vehicle center from the line
		self.center_dist = 0

		#Lane lane_width
		self.lane_width = 0

	def reset(self):
		del self.recent_fit[:]
		self.best_fit = None
		self.detected = False
		self. failures = 0
		self.radius_of_curvature = None
		self.center_dist = 0
		self.lane_width = 0

	def sanity_check(self, left_fit, right_fit, left_curve_radius, right_curve_radius, top_lane_width, bottom_lane_width):
		# Check that both lines have similar curvature
		if abs(left_curve_radius - right_curve_radius) > 1500:
			return False

		#Check that both lines are separated by approximately the right distance horizontally
		lane_width = (top_lane_width - bottom_lane_width)/2
		if abs(2.0 - lane_width)>0.5:
			return False

		#Check that both lines are roughly parallel
		if abs(top_lane_width - bottom_lane_width)>0.7:
			return False

		return True

	def update_lines(self, left_fit, right_fit, left_curve_radius, right_curve_radius, center_dist, top_lane_width, bottom_lane_width):
		is_detected_ok = self.sanity_check(left_fit, right_fit, left_curve_radius, right_curve_radius, top_lane_width, bottom_lane_width)==True

		#Update history with the current detection
		if(left_fit is not None and right_fit is not None and is_detected_ok):
			self.detected = True
			if(len(self.recent_fit)==self.max_lines):
				#Remove the oldest fit from the history
				self.recent_fit.pop(0)
			# Add the new lines
			self.recent_fit.append((left_fit, right_fit))
			self.radius_of_curvature = (left_curve_radius, right_curve_radius)
			self.center_dist = center_dist
			self.lane_width = (top_lane_width + bottom_lane_width)/2
			#update best fit
			self.best_fit = np.average(self.recent_fit, axis=0)

		# Do not take into account this failed detection
		else:
			self.detected = False
			self.failures += 1


	def AdvancedLanesLineFindingPipeline(self, img, output_dir = "", file_name = "", save_steps = False):
		#Step1: Distortion correction
		undistorted = cv2.undistort(img, cameraMatrix, distortionCoeff, None, cameraMatrix)

		#Step2: Perspective transformation
		warped, Minv = perspective_transform(undistorted)

		#Step3: gradient threshold
		gradient = color_gradient_threshold(warped)

		#Step4: Detect Lines
		if(self.best_fit is None or self.failures >= self.max_lines):
			self.reset()
			polyfit_image, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_windows_polyfit(gradient)
		else:
			polyfit_image, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_windows_polyfit(gradient, self.best_fit[0], self.best_fit[1])

		#Step5: Compute Radius
		left_curve_radius, right_curve_radius, center_dist, top_lane_width, bottom_lane_width = compute_curvature_radius(gradient, left_fit, right_fit, left_lane_inds, right_lane_inds)

		#Step6: Return image with information
		self.update_lines(left_fit, right_fit, left_curve_radius, right_curve_radius, center_dist, top_lane_width, bottom_lane_width)
		if self.detected:
			#use current detected lines
			lanes = draw_lane(img, gradient, Minv, self.recent_fit[-1][0], self.recent_fit[-1][1])
			result = draw_data(lanes, polyfit_image, gradient, self.radius_of_curvature[0], self.radius_of_curvature[1], self.center_dist, self.lane_width, True, False)
		elif not self.detected and self.best_fit is not None:
			#use history
			lanes = draw_lane(img, gradient, Minv, self.best_fit[0], self.best_fit[1])
			result = draw_data(lanes, polyfit_image, gradient, left_curve_radius, right_curve_radius, center_dist, (top_lane_width+bottom_lane_width)/2, False, True)
		else:
			#If no history is awailable, draw current 'falsy' detection
			lanes = draw_lane(img, gradient, Minv, left_fit, right_fit)
			result = draw_data(lanes, polyfit_image, gradient, left_curve_radius, right_curve_radius, center_dist, (top_lane_width+bottom_lane_width)/2, False, False)

		#save images
		output_dir += file_name + "/"
		save_transformed_image(img, undistorted, False, output_dir, file_name + "_0")
		save_transformed_image(img, warped, False, output_dir, file_name + "_1")
		save_transformed_image(img, gradient, True, output_dir, file_name + "_2")
		save_transformed_image(None, polyfit_image, False, output_dir, file_name + "_3")
		save_transformed_image(None, lanes, False, output_dir, file_name + "_4")

		return result

if __name__ == "__main__":
	#perform camera calibrations
	if os.path.exists(calibrationFile):
		print("Read in the calibration data...\n")
		wide_dist_pickle = pickle.load(open(calibrationFile, "rb"))
		cameraMatrix = wide_dist_pickle["mtx"]
		distortionCoeff = wide_dist_pickle["dist"]
	else:
		print("Calibration Camera...")
		objpoints, imgpoints = get_points_for_calibration(9,6)
		img = cv2.imread('./test_images/test1.jpg')
		ret, cameraMatrix, distortionCoeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
		print("Save the camera calibration result for later use\n")
		wide_dist_pickle = {}
		wide_dist_pickle["mtx"] = cameraMatrix
		wide_dist_pickle["dist"] = distortionCoeff
		pickle.dump(wide_dist_pickle, open(calibrationFile, "wb"))

#Create directory if it does not exist
output_video_dir = "video_output/"
if not os.path.isdir(output_video_dir):
	os.makedirs(output_video_dir)
output_image_dir = "images_output/"
if not os.path.isdir(output_image_dir):
	os.makedirs(ouput_image_dir)

#Run pipeline on test images and save image transformation steps
'''test_dir = "test_images/"
for file_name in os.listdir(test_dir):
	print("Run pipeline for '" + file_name + "'")
	line = Line()
	#Read image and convert to RGB
	img = cv2.cvtColor(cv2.imread(test_dir + file_name), cv2.COLOR_BGR2RGB)
	#Process image
	line.AdvancedLanesLineFindingPipeline(img, output_image_dir, file_name.split(".")[0],True)'''

# Run pipeline on project and challenge videos
test_dir = "test_videos/"
for file_name in os.listdir(test_dir):
	print("\nRun pipeline for '" + file_name + "'...")
	line_video = Line(max_lines=3)
	video_input = VideoFileClip(test_dir + file_name)
	processed_video = video_input.fl_image(line_video.AdvancedLanesLineFindingPipeline)
	processed_video.write_videofile(output_video_dir + file_name, audio=False)
