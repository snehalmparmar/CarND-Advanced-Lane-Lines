import os
import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Parameters for calibration
mtx = []
dist = []
calibration_file = "calibration_pickle.p"

# Parameters for gradient threshold
ksize = 7
gradx_thresh = (50, 255)
grady_thresh = (50, 255)
magni_thresh = (25, 255)
dir_thresh = (0., 0.09)
hls_thresh = (110, 255)
rgb_thresh = (220, 255)

def get_points_for_calibration(nx, ny):
    # Prepare object points
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    for idx, fname in enumerate(images):
        # Read image
        img = cv2.imread(fname)

        # Convert image in grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners (for an 9x6 board)
        ret, corners = cv2.findChessboardCorners(img, (nx,ny), None)

        if (ret == True):
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imwrite('./output_images/calibration/corners_found' + str(idx) + '.jpg', img)

    return (objpoints, imgpoints)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = 1 if orient == 'x' else 0
    y = 1 if orient == 'y' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    sobel_scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(sobel_scaled)
    grad_binary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(np.add(np.square(sobelx), np.square(sobely)))
    sobel_scaled = np.uint8(255*mag/np.max(mag))
    # Apply threshold
    mag_binary = np.zeros_like(sobel_scaled)
    mag_binary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # Apply threshold
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return dir_binary

def color_threshold(img, hls_thresh=(0, 255), rgb_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S_channel = hls[:,:,2]
    R_channel = img[:,:,0]

    r_binary = np.zeros_like(R_channel)
    r_binary[(R_channel >= rgb_thresh[0]) & (R_channel <= rgb_thresh[1])] = 1

    s_binary = np.zeros_like(S_channel)
    s_binary[(S_channel >= hls_thresh[0]) & (S_channel <= hls_thresh[1])] = 1

    color_binary = np.zeros_like(R_channel)
    color_binary[(s_binary == 1) & (r_binary == 1)] = 1

    if color_binary.all() == 0:
        return r_binary
    else:
        return color_binary

def color_gradient_threshold(img):
    # Apply color gradient (S channel)
    color_binary = color_threshold(img, hls_thresh=hls_thresh, rgb_thresh=rgb_thresh)

    # Apply gradient thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=gradx_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=grady_thresh)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=magni_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)

    # Combine gradient & color thresholds
    gradient_binary = np.zeros_like(dir_binary)
    gradient_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined = np.zeros_like(dir_binary)
    combined[(gradient_binary == 1) | (color_binary == 1)] = 1

    return combined

def perspective_transform(img):
    img_size = (img.shape[0], img.shape[1])
    # Define src and dst points
    x_center = img_size[1]/2
    x_offset=120
    src = np.float32([(x_offset,img_size[0]), (x_center-54, 450), (x_center+54, 450), (img_size[1]-x_offset,img_size[0])])
    dst = np.float32([(x_offset,img_size[1]), (x_offset,0), (img_size[0]-x_offset, 0), (img_size[0]-x_offset,img_size[1])])
    # Apply transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return (cv2.warpPerspective(img, M, (img_size[0], img_size[1]), flags=cv2.INTER_LINEAR), Minv)

def sliding_windows_polyfit(img, previous_left_fit=None, previous_right_fit=None):

    # Get indices of all nonzero pixels along x and y axis
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    # Set margin for searching
    margin = 100
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Look for lines from scratch ('blind search')
    if (previous_left_fit is None or previous_right_fit is None):
        # Compute the histogram of the lower half image. It gives us the 2 pics where the lanes are located.
        histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
        # Separate the left part of the histogram from the right one. This is our starting point.
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base =  np.argmax(histogram[midpoint:]) + midpoint

        # Split the image in 9 horizontal strips
        n_windows = 9
        # Set height of windows
        window_height = int(img.shape[0]/n_windows)
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(n_windows):
            # Compute the windows boundaries
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_leftx_low = leftx_current - margin
            win_leftx_high = leftx_current + margin
            win_rightx_low = rightx_current - margin
            win_rightx_high = rightx_current + margin

            # Identify non zero pixels within left and right windows
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_leftx_low) & (nonzerox < win_leftx_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_rightx_low) & (nonzerox < win_rightx_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If non zeros pixels > minpix, recenter the next window on their mean
            if (len(good_left_inds) > minpix):
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if (len(good_right_inds) > minpix):
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Use last polynomial fit
    else:
        # Compute the windows boundaries
        previous_left_x = previous_left_fit[0]*nonzeroy**2 + previous_left_fit[1]*nonzeroy + previous_left_fit[2]
        win_leftx_low = previous_left_x - margin
        win_leftx_high =  previous_left_x + margin
        previous_right_x = previous_right_fit[0]*nonzeroy**2 + previous_right_fit[1]*nonzeroy + previous_right_fit[2]
        win_rightx_low = previous_right_x - margin
        win_rightx_high =  previous_right_x + margin
        # Identify non zero pixels within left and right windows
        good_left_inds = ((nonzerox >= win_leftx_low) & (nonzerox < win_leftx_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_rightx_low) & (nonzerox < win_rightx_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img = visualize_searching(img, nonzerox, nonzeroy, left_fit, right_fit, margin, left_lane_inds, right_lane_inds)

    return (out_img, left_fit, right_fit, left_lane_inds, right_lane_inds)

def visualize_searching(img, nonzerox, nonzeroy, left_fit, right_fit, margin, left_lane_inds, right_lane_inds):
    # Create an output image to draw on and visualize the result
    out_img = np.uint8(np.dstack((img, img, img))*255)
    window_img = np.zeros_like(out_img)

    # Color left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    '''plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)'''

    return result

def compute_curvature_radius(img, left_fit, right_fit, left_lane_inds, right_lane_inds):
    # Get indices of all nonzero pixels along x and y axis
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_converted = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_converted = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Choose point to compute curvature just in front of the car
    yvalue = img.shape[0]

    # Compute curvature radius
    left_curv_radius = ((1 + (2*left_fit_converted[0]*yvalue + left_fit_converted[1])**2)**1.5) / (2*np.absolute(left_fit_converted[0]))
    right_curv_radius = ((1 + (2*right_fit_converted[0]*yvalue + right_fit_converted[1])**2)**1.5) / (2*np.absolute(right_fit_converted[0]))

    # Compute distance in meters of vehicle center from the line
    car_center = img.shape[1]/2  # we assume the camera is centered in the car
    lane_center = ((left_fit[0]*yvalue**2 + left_fit[1]*yvalue + left_fit[2]) + (right_fit[0]*yvalue**2 + right_fit[1]*yvalue + right_fit[2])) / 2
    center_dist = (lane_center - car_center) * xm_per_pix

    # Compute lane width
    top_yvalue = 10
    bottom_yvalue = img.shape[0]
    top_leftx = left_fit[0]*bottom_yvalue**2 + left_fit[1]*bottom_yvalue + left_fit[2]
    bottom_leftx = left_fit[0]*bottom_yvalue**2 + left_fit[1]*bottom_yvalue + left_fit[2]
    top_rightx = right_fit[0]*bottom_yvalue**2 + right_fit[1]*bottom_yvalue + right_fit[2]
    bottom_rightx = right_fit[0]*bottom_yvalue**2 + right_fit[1]*bottom_yvalue + right_fit[2]
    bottom_lane_width = abs(bottom_leftx - bottom_rightx) * xm_per_pix
    top_lane_width = abs(top_leftx - top_rightx) * xm_per_pix

    return (left_curv_radius, right_curv_radius, center_dist, top_lane_width, bottom_lane_width)

def draw_lane(img, warped, Minv, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[1]-1, img.shape[1])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result

def draw_data(img, top_img, bottom_img, left_curv_radius, right_curv_radius, center_dist, lane_width, is_detected, use_history):
    result = np.copy(img)

    # Add text to the original image
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Left radius curvature: ' + '{:04.2f}'.format(left_curv_radius) + 'm'
    cv2.putText(result, text, (50, 70), font, 1, (255,255,255), 2, cv2.LINE_AA)

    text = 'Right radius curvature: ' + '{:04.2f}'.format(right_curv_radius) + 'm'
    cv2.putText(result, text, (50, 100), font, 1, (255,255,255), 2, cv2.LINE_AA)

    text = 'Lane width: ' + '{:04.2f}'.format(lane_width) + 'm'
    cv2.putText(result, text, (50, 130), font, 1, (255,255,255), 2, cv2.LINE_AA)

    if center_dist > 0:
        text = 'Vehicule position: {:04.2f}'.format(center_dist) + 'm left of center'
    else:
        text = 'Vehicule position: {:04.2f}'.format(center_dist) + 'm right of center'
    cv2.putText(result, text, (50, 160), font, 1, (255,255,255), 2, cv2.LINE_AA)

    if is_detected:
        color = (0,255,0)
        text = "Good detection"
    elif not is_detected and use_history:
        color = (255,215,0)
        text = "Bad detection --> Use history"
    else:
        color = (255,0,0)
        text = "Bad detection, no history --> Use it anyway"

    cv2.putText(result, text, (50, 190), font, 1, color, 2, cv2.LINE_AA)
    # Add transformed images to the original image
    mask = np.ones_like(top_img)*255
    img_1 = cv2.resize(top_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bottom_img_3_channels = np.uint8(np.dstack((bottom_img, bottom_img, bottom_img))*255)
    img_2 = cv2.resize(bottom_img_3_channels, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    offset = 50
    endy = offset+img_1.shape[0]
    endx_img_1 = img.shape[1]-offset
    startx_img_1 = endx_img_1-img_1.shape[1]
    endx_img_2 = startx_img_1-25
    startx_img_2 = endx_img_2-img_2.shape[1]

    result[offset:endy, startx_img_1:endx_img_1] = img_1
    result[offset:endy, startx_img_2:endx_img_2] = img_2

    return result

def save_image_transform(original, transformed, is_gray, output_dir, file_name):
    plt.clf()
    if original is not None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(original)
        ax1.set_title('Original Image', fontsize=30)
        if (is_gray == True):
            ax2.imshow(transformed, cmap='gray')
        else:
            ax2.imshow(transformed)
        ax2.set_title('Result Image', fontsize=30)
    else:
        plt.imshow(transformed)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + file_name + '.jpg')

class Line:
    def __init__(self, max_lines=5):
        # Was the line detected in the last iteration?
        self.detected = False
        # Number of failed detection
        self.failures = 0
        # Max number of last lines
        self.max_lines = max_lines
        # Polynomial coefficients for the most recent fit
        self.recent_fit = []
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # Radius of curvature of the lines
        self.radius_of_curvature = None
        # Distance in meters of vehicle center from the line
        self.center_dist = 0
        # Lane width
        self.lane_width = 0

    def reset(self):
        del self.recent_fit[:]
        self.best_fit = None
        self.detected = False
        self.failures = 0
        self.radius_of_curvature = None
        self.center_dist = 0
        self.lane_width = 0

    def sanity_check(self, left_fit, right_fit, left_curv_radius, right_curv_radius, top_lane_width, bottom_lane_width):
        # Check that both lines have similar curvature
        if abs(left_curv_radius - right_curv_radius) > 1500:
            return False

        # Check that both lines are separated by approximately the right distance horizontally
        lane_width = (top_lane_width + bottom_lane_width) / 2
        if abs(2.0 - lane_width) > 0.5:
            return False

        # Check that both lines are roughly parallel
        if abs(top_lane_width - bottom_lane_width) > 0.7:
            return False

        return True

    def update_lines(self, left_fit, right_fit, left_curv_radius, right_curv_radius, center_dist, top_lane_width, bottom_lane_width):
        is_detection_ok = self.sanity_check(left_fit, right_fit, left_curv_radius, right_curv_radius, top_lane_width, bottom_lane_width) == True

        # Update history with the current detection
        if (left_fit is not None and right_fit is not None and is_detection_ok):
            self.detected = True
            if (len(self.recent_fit) == self.max_lines):
                # Remove the oldest fit from the history
                self.recent_fit.pop(0)
            # Add the new lines
            self.recent_fit.append((left_fit, right_fit))
            self.radius_of_curvature = (left_curv_radius, right_curv_radius)
            self.center_dist = center_dist
            self.lane_width = (top_lane_width + bottom_lane_width) / 2
            # Update best fit
            self.best_fit = np.average(self.recent_fit, axis=0)

        # Do not take into account this failed detection
        else:
            self.detected = False
            self.failures += 1

    def process_img(self, img, output_dir = "", file_name = "", save_steps = False):

        ### 1. Distortion correction ###
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        ### 2. Perspective transformation ###
        warped, Minv = perspective_transform(undistorted)

        ### 3. Gradient threshold ###
        gradient = color_gradient_threshold(warped)

        ### 4. Detect lines ###
        if (self.best_fit is None or self.failures >= self.max_lines):
            self.reset()
            polyfit_image, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_windows_polyfit(gradient)
        else:
            polyfit_image, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_windows_polyfit(gradient, self.best_fit[0], self.best_fit[1])

        ### 5. Compute radius ###
        left_curv_radius, right_curv_radius, center_dist, top_lane_width, bottom_lane_width = compute_curvature_radius(gradient, left_fit, right_fit, left_lane_inds, right_lane_inds)

        ### 6. Return image with information ###
        self.update_lines(left_fit, right_fit, left_curv_radius, right_curv_radius, center_dist, top_lane_width, bottom_lane_width)

        if self.detected:
            # Use current detected line
            lanes = draw_lane(img, gradient, Minv, self.recent_fit[-1][0], self.recent_fit[-1][1])
            result = draw_data(lanes, polyfit_image, gradient, self.radius_of_curvature[0], self.radius_of_curvature[1], self.center_dist, self.lane_width, True, False)
        elif not self.detected and self.best_fit is not None:
            # Use history
            lanes = draw_lane(img, gradient, Minv, self.best_fit[0], self.best_fit[1])
            result = draw_data(lanes, polyfit_image, gradient, left_curv_radius, right_curv_radius, center_dist, (top_lane_width+bottom_lane_width)/2, False, True)
        else:
            # In case there's no history draw the current 'falsy' detection --> better than nothing
            lanes = draw_lane(img, gradient, Minv, left_fit, right_fit)
            result = draw_data(lanes, polyfit_image, gradient, left_curv_radius, right_curv_radius, center_dist, (top_lane_width+bottom_lane_width)/2, False, False)

        # Save images
        if save_steps:
            output_dir += file_name + "/"
            save_image_transform(img, undistorted, False, output_dir, file_name + "_0")
            save_image_transform(img, warped, False, output_dir, file_name + "_1")
            save_image_transform(img, gradient, True, output_dir, file_name + "_2")
            save_image_transform(None, polyfit_image, False, output_dir, file_name + "_3")
            save_image_transform(None, lanes, False, output_dir, file_name + "_4")

        return result

if __name__ == '__main__':
    ### Camera calibration ###
    if os.path.exists(calibration_file):
        print("Read in the calibration data\n")
        calibration_pickle = pickle.load(open(calibration_file, "rb"))
        mtx = calibration_pickle["mtx"]
        dist = calibration_pickle["dist"]
    else:
        print("Calibrate camera...")
        objpoints, imgpoints = get_points_for_calibration(9, 6)
        img = cv2.imread('./test_images/test1.jpg')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

        print("Save the camera calibration result for later use\n")
        calibration_pickle = {}
        calibration_pickle["mtx"] = mtx
        calibration_pickle["dist"] = dist
        pickle.dump(calibration_pickle, open(calibration_file, "wb"))

    # Create directory if it does not exist
    output_video_dir = "videos_output/"
    if not os.path.isdir(output_video_dir):
        os.makedirs(output_video_dir)
    output_image_dir = "images_output/"
    if not os.path.isdir(output_image_dir):
        os.makedirs(output_image_dir)

    # Run pipeline on test images and save image transformation steps
    test_dir = "test_images/"
    for file_name in os.listdir(test_dir):
        print("Run pipeline for '" + file_name + "'")
        line = Line()
        # Read image and convert to RGB
        img = cv2.cvtColor(cv2.imread(test_dir + file_name), cv2.COLOR_BGR2RGB)
        # Process image
        line.process_img(img, output_image_dir, file_name.split(".")[0], True)

    # Run pipeline on project and challenge videos
    '''test_dir = "test_videos/"
    for file_name in os.listdir(test_dir):
        print("\nRun pipeline for '" + file_name + "'...")
        line_video = Line(max_lines=3)
        video_input = VideoFileClip(test_dir + file_name)
        processed_video = video_input.fl_image(line_video.process_img)
        processed_video.write_videofile(output_video_dir + file_name, audio=False)'''
