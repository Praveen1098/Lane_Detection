import numpy as np
import cv2

class Line():
    def reset(self):
        self.N_FITS = 5  
        self.detected = False
        self.current_x = [[]]
        self.current_y = [[]] 
        self.all_x = [[]] 
        self.all_y = [[]] 
        self.current_fit = np.array([0, 0, 0]) 
        self.all_fit = np.array([0, 0, 0]) 
        self.radius_of_curvature = 0.0 
        self.line_base_pos = 0.0 
        
    def __init__(self):
        self.reset()
        
def reset_lanes():
    left_lane.reset()
    right_lane.reset()

    

US_LANE_WIDTH = 3.7 
US_LANE_LINE_LENGTH = 3.0 
LANE_WIDTH_PIX = 675
LANE_LINE_LENGTH_PIX = 83 

ym_per_pix = US_LANE_LINE_LENGTH/LANE_LINE_LENGTH_PIX 
xm_per_pix = US_LANE_WIDTH/LANE_WIDTH_PIX 

left_lane = Line()
right_lane = Line()

def camera_undistort(img, mtx, dist, debug=False):
    
    img_out = cv2.undistort(img, mtx, dist, None, mtx)
    return img_out

def apply_CLAHE_LAB(img, debug=False):
    
    image_wk = np.copy(img)
    image_wk = cv2.cvtColor(image_wk, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    image_wk[:,:,0] = clahe.apply(image_wk[:,:,0])
    image_wk = cv2.cvtColor(image_wk, cv2.COLOR_LAB2RGB)
        
    return image_wk

def do_perspective_transform(img, M, debug=False):
    
    img_warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return img_warped

def set_perspective_transform(img):
    width = img.shape[1]
    height = img.shape[0]
    
    src = np.array([[150, height],
                [480, 480],
                [720, 480],
                [width, height]], np.float32)
    

    dst = np.array([[300, height],
                [300, 0],
                [980, 0],
                [980, height]], np.float32)
    '''
    dst = np.array([[300, 0],
                [300, height],
                [980, height],
                [980, 0]], np.float32)
    '''
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv


def color_binary_thresh(img, debug=False):

    img_CBTin = np.copy(img)
    
    def color_select(img, val, thresh=(0, 255)):
        channel = img[:,:,val]
        binary_output = np.zeros_like(channel)
        if thresh[0] == thresh[1]:
            binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
        else:
            binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
        return binary_output

    img_LAB = cv2.cvtColor(img_CBTin, cv2.COLOR_RGB2LAB)
    
    # Make YELLOW binary image
    binary_yellow_L = color_select(img_LAB, 0, thresh=(130, 255))
    binary_yellow_A = color_select(img_LAB, 1, thresh=(100, 150))
    binary_yellow_B = color_select(img_LAB, 2, thresh=(145, 210))
    
    binary_yellow_R = color_select(img_CBTin, 0, thresh=(255, 255))
    binary_yellow_G = color_select(img_CBTin, 1, thresh=(180, 255))
    binary_yellow_b = color_select(img_CBTin, 2, thresh=(0, 170))
    binary_yellow = np.zeros_like(binary_yellow_L)
    binary_yellow[((binary_yellow_R == 1) & (binary_yellow_G == 1) & (binary_yellow_b == 1))|((binary_yellow_L == 1) & (binary_yellow_A == 1) & (binary_yellow_B == 1))] = 1
    
    # Make WHITE binary image
    
    binary_white_L = color_select(img_LAB, 0, thresh=(230, 255))
    binary_white_A = color_select(img_LAB, 1, thresh=(120, 140))
    binary_white_B = color_select(img_LAB, 2, thresh=(120, 140))

    binary_white_R = color_select(img_CBTin, 0, thresh=(100, 255))
    binary_white_G = color_select(img_CBTin, 1, thresh=(100, 255))
    binary_white_b = color_select(img_CBTin, 2, thresh=(200, 255))
    binary_white = np.zeros_like(binary_white_L)
    
    # binary_white[(binary_white_L == 1) & (binary_white_A == 1) & (binary_white_B == 1)] = 1
    binary_white[((binary_white_R == 1) & (binary_white_G == 1) & (binary_white_b == 1)) | ((binary_white_L == 1) & (binary_white_A == 1) & (binary_white_B == 1))] = 1
    
    binary_color = np.zeros_like(binary_yellow)
    binary_color[(binary_yellow == 1) | (binary_white == 1)] = 1
    
    binary_color_float = binary_color.astype(np.float)
    binary_color_float = cv2.blur(binary_color_float, (3, 3))
    binary_color_blur = np.zeros_like(binary_color)
    binary_color_blur[ (binary_color_float > 0.0) ] = 1

    return binary_color_blur



def sobel_denoise(binary_img, kernel=5, thresh=0.7, debug=False):
    
    binary_float = binary_img.astype(np.float)
    binary_float = cv2.blur(binary_float, (kernel, kernel))
    binary_denoise = np.zeros_like(binary_img)
    binary_denoise[ (binary_float > thresh) ] = 1

    return binary_denoise
    
def sobel_x_thresh(img, sobel_kernel=3, thresh=(0, 255), debug=False):
   
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    
    return binary_output

def sobel_mag_thresh(img, sobel_kernel=3, thresh=(0, 255), debug=False):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
        
    return binary_output

def sobel_dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2), debug=False):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir, dtype=np.uint8)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    
    return binary_output

def sobel_magdir_combine(binary_mag, binary_dir, debug=False):

    binary_output = np.zeros_like(binary_mag)
    binary_output[(binary_mag == 1) & (binary_dir == 1)] = 1
        
    return binary_output

def grad_combo_binary_thresh(img_gradx, img_magdir, debug=False):
    
    binary_gradient = np.zeros_like(img_gradx)
    binary_gradient[ (img_gradx == 1) | (img_magdir == 1) ] = 1
    
    return binary_gradient


def combine_color_gradient_threshold(binary_color, binary_gradient, debug=False):
    
    if (left_lane.detected == True) and (right_lane.detected == True):
        binary_final = binary_color
    else:
        binary_final = cv2.bitwise_and(binary_color, binary_gradient)

    return binary_final


def search_for_new_lanes(binary_warped, debug=False):
    
    N_WINDOWS = 8
    FOUND_MIN_PIX = 50 
    HIST_PEAK_MARGIN_M = 2.2 
    NEW_WINDOW_MARGIN_M = 0.5 
    NEW_LANE_MARGIN_M = 0.2
    
    
    left_lane_xpts = []
    left_lane_ypts = []
    right_lane_xpts = []
    right_lane_ypts = []
    hist_peak_margin = np.int(HIST_PEAK_MARGIN_M / xm_per_pix)
    window_margin = np.int(NEW_WINDOW_MARGIN_M / xm_per_pix)
    lane_margin = np.int(NEW_LANE_MARGIN_M / xm_per_pix)
    

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    histogram = np.sum(binary_warped[binary_warped.shape[0]//4*3:,:], axis=0)
    if np.sum(histogram) == 0:
       
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    first_peak_x = np.argmax(histogram)
    look_right_x = min(first_peak_x + hist_peak_margin, binary_warped.shape[1]-1)
    look_left_x = max(first_peak_x - hist_peak_margin, 1)
    right_of_first_peak = max(histogram[look_right_x:])
    left_of_first_peak = max(histogram[:look_left_x])
    if right_of_first_peak > left_of_first_peak:
        
        win_center_leftx_base = first_peak_x
        win_center_rightx_base = np.argmax(histogram[look_right_x:]) + look_right_x
    else:
        
        win_center_rightx_base = first_peak_x
        win_center_leftx_base = np.argmax(histogram[:look_left_x])

    
    win_height = np.int(binary_warped.shape[0]/N_WINDOWS)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    
    win_leftx_momentum = 0
    win_rightx_momentum = 0
    win_center_leftx_current = win_center_leftx_base
    win_center_rightx_current = win_center_rightx_base

    
    current_lane_width = win_center_rightx_current - win_center_leftx_current

    
    for n_window in range(N_WINDOWS):
        win_center_leftx_prev = win_center_leftx_current 
        win_center_rightx_prev = win_center_rightx_current

        
        win_y_low = binary_warped.shape[0] - (n_window+1)*win_height
        win_y_high = binary_warped.shape[0] - n_window*win_height
        win_xleft_low = win_center_leftx_current - window_margin
        win_xleft_high = win_center_leftx_current + window_margin
        win_xright_low = win_center_rightx_current - window_margin
        win_xright_high = win_center_rightx_current + window_margin

        
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2) 

        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                        & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                        & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        num_left = len(good_left_inds)
        num_right = len(good_right_inds)

        
        if (num_left > FOUND_MIN_PIX) and (num_right > FOUND_MIN_PIX):
            
            win_center_leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            win_center_rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            current_lane_width = win_center_rightx_current - win_center_leftx_current
        elif (num_left < FOUND_MIN_PIX) and (num_right > FOUND_MIN_PIX):
            
            win_center_rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            win_center_leftx_current = win_center_rightx_current - current_lane_width
        elif (num_left > FOUND_MIN_PIX) and (num_right < FOUND_MIN_PIX):
            
            win_center_leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            win_center_rightx_current = win_center_leftx_current + current_lane_width
        elif (num_left < FOUND_MIN_PIX) and (num_right < FOUND_MIN_PIX):
    
            win_center_leftx_current = win_center_leftx_prev + win_leftx_momentum
            win_center_rightx_current = win_center_rightx_prev + win_rightx_momentum

        win_leftx_momentum = (win_center_leftx_current - win_center_leftx_prev)
        win_rightx_momentum = (win_center_rightx_current - win_center_rightx_prev)

        win_xleft_low = win_center_leftx_current - lane_margin
        win_xleft_high = win_center_leftx_current + lane_margin
        win_xright_low = win_center_rightx_current - lane_margin
        win_xright_high = win_center_rightx_current + lane_margin
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (255,0,255), 2) 
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (255,0,255), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                        & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                        & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_xpts.append(nonzerox[good_left_inds])
        left_lane_ypts.append(nonzeroy[good_left_inds])
        right_lane_xpts.append(nonzerox[good_right_inds])
        right_lane_ypts.append(nonzeroy[good_right_inds])

    left_lane.current_x = np.concatenate(left_lane_xpts)
    left_lane.current_y = np.concatenate(left_lane_ypts)
    right_lane.current_x = np.concatenate(right_lane_xpts)
    right_lane.current_y = np.concatenate(right_lane_ypts)
    
    if (len(left_lane.current_x) > 0) and (len(left_lane.current_x) == len(left_lane.current_y)):
        left_lane.current_fit = np.polyfit(left_lane.current_y, left_lane.current_x, 2)
    if (len(right_lane.current_x) > 0) and (len(right_lane.current_x) == len(right_lane.current_y)):
        right_lane.current_fit = np.polyfit(right_lane.current_y, right_lane.current_x, 2)
    
    
def search_for_existing_lanes(binary_warped, debug=False):
    DETECTED_LANE_WINDOW_MARGIN = 0.3
    left_fit = left_lane.current_fit
    right_fit = right_lane.current_fit
    margin = np.int(DETECTED_LANE_WINDOW_MARGIN / xm_per_pix)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                    & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                     & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    left_lane.current_x = nonzerox[left_lane_inds]
    left_lane.current_y = nonzeroy[left_lane_inds] 
    right_lane.current_x = nonzerox[right_lane_inds]
    right_lane.current_y = nonzeroy[right_lane_inds]

    if (len(left_lane.current_x) > 0) and (len(left_lane.current_x) == len(left_lane.current_y)):
        left_lane.current_fit = np.polyfit(left_lane.current_y, left_lane.current_x, 2)
    if (len(right_lane.current_x) > 0) and (len(right_lane.current_x) == len(right_lane.current_y)):
        right_lane.current_fit = np.polyfit(right_lane.current_y, right_lane.current_x, 2) 

def lane_validity_check(binary_warped, debug=False):
    LANE_DETECT_MIN_PIX = 50 
    LANE_DETECT_MAX_PIX = 70000 
    LANE_WIDTH_MIN_M = 1.0 
    LANE_WIDTH_MAX_M = 6.0 

    left_ok = True
    right_ok = True
    min_lane_width = 0
    max_lane_width = 0
    left_fit = left_lane.current_fit
    right_fit = right_lane.current_fit
    left_x_num = len(left_lane.current_x)
    left_y_num = len(left_lane.current_y)
    right_x_num = len(right_lane.current_x)
    right_y_num = len(right_lane.current_y)
    lane_width_min_pix = np.int(LANE_WIDTH_MIN_M / xm_per_pix)
    lane_width_max_pix = np.int(LANE_WIDTH_MAX_M / xm_per_pix)

    y_eval = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftx_eval = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    rightx_eval = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    min_lane_width = min(rightx_eval - leftx_eval)
    max_lane_width = max(rightx_eval - leftx_eval)
    
    if min_lane_width < lane_width_min_pix:
        left_ok = False
        right_ok = False

    if max_lane_width > lane_width_max_pix:
        left_ok = False
        right_ok = False
     
    if ((left_x_num != left_y_num)
        or (left_x_num < LANE_DETECT_MIN_PIX) 
        or (left_x_num > LANE_DETECT_MAX_PIX)):
        left_ok = False
        right_ok = False
        
    if ((right_x_num != right_y_num)
        or (right_x_num < LANE_DETECT_MIN_PIX)
        or (right_x_num > LANE_DETECT_MAX_PIX)):
        left_ok = False
        right_ok = False
    
    left_lane.detected = left_ok
    right_lane.detected = right_ok

def update_lane_pts_and_polyfit():
    if left_lane.detected == True and right_lane.detected == True:
        while len(left_lane.all_x) >= left_lane.N_FITS:
            left_lane.all_x.pop(0)
            left_lane.all_y.pop(0)
        while len(right_lane.all_x) >= right_lane.N_FITS:
            right_lane.all_x.pop(0)
            right_lane.all_y.pop(0)

        left_lane.all_x.append(left_lane.current_x)
        left_lane.all_y.append(left_lane.current_y)
        right_lane.all_x.append(right_lane.current_x)
        right_lane.all_y.append(right_lane.current_y)

        left_lane_xpts = np.concatenate(left_lane.all_x)
        left_lane_ypts = np.concatenate(left_lane.all_y)
        right_lane_xpts = np.concatenate(right_lane.all_x)
        right_lane_ypts = np.concatenate(right_lane.all_y)

        left_lane.all_fit = np.polyfit(left_lane_ypts, left_lane_xpts, 2)
        right_lane.all_fit = np.polyfit(right_lane_ypts, right_lane_xpts, 2) 

def make_lane_area_image(binary_warped, debug=False):
    left_fit = left_lane.all_fit
    right_fit = right_lane.all_fit
    left_x = left_lane.current_x
    left_y = left_lane.current_y
    right_x = right_lane.current_x
    right_y = right_lane.current_y

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    img_lane_area = np.dstack((binary_warped, binary_warped, binary_warped))*0

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    if left_lane.detected == True and right_lane.detected == True:
        lane_area_color = (0, 50, 0)
    else:
        lane_area_color = (50, 0, 0)
    cv2.fillPoly(img_lane_area, np.int_([pts]), lane_area_color)
    img_lane_area[left_y, left_x] = [255, 0, 0]
    img_lane_area[right_y, right_x] = [255, 0, 255]
        
        
    return img_lane_area

def calc_radius_and_offset(img, debug):
    leftx = np.concatenate(left_lane.all_x)
    lefty = np.concatenate(left_lane.all_y)
    rightx = np.concatenate(right_lane.all_x)
    righty = np.concatenate(right_lane.all_y)
    y_eval = img.shape[0] # bottom y val
    
    if (len(leftx) > 0) and (len(rightx) > 0):
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        left_curverad = (((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)
                          / np.absolute(2*left_fit_cr[0]))
        right_curverad = (((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)
                          / np.absolute(2*right_fit_cr[0]))

        left_lane.radius_of_curvature = left_curverad
        right_lane.radius_of_curvature = right_curverad
        p_left = np.poly1d(left_fit_cr)
        p_right = np.poly1d(right_fit_cr)
        x_midpoint = np.int(img.shape[1]/2)*xm_per_pix

        left_lane.line_base_pos = x_midpoint - p_left(y_eval*ym_per_pix)
        right_lane.line_base_pos = p_right(y_eval*ym_per_pix) - x_midpoint



def overlay_lane_area(img_undist, img_lane_lines, Minv, debug=False):
    def overlay_text(image, text, pos):
        cv2.putText(image, text, pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0,0,0), thickness=10)
        cv2.putText(image, text, pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255,255,255), thickness=2)
        
    newwarp = cv2.warpPerspective(img_lane_lines, Minv, (img_lane_lines.shape[1], img_lane_lines.shape[0])) 
    overlaid_result = cv2.addWeighted(img_undist, 1, newwarp, 1, 0)
    left_curverad = left_lane.radius_of_curvature
    right_curverad = right_lane.radius_of_curvature
    lane_offset = right_lane.line_base_pos - left_lane.line_base_pos
    if lane_offset < 0:
        side = 'right'
    else:
        side = 'left'
    overlay_text(overlaid_result, 'Radius of curvature: L={:.0f} m, R={:.0f} m'
                                  .format(left_curverad, right_curverad), (50,100))
    overlay_text(overlaid_result, 'Offset from lane center: {:.2f} m {}'
                                  .format(abs(lane_offset), side), (50,150))
    
    
    return overlaid_result

def my_lane_pipeline(img, debug=False):
    K = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
         [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    dist = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
    
    K = np.array(K)
    dist = np.array(dist)
    img_undist = camera_undistort(img, K, dist, debug=False)

    img_CLAHE = apply_CLAHE_LAB(img_undist, debug=False)

    H, H_inv = set_perspective_transform(img_undist)
    img_warped = do_perspective_transform(img_undist, H, debug=False)
    
    binary_color = color_binary_thresh(img_CLAHE, debug=False)
    binary_color_warped = do_perspective_transform(binary_color, H, debug=False)

    if (left_lane.detected == False) or (right_lane.detected == False):
        binary_gradx_warped = sobel_x_thresh(img_warped, thresh=(10, 255), debug=False)
        binary_gradx_warped = sobel_denoise(binary_gradx_warped, kernel=5, thresh=0.7, debug=False)
        binary_mag = sobel_mag_thresh(img_undist, sobel_kernel=3, thresh=(5, 255), debug=False)
        binary_dir = sobel_dir_thresh(img_undist, sobel_kernel=3, thresh=(0.5, 1.3), debug=False)
        binary_magdir = sobel_magdir_combine(binary_mag, binary_dir, debug=False)
        binary_magdir = sobel_denoise(binary_magdir, kernel=5, thresh=0.7, debug=False)
        binary_magdir_warped = do_perspective_transform(binary_magdir, H, debug=False)
        binary_gradient_warped = grad_combo_binary_thresh(binary_gradx_warped, binary_magdir_warped, debug=False)
    else:
        binary_gradient_warped = None

    binary_final = combine_color_gradient_threshold(binary_color_warped, binary_gradient_warped, debug=False)
    
    if (left_lane.detected == False) or (right_lane.detected == False):
        search_for_new_lanes(binary_final, debug=False)
    else:
        search_for_existing_lanes(binary_final, debug=False)
        
    lane_validity_check(binary_final, debug=False)
    update_lane_pts_and_polyfit()
    img_lane_lines = make_lane_area_image(binary_final, debug=False)

    calc_radius_and_offset(img_lane_lines, debug=False)
    
    overlaid_result = overlay_lane_area(img_undist, img_lane_lines, H_inv, debug)
    
    return overlaid_result

if __name__ == '__main__':
    cap = cv2.VideoCapture("C:\\Users\\Praveen\\ENPM673_Project_2\\data_2\\Input_1.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('Output2.mp4', fourcc, 30, (1280, 720))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lanes = my_lane_pipeline(frame)
        lanes = cv2.cvtColor(lanes, cv2.COLOR_RGB2BGR)
        cv2.imshow('result', lanes)
        out.write(lanes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
