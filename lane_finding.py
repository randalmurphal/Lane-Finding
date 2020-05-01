import cv2
import numpy as np
import matplotlib.pyplot as plt


# Finds the high gradients in the image (sharp change in brightness in pixels)
def canny(img):
    # Convert to grayscale so that image only has 1 intensity value (brightness)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Get RGB -> Grayscale img

    '''
        Use Guassian blur to remove image noise
          - Smoothes image by setting a kernel and setting center value to
              the avg of the neighboring pixel values (with varying weights)
              - kernel determines how many neighboring pixels to compare to
    '''
    # Applying Gauss blur is optional bc it automatically applies with Canny filter
    #   5x5 kernel, stand. dev. w/r x = 0 -> w/r y = 0 (by default equal to sigmaX)
    blur = cv2.GaussianBlur(src=gray,ksize=(5, 5), sigmaX=0)

    '''
        Can see matrix of image as x, y coords. Can apply functions with x, y vars
            - Canny function = apply derivative w/r to x and y directions
                - Measures change in intensity w/r to the adjacent pixel values
                    and traces the strongest gradients with white pixels
    '''
    # Only takes in gradients b/t thresholds (not too high or too low of changes)
    #   - Could maybe be trained to determine the best thresholds? (needs a lot of data?)
    canny = cv2.Canny(image=blur, threshold1=50, threshold2=150)
    return canny


# Takes in canny image & specifies region of interest -- the line/lines that the road follows (lane lines)
#   - limits x & y coords to trace a triangle (enclosed view of lines)
def region_of_interest(img):
    height = img.shape[0] # height of the image

    # Create triangle (off of plotted image coords) to narrow down region of interest
    # - need to set as an array of polygons bc fillPoly fills an area bounded by array of polygons (not just 1)
    polys = np.array( [ [(200, height),(1100, height),(550, 250)] ] ) # Array of 1 shape, w/ 3 vertices

    # Create a mask which will hold the region of interest (on top of blank black matrix)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polys, 255) # Fills mask with polynomial (triangle) -- 255 = white

    '''
        Use bitwise_and b/t canny image (with lane line gradients) and mask (with white triangle)
        to extract the high gradient values within the region of interest (triangle). This allows
        us to get the lane lines, and none of the background values (unless in region of interest)
            - bitwise_and with white pixels keeps the value, so keeps lane lines from canny img
    '''
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


# Takes in image where we will later display lines onto, as well as image with lines
def display_lines(img, lines):
    line_img = np.zeros_like(img)

    if lines is not None:
        # Draw lines onto blank black image
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=5)

    return line_img


# Get avg of slopes/y-ints of left and right lane lines (instead of many smaller lines)
def average_slope_intercept(img, lines):
    l_fit = [] # avg of left lane line
    r_fit = [] # avg of right lane line

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1) # Poly of degree = 1 (linear)
        slope = params[0]
        intercept = params[1]
        # If slope is on left or right? (if slanted left or right)
        if slope < 0:
            l_fit.append((slope, intercept))
        else:
            r_fit.append((slope, intercept))

    l_fit_avg = np.average(l_fit, axis=0)
    r_fit_avg = np.average(r_fit, axis=0)
    # Blend the lines to make one connected line, not a bunch of fragmented lines
    l_line = make_coordinates(img, l_fit_avg)
    r_line = make_coordinates(img, r_fit_avg)

    return np.array([l_line, r_line])


# Get the x & y coords from the slope & intercept (y = mx + b)
def make_coordinates(img, line_params):
    slope, intercept = line_params
    # y-axis starts 0 at top of img
    y1 = img.shape[0] # height
    y2 = int(y1*(3/5)) # line will go 3/5 of the image height

    # y = mx + b -- solve for x1 & x2
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])


def find_lane_lines(img):
    ### For detecting lines in an image

    # If not making a copy, any changes road would change original img variable
    road = np.copy(img)

    canny_img = canny(road)
    cropped_img = region_of_interest(canny_img)

    '''
    Hough Space:
        - From y = mx + b, hough space is a coord system that is m vs b instead of x vs y
        - For each point, can have multiple m/b pairs that go through that point.
            - Each of these can be plotted in hough space to give a straight line

        - To find line that connects two (or more) points, can look at point of intersection in hough space
            and that gives the m/b pair for that line

        - Can calculate line of "best fit" for set of points by taking hough space "bin" with the highest
            number of intersections for that series of points

        - Better to use polar coords to overcome div. by 0 error for vertical line slopes
            - Hough space gives sinusoidal curve instead of straight lines, but still used same (with intersections)
    '''

    lines = cv2.HoughLinesP(image=cropped_img, lines=np.array([]), rho=2, theta=(np.pi/180),
                            threshold=100, minLineLength=40, maxLineGap=5)

    avg_lines = average_slope_intercept(road, lines)
    line_img = display_lines(road, avg_lines)

    # Put lines onto the original image -- takes sum of road_img + line_img
    #   alpha = weight of 1st image, beta = weight of 2nd img, gamma = scalar onto summed image value
    combo_img = cv2.addWeighted(src1=road, alpha=0.8, src2=line_img, beta=1, gamma=1)

    return combo_img


if __name__ == '__main__':

    ### Detecting Lane lines from an image

    img = cv2.imread('test_image.jpg')

    lane_lines_img = find_lane_lines(img)

    cv2.imshow("Result", lane_lines_img)
    cv2.waitKey(0) # Displays image (ms)


    ### For detecting lines from a video

    cap = cv2.VideoCapture('test_video.mp4')

    while(cap.isOpened()):
        _, frame = cap.read()

        lane_lines_img = find_lane_lines(frame)

        cv2.imshow("Result", lane_lines_img)
        cv2.waitKey(1) # Wait 1ms between frames

        if cv2.waitKey(1) == ord('q'):
            break # break if u press q

    cap.release()
    cv2.destroyAllWindows()
