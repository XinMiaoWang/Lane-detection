import numpy as np
import cv2
import matplotlib.pyplot as plt

def canny(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blur2 = cv2.medianBlur(gray_img, 5)
    # cv2.namedWindow("My blur1", 0);
    # cv2.resizeWindow("My blur1", 640, 480)
    # cv2.imshow("My blur1", blur)
    #
    # cv2.namedWindow("My blur2", 0);
    # cv2.resizeWindow("My blur2", 640, 480)
    # cv2.imshow("My blur2", blur2)
    # canny1 = cv2.Canny(blur, 50, 150)
    # cv2.namedWindow("canny1", 0);
    # cv2.resizeWindow("canny1", 640, 480)
    # cv2.imshow("canny1", canny1)

    canny2 = cv2.Canny(blur2, 50, 150)
    # cv2.namedWindow("canny2", 0);
    # cv2.resizeWindow("canny2", 640, 480)
    # cv2.imshow("canny2", canny2)
    # plt.imshow(canny2)
    # plt.show()
    return canny2

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_img

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    # cv2.namedWindow("mask", 0);
    # cv2.resizeWindow("mask", 640, 480)
    # cv2.imshow("mask", mask)
    return masked_img

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_avg)
    right_line = make_coordinates(img, right_fit_avg)
    return np.array([left_line, right_line])

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# 讀影片-----Start
cap = cv2.VideoCapture("../data/origin.mp4")

while (cap.isOpened()):
    ret, frame = cap.read()

    canny_frame = canny(frame)
    cropped_img = region_of_interest(canny_frame)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, avg_lines)
    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    cv2.imshow("result", combo_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# 讀影片-----End



# 讀照片-----Start
# img = cv2.imread('../data/origin.jpg')
# cv2.imshow("My Image", img)
# canny_frame = canny(img)
# cropped_img = region_of_interest(canny_frame)
# lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# avg_lines = average_slope_intercept(img, lines)
# line_img = display_lines(img, avg_lines)
# combo_img = cv2.addWeighted(img, 0.8, line_img, 1, 1)
# cv2.namedWindow("result",0);
# cv2.resizeWindow("result", 640, 480)
# cv2.imshow("result", combo_img)
# cv2.waitKey(0)
# 讀照片-----End