import cv2
import numpy as np

width = 640
height = 480

kernel = np.ones((9, 9), np.uint8) # for closing

color_ranges = {
    'red': {'lower1': np.array([0, 100, 100]), 'upper1': np.array([10, 255, 255]),
            'lower2': np.array([160, 100, 100]), 'upper2': np.array([179, 255, 255])},
    'green': {'lower': np.array([35, 100, 100]), 'upper': np.array([85, 255, 255])},
    'blue': {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
    'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
    'orange': {'lower': np.array([10, 100, 100]), 'upper': np.array([25, 255, 255])},
    'purple': {'lower': np.array([130, 100, 100]), 'upper': np.array([160, 255, 255])},
    'pink': {'lower': np.array([145, 30, 150]), 'upper': np.array([165, 120, 255])},
}

def make_mask_for_green(hsv):
    return cv2.inRange(hsv, color_ranges["green"]["lower"], color_ranges["green"]["upper"])


def make_mask_for_red(hsv):
    mask1 = cv2.inRange(hsv, color_ranges["red"]["lower1"], color_ranges["red"]["upper1"])
    mask2 = cv2.inRange(hsv, color_ranges["red"]["lower2"], color_ranges["red"]["upper2"])

    return mask1 + mask2

def make_four_color_mask(hsv):
    red_mask1 = cv2.inRange(hsv, color_ranges["red"]["lower1"], color_ranges["red"]["upper1"])
    red_mask2 = cv2.inRange(hsv, color_ranges["red"]["lower2"], color_ranges["red"]["upper2"])
    green_mask = cv2.inRange(hsv, color_ranges["green"]["lower"], color_ranges["green"]["upper"])
    blue_mask = cv2.inRange(hsv, color_ranges["blue"]["lower"], color_ranges["blue"]["upper"])
    yellow_mask = cv2.inRange(hsv, color_ranges["yellow"]["lower"], color_ranges["yellow"]["upper"])

    return red_mask1 + red_mask2 + green_mask + blue_mask + yellow_mask



def mark_object(contours, frame):
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawMarker(frame, (int(cx), int(cy)), color=(255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, f"x: {x}, y:{y}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == '__main__':
    # cap = cv2.VideoCapture("PRO_1/resources/movingball.mp4")
    # cap = cv2.VideoCapture("PRO_1/resources/IMG_1967.MOV")
    cap = cv2.VideoCapture("PRO_1/resources/rgb_ball_720.mp4")
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS) # Get video properties for the output video

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_path = 'output_marked_video.avi'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        resized_frame = cv2.resize(frame, (width, height)) # resize the frame
        blured_frame = cv2.GaussianBlur(resized_frame, (13, 13), 0) # troche bluru


        hsv = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2HSV) # Convert BGR to HSV
        mask = make_four_color_mask(hsv)
        res = cv2.bitwise_and(blured_frame, blured_frame, mask=mask) # Bitwise-AND mask and original image

        closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel) # zamknięcie aby odfiltrować otwory w kuli

        marked_frame = resized_frame.copy() # Create a copy for drawing markers

        contours, _ = cv2.findContours(mask, 1, 2) # Find contours in the mask
        mark_object(contours, marked_frame)

        out.write(marked_frame)

        # Display the frame
        cv2.imshow("Original", resized_frame)
        cv2.imshow("Object detection", marked_frame)
        cv2.imshow('Mask', closing)
        if cv2.waitKey(25) & 0xFF == ord('q'): # Press 'q' to exit
            break
    cap.release() # Release the video capture object and close all windows
    cv2.destroyAllWindows()