# import cv2
# import numpy as np
#
# # Global variables
# resized_image = None
# original_image = None
#
#
# def change_h(x):
#     process_image()  # Call the processing function when trackbar changes
#
#
# def process_image():
#     global resized_image
#
#     if resized_image is None:
#         return
#
#     # Create a copy of the processed image
#     c_img = resized_image.copy()
#
#     # Get trackbar values
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     ksize = max(1, cv2.getTrackbarPos('ksize', 'obrazek'))  # Make sure ksize is at least 1
#
#     if ksize % 2 == 0:  # ksize must be odd for blur
#         ksize += 1
#
#     # Convert to grayscale
#     gimg = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
#
#     # Apply blur
#     bimg = cv2.blur(gimg, (ksize, ksize))
#
#     try:
#         # Apply Hough transform to detect circles
#         # Make sure high_color and low_color are not zero (or swap if needed)
#         dp = max(1, high_color) / 100 if high_color > 0 else 1  # dp should not be zero
#         min_dist = max(20, low_color)  # Minimum distance should be reasonable
#
#         circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, dp,
#                                    min_dist, param1=50, param2=30,
#                                    minRadius=0, maxRadius=0)
#
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#
#             # Draw detected circles
#             for i in circles[0, :]:
#                 # Draw the outer circle
#                 cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#                 # Draw the center of the circle
#                 cv2.circle(c_img, (i[0], i[1]), 2, (0, 0, 255), 3)
#     except Exception as e:
#         print(f"Error detecting circles: {e}")
#
#     # Display the result
#     cv2.imshow('obrazek', c_img)

# high: int = 0
# low: int = 0
# def on_change() -> None:
#     global high, low

# def main():
#     # global resized_image, original_image
#
#     # First create the window before adding any trackbars
#     # cv2.namedWindow('obrazek')
#
#     # Create trackbars
#     # cv2.createTrackbar('low', 'obrazek', 50, 255, on_change)
#     # cv2.createTrackbar('high', 'obrazek', 50, 255, onChange=on_change)
#     # cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)
#
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     # Try to load the image
#     image_path = "../PRO_2/pliki/tray1.jpg"
#     original_image = cv2.imread(image_path)
#
#     if original_image is None:
#         print(f"Error: Could not load image from '{image_path}'")
#         print("Please check if the file path is correct and the file exists.")
#         return
#
#     # Scale factor for resizing (0.5 means half the original size)
#     scale_factor = 0.65
#
#     # Calculate new dimensions based on the scale factor
#     width = int(original_image.shape[1] * scale_factor)
#     height = int(original_image.shape[0] * scale_factor)
#
#     # Resize the image while maintaining aspect ratio
#     resized_image = cv2.resize(original_image, (width, height))
#     blured_frame = cv2.GaussianBlur(resized_image, (7, 7), 0)  # troche bluru
#     gray_frame = cv2.cvtColor(blured_frame, cv2.COLOR_RGB2GRAY)
#
#     circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, high_color, low_color)
#     print(circles)  # Wyświetlenie wykrytych okręgów (surowe dane)
#
#     # Zaokrąglenie współrzędnych wykrytych okręgów do liczb całkowitych
#     circles = np.uint16(np.around(circles))
#     print(circles)  # Wyświetlenie zaokrąglonych współrzędnych okręgów
#
#     # Iteracja po wykrytych okręgach i rysowanie ich na obrazie
#     for i in circles[0, :]:
#         # Rysowanie okręgu na obrazie (środek: (i[0], i[1]), promień: i[2])
#         cv2.circle(gray_frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
#
#     cv2.imshow("frame", gray_frame)
#     # Initial processing
#     # process_image()
#
#     # Wait for user to close window
#     print("Press any key to exit...")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()

import cv2
import numpy as np
import math

# Global variables
resized_image = None
original_image = None
previous_result_string: str = ""
show_orange_box_detection = False  # Flag to toggle between circle and box detection


def on_change(x):
    # This function will be called when trackbar values change
    # We will just pass since we'll process the image in the main loop
    pass


def detect_orange_box(image) -> tuple[int, int, int, int]:
    """Detect orange box in the given image"""
    # if image is None:
    #     return None, "No image provided"

    # Create a copy for displaying results
    output_image = image.copy()

    # Convert to HSV color space for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for orange color in HSV
    # These values may need adjustment based on lighting conditions
    lower_orange = np.array([5, 100, 150])
    upper_orange = np.array([25, 255, 255])

    # Create a mask for the orange color
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found
    # if not contours:
    #     return None

    # Find the largest contour (assuming it's the orange box)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw the bounding rectangle
    # cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the actual contour
    # cv2.drawContours(output_image, [largest_contour], 0, (255, 0, 0), 2)

    # Draw corner points for better visualization
    # corners = np.array([
    #     [x, y],  # Top-left
    #     [x + w, y],  # Top-right
    #     [x + w, y + h],  # Bottom-right
    #     [x, y + h]  # Bottom-left
    # ])
    #
    # for corner in corners:
    #     cv2.circle(output_image, tuple(corner), 5, (0, 0, 255), -1)
    #
    # # Calculate dimensions and perimeter
    # perimeter = cv2.arcLength(largest_contour, True)
    # area = cv2.contourArea(largest_contour)
    #
    # result_text = f"Orange box dimensions: {w}x{h} pixels\n"
    # result_text += f"Perimeter: {perimeter:.2f} pixels\n"
    # result_text += f"Area: {area:.2f} square pixels"

    return x, y, w, h


def main():
    global resized_image, original_image, previous_result_string, show_orange_box_detection

    # First create the windows before adding any trackbars
    cv2.namedWindow('frame')
    cv2.namedWindow('working frame')

    # Create trackbars - using 'frame' as the window name to match your imshow
    cv2.createTrackbar('low', 'frame', 1, 255, on_change)  # Initial value 30, max 100
    cv2.createTrackbar('high', 'frame', 1, 255, on_change)  # Initial value 10, max 30
    cv2.createTrackbar('blur k size', 'frame', 1, 100, on_change)  # Initial value 10, max 30

    # Try to load the image
    image_path = "../PRO_2/pliki/tray2.jpg"
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'")
        print("Please check if the file path is correct and the file exists.")
        return

    # Scale factor for resizing (0.5 means half the original size)
    scale_factor = 0.65

    # Calculate new dimensions based on the scale factor
    width = int(original_image.shape[1] * scale_factor)
    height = int(original_image.shape[0] * scale_factor)

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(original_image, (width, height))

    # print("Controls:")
    # print("- Press 'o' to toggle orange box detection")
    # print("- Press any other key to exit")

    # Main processing loop
    while True:
        # Get current positions of trackbars
        low_color = cv2.getTrackbarPos('low', 'frame')
        high_color = cv2.getTrackbarPos('high', 'frame')
        ksize = cv2.getTrackbarPos('blur k size', 'frame')

        # Make sure we have valid values (prevent zeros that could cause errors)
        low_color = max(1, low_color)
        high_color = max(1, high_color)
        if ksize % 2 == 0:
            ksize += 1

        # Create a copy of the resized image to draw on
        display_image = resized_image.copy()

        # Process the image
        blured_frame = cv2.GaussianBlur(display_image, (ksize, ksize), 0)  # troche bluru
        gray_frame = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2GRAY)  # Changed RGB to BGR to match OpenCV format

        x, y, w, h = detect_orange_box(display_image)
        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        try:
            # Apply HoughCircles
            circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT,
                                       dp=high_color / 10.0,  # Scaling to get a reasonable dp value
                                       minDist=low_color,
                                       param1=50, param2=30,
                                       minRadius=17, maxRadius=26)

            # Only process if circles were found
            result_string = "No circles found"
            sum_inside_box: float = 0.0
            sum_outside_box: float = 0.0
            counter_5gr: int = 0
            counter_5zl: int = 0
            sum_area_5zl: int = 0
            sum_area_5gr: int = 0
            if circles is not None:
                circles = np.uint16(np.around(circles))

                # Draw circles on the colored image
                counter: int = 1
                result_string = f"Found {len(circles[0])} circles"
                for i in circles[0, :]:
                    cv2.circle(display_image, center=(i[0], i[1]), radius=i[2], color=(0, 255, 0), thickness=1)
                    if 17 <= i[2] <= 22:
                        counter_5gr += 1
                        area = math.pi * i[2]**2
                        sum_area_5gr += area
                        cv2.drawMarker(display_image, position=(i[0], i[1]), color=(255, 0, 0), thickness=2,
                                       markerType=cv2.MARKER_CROSS)
                        result_string += f"\nCircle No: {counter} is 5gr, area = {round(area, 2)}"
                        if x < i[0] < x + w and y < i[1] < y + h:
                            result_string += ", inside the box"
                            sum_inside_box += 0.05
                        else:
                            result_string += ", outside the box"
                            sum_outside_box += 0.05
                    elif 23 <= i[2] <= 26:
                        counter_5zl += 1
                        area = math.pi * i[2]**2
                        sum_area_5zl += area
                        cv2.drawMarker(display_image, position=(i[0], i[1]), color=(0, 0, 255), thickness=2,
                                       markerType=cv2.MARKER_CROSS)
                        result_string += f"\nCircle No: {counter} is 5zl, area = {round(area, 2)}"
                        if x < i[0] < x + w and y < i[1] < y + h:
                            result_string += ", inside the box"
                            sum_inside_box += 5.0
                        else:
                            result_string += ", outside the box"
                            sum_outside_box += 5.0
                    cv2.putText(display_image, f"{counter}", org=(i[0] + 20, i[1] + 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(0, 0, 0), thickness=2)
                    counter += 1

            result_string += f"\nSum inside the box: {round(sum_inside_box, 2)}, outside the box: {round(sum_outside_box, 2)}"
            result_string += f"\nBox area: {round(w * h, 2)}"
            result_string += f"\nBox area/5zł area: {(w*h)/(sum_area_5zl/counter_5zl)}"
            result_string += f"\nBox area/5gr area: {round((w*h)/(sum_area_5gr/counter_5gr), 2)}"
            result_string += f"\n5gr counter: {counter_5gr}, 5zł counter: {counter_5zl}"
            if result_string != previous_result_string:
                print(result_string)
                previous_result_string = result_string


        except Exception as e:
            print(f"Error detecting circles: {e}")

        # Display the processed image
        cv2.imshow("frame", display_image)
        cv2.imshow("working frame", gray_frame)

        # Wait for key press with a short timeout
        key = cv2.waitKey(100) & 0xFF

        # Process key presses
        if key == ord('o'):  # Toggle orange box detection mode
            show_orange_box_detection = not show_orange_box_detection
            previous_result_string = ""  # Reset previous result string to force update
            print(f"Orange box detection: {'ON' if show_orange_box_detection else 'OFF'}")
        elif key != 255:  # Any other key press
            break

    # Wait for user to close window
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()