import cv2
import numpy as np
import math
import os

# Global variables
resized_image = None
original_image = None
previous_result_string: str = ""
show_orange_box_detection = False  # Flag to toggle between circle and box detection
current_image_index = 0
image_files = []


def on_change(x):
    # This function will be called when trackbar values change
    # We will just pass since we'll process the image in the main loop
    pass


def detect_orange_box(image) -> tuple[int, int, int, int]:
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

    if not contours:
        return 0, 0, 0, 0  # Return default values if no contours found

    # Find the largest contour (assuming it's the orange box)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h


def load_image(image_path):
    global original_image, resized_image, previous_result_string

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return False

    # Scale factor for resizing (0.65 means 65% of the original size)
    scale_factor = 0.65

    # Calculate new dimensions based on the scale factor
    width = int(original_image.shape[1] * scale_factor)
    height = int(original_image.shape[0] * scale_factor)

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(original_image, (width, height))

    # Reset previous result string to force update
    previous_result_string = ""

    return True


def load_next_image():
    global current_image_index, image_files

    if not image_files:
        return False

    current_image_index = (current_image_index + 1) % len(image_files)
    image_path = image_files[current_image_index]

    print(f"Loading image {current_image_index + 1}/{len(image_files)}: {os.path.basename(image_path)}")
    return load_image(image_path)


def load_previous_image():
    global current_image_index, image_files

    if not image_files:
        return False

    current_image_index = (current_image_index - 1) % len(image_files)
    image_path = image_files[current_image_index]

    print(f"Loading image {current_image_index + 1}/{len(image_files)}: {os.path.basename(image_path)}")
    return load_image(image_path)


def main():
    global resized_image, original_image, previous_result_string, show_orange_box_detection
    global current_image_index, image_files

    # Directory containing images
    image_directory = "../PRO_2/pliki/"  # Change this to your directory path

    # Supported image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # Get all image files from the directory
    image_files = []
    try:
        for file in os.listdir(image_directory):
            extension = os.path.splitext(file)[1].lower()
            if extension in image_extensions:
                image_files.append(os.path.join(image_directory, file))
    except Exception as e:
        print(f"Error accessing directory '{image_directory}': {e}")
        return

    if not image_files:
        print(f"No image files found in directory '{image_directory}'")
        return

    # Sort files alphabetically
    image_files.sort()

    # First create the windows before adding any trackbars
    cv2.namedWindow('frame')
    cv2.namedWindow('working frame')

    # Create trackbars
    cv2.createTrackbar('low', 'frame', 1, 255, on_change)
    cv2.createTrackbar('high', 'frame', 1, 255, on_change)
    cv2.createTrackbar('blur k size', 'frame', 1, 100, on_change)

    # Load the first image
    current_image_index = 0
    if not load_image(image_files[current_image_index]):
        return

    print("\nControls:")
    print("- Press LEFT ARROW to go to previous image")
    print("- Press RIGHT ARROW to go to next image")
    print("- Press ESC to exit")
    print(f"\nLoaded image 1/{len(image_files)}: {os.path.basename(image_files[current_image_index])}\n")

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
        blured_frame = cv2.GaussianBlur(display_image, (ksize, ksize), 0)
        gray_frame = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = detect_orange_box(display_image)
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        try:
            # Apply HoughCircles
            circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT,
                                       dp=high_color / 10.0,
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
                    if 17 <= i[2] <= 20:
                        counter_5gr += 1
                        area = math.pi * i[2] ** 2
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
                    elif 21 <= i[2] <= 25:
                        counter_5zl += 1
                        area = math.pi * i[2] ** 2
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

            if counter_5zl > 0 and sum_area_5zl > 0:
                result_string += f"\nBox area/5zł area: {round((w * h) / (sum_area_5zl / counter_5zl), 2)}"

            if counter_5gr > 0 and sum_area_5gr > 0:
                result_string += f"\nBox area/5gr area: {round((w * h) / (sum_area_5gr / counter_5gr), 2)}"

            result_string += f"\n5gr counter: {counter_5gr}, 5zł counter: {counter_5zl}"

            if result_string != previous_result_string:
                print(result_string)
                previous_result_string = result_string

        except Exception as e:
            print(f"Error detecting circles: {e}")

        # Add image number to display
        image_info = f"Image {current_image_index + 1}/{len(image_files)}: {os.path.basename(image_files[current_image_index])}"
        cv2.putText(display_image, image_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the processed image
        cv2.imshow("frame", display_image)
        cv2.imshow("working frame", gray_frame)

        # Wait for key press with a short timeout
        key = cv2.waitKey(100) & 0xFF

        # Process key presses
        if key == ord('p'):  # Left arrow or 'a' - previous image
            load_previous_image()
        elif key == ord('n'):  # Right arrow or 'd' - next image
            load_next_image()
        elif key == 27:  # ESC key
            break
        elif key != 255:  # Any other key press
            print("Pressed key:", key)

    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()