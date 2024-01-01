
import cv2 as cv
import numpy as np
import time
import math

###########################################################################
# Initialization Settings
###########################################################################

# Load calibration data
calib_data_path = "../calib_data/MultiMatrix.npz"  # Adjust the path as needed
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]


# ArUco Marker settings
MARKER_SIZE_MM = 35  # Millimeters

# HSV color ranges for contour detection
lower_ranges = {
    "red": np.array([125, 106, 140]),
    "green": np.array([27, 56, 52])
}
upper_ranges = {
    "red": np.array([179, 255, 236]),
    "green": np.array([90, 255, 255])
}

# Initialize ArUco marker dictionary
marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
param_markers = cv.aruco.DetectorParameters()

###################################################################################
#Functions definition
###################################################################################

Objects_present = False

def find_marker_size_in_pixels(marker_corners):
    if len(marker_corners) == 0:
        return None
    first_marker = marker_corners[0][0]
    top_left, top_right = first_marker[0], first_marker[1]
    return np.linalg.norm(top_left - top_right)

def warp_to_marker_plane(frame, marker_corners, marker_size, plane_width, plane_height):
    if len(marker_corners) == 0:
        return None, None
    first_marker = marker_corners[0][0]
    marker_plane_points = np.array([
        [plane_width / 2 - marker_size / 2, plane_height / 2 - marker_size / 2],
        [plane_width / 2 + marker_size / 2, plane_height / 2 - marker_size / 2],
        [plane_width / 2 + marker_size / 2, plane_height / 2 + marker_size / 2],
        [plane_width / 2 - marker_size / 2, plane_height / 2 + marker_size / 2]
    ], dtype="float32")
    h, _ = cv.findHomography(first_marker, marker_plane_points)
    return cv.warpPerspective(frame, h, (plane_width, plane_height)), h

def getOrientation(pts, img):
    # Construct a buffer for PCA analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean=np.empty((0)))

    # Calculate the angle of orientation
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

    return angle

def process_contours(warped_frame, color_name, lower_range, upper_range, counter, contour_details, marker_corners,PLANE_WIDTH, PLANE_HEIGHT):
    hsv = cv.cvtColor(warped_frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_range, upper_range)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if color_name =="red":
        contour_color_set = (0,0,255)
    else:
        contour_color_set = (0,255,0)
        
    

    for c in cnts:
        if cv.contourArea(c) > 400:
            M = cv.moments(c)
            if M["m00"] != 0:
                cv.drawContours(warped_frame, c, -1, contour_color_set, 3)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                angle = getOrientation(c, warped_frame)
                angle_deg = -int(np.rad2deg(angle)) - 90

                # Calculate distances
                scale_mm_per_pixel = MARKER_SIZE_MM / find_marker_size_in_pixels(marker_corners)
                distance_x_mm = (cX - PLANE_WIDTH / 2) * scale_mm_per_pixel*1.25
                distance_y_mm = (cY - PLANE_HEIGHT / 2) * scale_mm_per_pixel*1.1

                detail = f"C{counter}:{color_name}, X: {distance_x_mm:.2f}mm, Y: {distance_y_mm:.2f}mm, Angle: {angle_deg}deg"
                contour_details.append(detail)
                counter += 1
    return counter


def get_coords_cam():       

    cap = cv.VideoCapture(0,cv.CAP_DSHOW)
    start_time = time.time()
    duration = 3
    x_coord_C1 = []
    y_coord_C1 = []
    sum_x = 0
    sum_y = 0
    count = 0
    color_of_c1 = ""
    contour_data = {} 

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break

        contour_details = []
        counter = 1
        
        undistorted_frame = cv.undistort(frame, cam_mat, dist_coef, None)
        gray_frame = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, _ = cv.aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)

        if marker_corners:
            PLANE_WIDTH, PLANE_HEIGHT = 500, 500
            warped_frame, _ = warp_to_marker_plane(undistorted_frame, marker_corners, MARKER_SIZE_MM, PLANE_WIDTH, PLANE_HEIGHT)

            if warped_frame is not None:
                for color in lower_ranges:
                    counter = process_contours(warped_frame, color, lower_ranges[color], upper_ranges[color], counter, contour_details, marker_corners,PLANE_WIDTH, PLANE_HEIGHT)

                # Display all contour details
                for i, detail in enumerate(contour_details):
                    cv.putText(warped_frame, detail, (10, 20 + 20 * i), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv.imshow("Warped Frame with Contours", warped_frame)
                cv.waitKey(500)

        else:
            cv.imshow("Frame", frame)               
        
        for detail in contour_details:
            # Extract contour ID, color, X, and Y values
            parts = detail.split(',')  
            contour_id = parts[0].split(':')[0].strip()  # Extract contour ID
            color = parts[0].split(':')[1].strip()  # Extract color
            x = float(parts[1].split(':')[1].replace('mm', '').strip())  # Extract and convert X
            y = float(parts[2].split(':')[1].replace('mm', '').strip())  # Extract and convert Y
            ang = float(parts[3].split(':')[1].replace('deg', '').strip())

            # Initialize dictionary for new contours
            if contour_id not in contour_data:
                contour_data[contour_id] = {'sum_x': 0, 'sum_y': 0, 'count': 0,'sum_ang': 0,'color': color}

            # Update sums and count
            contour_data[contour_id]['sum_x'] += x
            contour_data[contour_id]['sum_y'] += y
            contour_data[contour_id]['sum_ang'] += ang
            contour_data[contour_id]['count'] += 1   
        
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        

    cap.release()
    cv.destroyAllWindows()  

    # Calculate and print averages for each contour
    
    output_array = []
    
    for contour_id, data in contour_data.items():
        if data['count'] > 0:
            avg_x = data['sum_x'] / data['count']
            avg_y = data['sum_y'] / data['count']
            avg_ang = data['sum_ang'] / data['count']
            color = data['color']
            output_array.append((contour_id, avg_x, avg_y,avg_ang, color))

    return output_array   # ContourID

######################################################################
#### Arduino Serial Communication Setup
######################################################################
import serial
import time
import re

# Setup the serial connection
ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)  # Wait for the connection to establish

# Variables to keep track of the most recent target positions
current_target_x = None
current_target_y = None
current_target_z = None

def send_gcode_command(command):
    global current_target_x, current_target_y, current_target_z

    # Update the target positions based on the command
    match = re.search(r'X(-?\d+\.?\d*)', command)
    if match:
        current_target_x = float(match.group(1))
    match = re.search(r'Y(-?\d+\.?\d*)', command)
    if match:
        current_target_y = float(match.group(1))
    match = re.search(r'Z(-?\d+\.?\d*)', command)
    if match:
        current_target_z = float(match.group(1))

    # Send the command
    ser.write((command + '\n').encode())
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line == "ok":
            break
        elif line:
            print("Received:", line)

    if 'G0' in command or 'G1' in command:  # Check position for G0/G1 commands
        print("Waiting for position to be reached...")
        wait_until_position_reached()

    if 'Servo' in command:
        time.sleep(2)

def get_current_position():
    ser.write(b'?\n')
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith('<'):
            return line
        time.sleep(0.1)

def wait_until_position_reached():
    global current_target_x, current_target_y, current_target_z
    while True:
        response = get_current_position()
        match = re.search(r'MPos:(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)', response)
        if match:
            current_x, current_y, current_z = map(float, match.groups())
            if ((current_target_x is None or abs(current_x - current_target_x) < 0.1) and
                (current_target_y is None or abs(current_y - current_target_y) < 0.1) and
                (current_target_z is None or abs(current_z - current_target_z) < 0.1)):
                return
        time.sleep(0.5)


######################################################################
#### Code starts
######################################################################


global green_home_counter,red_home_counter

green_home_counter=0 #Count number of green boxes at home
red_home_counter=0 #Count number of red boxes at home
green_home_position = "G0 X55 Y-25" # Green stacking location
red_home_position = "G0 X55 Y50" # Red stacking location
home_location = "G0 Z35 X110 Y-40" # Gripper Home Location

###########################
# MAIN Loop (for recursion)
###########################

def main_loop():
    global object_count, warning_flag, x_int, y_int, col_int
    global red_home_counter, green_home_counter 

    Detect_array = np.array(get_coords_cam())
    print(Detect_array)

    object_count = 0
    x_int = []
    y_int = []
    col_int = []

    for i in range(Detect_array.shape[0]):
        if float(Detect_array[i, 1]) < 0:
            x_int.append(float(Detect_array[i, 1]))
            y_int.append(float(Detect_array[i, 2]))
            col_int.append(Detect_array[i, 4])
            object_count += 1

    print(object_count)
    if object_count > 0:
        print(x_int[0])
        print(y_int[0])
        print(col_int[0])

    warning_flag = 0

    if object_count > 0:
        if x_int[0] < -115 or x_int[0] > 110 or y_int[0] > 85 or y_int[0] < -27:
            warning_flag = 1
            ser.write(("LCD:Alert: Object beyond workspace" + '\n').encode())
            print("Object beyond workspace")

    if object_count == 0:
        str1 = "LCD:No objects in WS"
        ser.write((str1 + '\n').encode())

    while object_count > 0 and warning_flag == 0:
        str1 = f"LCD:{object_count} objects in WS. Picking up {col_int[0]} object"
        ser.write((str1 + '\n').encode())

        if col_int[0]=="red":
            print("Hoorah, Red color")
            color_home=red_home_position 
            z_drop_val = 1+red_home_counter*18
            red_home_counter+=1

        else:
            print("Hoorah, Green color")
            color_home=green_home_position
            z_drop_val = 1+green_home_counter*18
            green_home_counter+=1
            
                        
        commands = [    
        home_location, #"G0 Z35 X110 Y0", 
        "Servo: Open",
        home_location, 
        f"G0 X{x_int[0]} Y{y_int[0]}", # Target co-ordinates from CV
        "G0 Z0", # Reach Z to plane
        "Servo: Close",  # Close clamp
        "G0 Z35", # Lift object
        color_home, # Return Object to Green Home
        f"G0 Z{z_drop_val}", # Lower clamp to drop plane
        "Servo: Open", #Open Clamp
        "G0 Z35", 
        home_location,
        "Servo: Close"
        ]

        # Send the commands one by one
        for cmd in commands:
            print("Sending command:", cmd)
            send_gcode_command(cmd)

        # ... (rest of your commands and logic)

        # After completing the commands, check for new objects again
        main_loop()  # Recursive call to recheck the conditions and continue if needed

# Initial call to start the process
main_loop()
ser.close() 






     
    
    
    







