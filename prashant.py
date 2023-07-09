import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Start capturing video input
cap = cv2.VideoCapture(0)
cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('MediaPipe Pose', 1440, 960)

current_exercise = "cobra_pose"
squat_threshold = 90

# calculate the angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    pushup_threshold = 0.2  # Threshold for push-up detection
    is_pushup_down = False
    pushup_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        # Convert the BGR image to RGB
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        frame.flags.writeable = False

        # Perform pose estimation
        results = pose.process(frame)

        # Draw the pose annotations on the image
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if current_exercise == "squat":

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                point_23 = [landmarks[23].x, landmarks[23].y]
                point_25 = [landmarks[25].x, landmarks[25].y]
                point_27 = [landmarks[27].x, landmarks[27].y]

                angle = calculate_angle(point_23, point_25, point_27)

                if angle < squat_threshold:
                    if not is_squatting:
                        is_squatting = True
                else:
                    if is_squatting: 
                        print(angle)
                        
                        is_squatting = False
                        squat_completed = True

            if squat_completed:
                print("Squat Completed!")
                squat_completed = False

        if current_exercise == "pushup":
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose_coords = landmarks[mp_pose.PoseLandmark.NOSE]
                wrist_coords = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                # Check the vertical movement of the nose or wrist
                if (nose_coords.y < wrist_coords.y and not is_pushup_down) or (
                        nose_coords.y > wrist_coords.y and is_pushup_down):
                    is_pushup_down = not is_pushup_down

                    # If nose or wrist goes up and push-up was down, count one push-up
                    if not is_pushup_down:
                        pushup_count += 1
                        print("Push-up completed! Count:", pushup_count)
            # Reset push-up completed flag if the person is not in the push-up position
            else:
                is_pushup_down = False
        if current_exercise == "tree_pose":
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

            point_23 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            point_25 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            point_27 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            point_13 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            point_15 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]


            angle = calculate_angle(point_23, point_25, point_27)
            anglez = calculate_angle(point_11, point_13, point_15)
            #print("Angle:", angle)
            threshold_angle = 85
            if angle <= threshold_angle and anglez >= 150:
                print("Tree pose completed!")
        
        def is_horizontal(point_11, point_23, threshold):
            return abs(point_11[1] - point_23[1]) < threshold

        if current_exercise == "cobra_pose":
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

            point_0 = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
            point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            point_23 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            point_25 = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            point_27 = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

            if is_horizontal(point_11, point_23, threshold=0.2):
                #print("User is in a horizontal position!")
                angle_back_bend = calculate_angle(point_11, point_23, point_27)
                print("Back bend Angle 11-23-25:", angle_back_bend)

                threshold_angle = 170  # Adjust as needed
                #if angle_back_bend >= threshold_angle:
                    #print("Cobra Pose complete!")


        cv2.imshow('MediaPipe Pose', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
