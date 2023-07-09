import argparse
import os
import struct
import wave
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import pvporcupine
from pvrecorder import PvRecorder
import multiprocessing
from multiprocessing import Process, Manager
import time
import pyttsx3

#image based variables
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pushup_threshold = 0.2  # Threshold for push-up detection
is_pushup_down = False
pushup_count = 0

# Start capturing video input
cap = cv2.VideoCapture(0)
cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('MediaPipe Pose', 1440, 960)
current_exercise = "pushup"
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


def main(current_exercise, calories, count, is_hidden):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--access_key',
        help='AccessKey obtained from Picovoice Console (https://console.picovoice.ai/)',
        required=True)

    parser.add_argument(
        '--keywords',
        nargs='+',
        help='List of default keywords for detection. Available keywords: %s' % ', '.join(
            '%s' % w for w in sorted(pvporcupine.KEYWORDS)),
        choices=sorted(pvporcupine.KEYWORDS),
        metavar='')

    parser.add_argument(
        '--keyword_paths',
        nargs='+',
        help="Absolute paths to keyword model files. If not set it will be populated from `--keywords` argument")

    parser.add_argument(
        '--library_path',
        help='Absolute path to dynamic library. Default: using the library provided by `pvporcupine`')

    parser.add_argument(
        '--model_path',
        help='Absolute path to the file containing model parameters. '
             'Default: using the library provided by `pvporcupine`')

    parser.add_argument(
        '--sensitivities',
        nargs='+',
        help="Sensitivities for detecting keywords. Each value should be a number within [0, 1]. A higher "
             "sensitivity results in fewer misses at the cost of increasing the false alarm rate. If not set 0.5 "
             "will be used.",
        type=float,
        default=None)

    parser.add_argument('--audio_device_index', help='Index of input audio device.', type=int, default=-1)

    parser.add_argument('--output_path', help='Absolute path to recorded audio for debugging.', default=None)

    parser.add_argument('--show_audio_devices', action='store_true')

    args = parser.parse_args()

    if args.show_audio_devices:
        for i, device in enumerate(PvRecorder.get_audio_devices()):
            print('Device %d: %s' % (i, device))
        return

    
    keyword_paths = ["keywords/cal.ppn", "keywords/pushup.ppn", "keywords/tree.ppn", "keywords/hide.ppn", "keywords/open.ppn", "keywords/squat.ppn", "keywords/cobra.ppn"]
    

    if args.sensitivities is None:
        args.sensitivities = [0.5] * len(keyword_paths)

    if len(keyword_paths) != len(args.sensitivities):
        raise ValueError('Number of keywords does not match the number of sensitivities.')

    try:
        porcupine = pvporcupine.create(
            access_key="5fVf05t0LvVl0F2zbnuSUAABmRpi4HucWIMuAiCMxIMlGpFZQOA6SQ==",
            library_path=args.library_path,
            model_path=args.model_path,
            keyword_paths=keyword_paths,
            sensitivities=args.sensitivities)
    except pvporcupine.PorcupineInvalidArgumentError as e:
        print("One or more arguments provided to Porcupine is invalid: ", args)
        print("If all other arguments seem valid, ensure that '%s' is a valid AccessKey" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationError as e:
        print("AccessKey activation error")
        raise e
    except pvporcupine.PorcupineActivationLimitError as e:
        print("AccessKey '%s' has reached it's temporary device limit" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationRefusedError as e:
        print("AccessKey '%s' refused" % args.access_key)
        raise e
    except pvporcupine.PorcupineActivationThrottledError as e:
        print("AccessKey '%s' has been throttled" % args.access_key)
        raise e
    except pvporcupine.PorcupineError as e:
        print("Failed to initialize Porcupine")
        raise e

    keywords = list()
    for x in keyword_paths:
        keyword_phrase_part = os.path.basename(x).replace('.ppn', '').split('_')
        if len(keyword_phrase_part) > 6:
            keywords.append(' '.join(keyword_phrase_part[0:-6]))
        else:
            keywords.append(keyword_phrase_part[0])

    print('Porcupine version: %s' % porcupine.version)

    recorder = PvRecorder(
        device_index=args.audio_device_index,
        frame_length=porcupine.frame_length)
    recorder.start()

    wav_file = None
    if args.output_path is not None:
        wav_file = wave.open(args.output_path, "w")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)

    print('Listening ... (press Ctrl+C to exit)')

    try:
        listen(recorder=recorder, porcupine=porcupine, wav_file=wav_file, keywords=keywords, current_exercise=current_exercise, calories=calories, count=count, is_hidden=is_hidden)
    except KeyboardInterrupt:
        print('Stopping ...')
    finally:
        recorder.delete()
        porcupine.delete()
        if wav_file is not None:
            wav_file.close()

def listen(recorder, porcupine, wav_file, keywords, current_exercise, calories, count, is_hidden):
    while True:
        pcm = recorder.read()
        result = porcupine.process(pcm)

        if wav_file is not None:
            wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))

        if result >= 0:

            # Once the keyword is detected run prashant code
            if keywords[result] == "pushup":
                count.value = 0
                current_exercise.value = "pushup"
            elif keywords[result] == "squat":
                count.value = 0
                current_exercise.value = "squat"
            elif keywords[result] == "tree":
                count.value = 0
                current_exercise.value = "tree_pose"
            elif keywords[result] == "open":
                is_hidden.value = False
            elif keywords[result] == "hide":
                is_hidden.value = True
            elif keywords[result] == "cobra":
                count.value = 0
                current_exercise.value = "cobra_pose"
            elif keywords[result] == "cal":
                    
                engine = pyttsx3.init()

                engine.say("You currently burnt " + str(round(calories.value, 2)) + " calories")
                engine.runAndWait()
                time.sleep(0.5)
                engine.stop()
                #engine.disconnect()

            #print('[%s] Detected %s' % (str(=datetime.now()), keywords[result]))


if __name__ == '__main__':
    with Manager() as manager:
        current_exercise = manager.Value('s', 'pushup')  # 's' specifies a string type
        calories = manager.Value('d', 0.0)  # 'i' specifies an integer type
        count = manager.Value('i', 0)
        is_hidden = manager.Value('b', False)
        is_squatting = False
        squat_completed = False

        p1 = Process(target=main, args=(current_exercise, calories, count, is_hidden))
        p1.start()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pushup_threshold = 0.2  # Threshold for push-up detection
            is_pushup_down = False
            last_time = 0
            
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
                
                if not is_hidden.value:

                    if current_exercise.value == "squat" or current_exercise.value == "pushup":
                        cv2.putText(frame, "Count: " + str(count.value), (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        minutes, seconds = divmod(count.value, 60)
                        time_str = "{:02d}:{:02d}".format(int(minutes), int(seconds))
                        cv2.putText(frame, "Time Elapsed: " + time_str, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


                    cv2.putText(frame, "Current Exercise: " + current_exercise.value, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "Calories " + str(round(calories.value, 2)), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if current_exercise.value == "squat":

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
                        count.value += 1
                        squat_completed = False
                        calories.value += 0.32

                if current_exercise.value == "pushup":
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
                                count.value += 1
                                calories.value += 0.325

                    # Reset push-up completed flag if the person is not in the push-up position
                    else:
                        is_pushup_down = False
                if current_exercise.value == "tree_pose":
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
                        if time.time() - last_time >= 1:
                            print("Tree pose completed!")
                            last_time = time.time()
                            count.value += 1
                            calories.value += 0.05

                
                def is_horizontal(point_11, point_23, threshold):
                    return abs(point_11[1] - point_23[1]) < threshold

                if current_exercise.value == "cobra_pose":
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        cobra_pose_completed = False


                        shoulder_coords = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                        hip_coords = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]


                        if is_horizontal(shoulder_coords, hip_coords, threshold=0.1):
                            print("User is in a horizontal position!")


                        point_0 = [landmarks[mp_pose.PoseLandmark.NOSE].x,
                                landmarks[mp_pose.PoseLandmark.NOSE].y]
                        point_11 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                        point_23 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]


                        angle = calculate_angle(point_0, point_11, point_23)
                        print("Angle 0-11-23:", angle)


                        threshold_angle = 158
                        if angle >= threshold_angle and is_horizontal(shoulder_coords, hip_coords, threshold=0.3):
                            if time.time() - last_time >= 1:
                                last_time = time.time()
                                count.value += 1
                                calories.value += 0.07
                                
                            #if not cobra_pose_completed:
                                #cobra_pose_completed = True
                                #print("Cobra Pose completed!")


                cv2.imshow('MediaPipe Pose', frame)

                # Break the loop on 'q' key press
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        # Release the webcam and close OpenCV window
        cap.release()
        cv2.destroyAllWindows()

