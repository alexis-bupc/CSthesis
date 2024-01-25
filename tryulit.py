
import csv
import copy
import argparse
# import itertools
from collections import Counter,deque
import math
import pyautogui
import PySimpleGUI as sg


import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from utils import CvFpsCalc
from model import KeyPointClassifier,PointHistoryClassifier
from utils.functions import draw_landmarks, select_mode, calc_bounding_rect,calc_landmark_list, pre_process_landmark,pre_process_point_history,logging_csv,draw_bounding_rect,draw_info,draw_info_text,draw_point_history



def create_help_panel():
    # Add your help menu layout here
    layout = [
        [sg.Text("Help Panel")],
        # ... (Additional elements for the help menu)
    ]
    return sg.Window('Help', layout, resizable=True, finalize=True)

def create_about_panel():
    # Add your about menu layout here
    layout = [
        [sg.Text("About Panel")],
        # ... (Additional elements for the about menu)
    ]
    return sg.Window('About', layout, resizable=True, finalize=True)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

# Variables
delay_between_predictions = 1
velocity_threshold = 2.0

def get_available_cameras():
    cameras = []
    for i in range(10):  # Check up to 10 cameras
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            cameras.append(f'Camera {i}')
            cap.release()
    return cameras

#For classification of intended and unintended commands
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main_window():
    sg.theme('DarkBlue3')
    layout = [
        [sg.Menu([['Main', ['Exit']],
             ['Select Camera', get_available_cameras()],
             ['Help', ['Help']],
             ['About', ['About']]])],
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Button('Exit', size=(10, 1))]
    ]
    return sg.Window('Hand Gesture Recognition', layout, resizable=True, finalize=True)

def main():
    # Argument parsing #################################################################
    args = get_args()
    prev_hand_position = None
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    window = main_window()
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open('model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    #PySimpleGUI

    selected_camera = 0
    cap = cv.VideoCapture(selected_camera)

    while True:
        fps = cvFpsCalc.get()
        event, values = window.read(timeout=10)

        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        

        if event and event.startswith('Camera'):
            selected_camera = int(event.split()[1])
            cap.release()
            cap = cv.VideoCapture(selected_camera)

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                
                current_hand_position = landmark_list[8]  # Assuming the landmark 8 is the hand position
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if prev_hand_position is not None:
                    hand_velocity = calculate_distance(
                        prev_hand_position, current_hand_position)

                    # Check if the hand is moving or not
                    
                    if hand_velocity > velocity_threshold:
                        None
                    else:
                        if hand_sign_id == 0:  # Full Screen
                            pyautogui.press('f5')
                            time.sleep(delay_between_predictions)
                        if hand_sign_id == 1:  # Next
                            pyautogui.press('right')
                            time.sleep(delay_between_predictions)
                        if hand_sign_id == 2:  # Previuos
                            pyautogui.press('left')
                            time.sleep(delay_between_predictions)
                        if hand_sign_id == 3:  # No Gesture
                            None
                        else:
                            point_history.append([0, 0])

                prev_hand_position = current_hand_position
                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        # cv.imshow('Hand Gesture Recognition', debug_image)
        imgbytes = cv.imencode('.png', debug_image)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    main()
