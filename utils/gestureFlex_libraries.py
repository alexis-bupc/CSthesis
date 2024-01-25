import csv
import copy
import argparse
import itertools
import math
import pyautogui
import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import mediapipe as mp
import time

from utils.functions import *
from PIL import Image
from io import BytesIO
from utils import CvFpsCalc
from collections import Counter,deque
from model import KeyPointClassifier,PointHistoryClassifier