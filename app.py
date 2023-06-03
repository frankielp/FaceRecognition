from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from imutils.video import VideoStream

from flask import Flask, request, render_template, Response, jsonify
from imutils.video import VideoStream
from mtcnn_facenet.src import facenet_config as facenet

import imutils
import os
import sys
import math
import pickle
from mtcnn_facenet.src.align import detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from pymongo import MongoClient
from datetime import datetime
import pymongo
# from bson.binary import Binary
from PIL import Image
import io
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client import file, client, tools
from httplib2 import Http
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
import glob
from mtcnn_facenet.src import facenet_config
from mtcnn_facenet.src import align_dataset_mtcnn
from mtcnn_facenet.src import classifier

disabled = True

SERVICE_ACCOUNT_FILE = 'service_account.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
DRIVE_API_VERSION = 'v3'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES
)
service = build('drive', DRIVE_API_VERSION, credentials=creds)

# connect to db
uri = "mongodb+srv://npn279:grab2023@cluster0.ek6wvyn.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

# select database Bootcamp
db = client.Bootcamp

# select a collection (table) USER
collection = db.USER
logindata = db.LOGIN_DATA

app = Flask(__name__)

# Initialize the video stream
video = VideoStream(src=0).start()

image_counter = 0

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'mtcnn_facenet/Models/facemodel.pkl'
# VIDEO_PATH = args.path
FACENET_MODEL_PATH = 'mtcnn_facenet/Models/20180402-114759.pb'
@app.route('/')
def index():
    return render_template('signin.html')


def generate_frames():
    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet_config.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = detect_face.create_mtcnn(
                sess, "mtcnn_facenet/src/align")

            people_detected = set()
            person_detected = collections.Counter()

            while (True):
                # success, frame = video.read()
                frame = video.read()

                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            print(bb[i][3]-bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3]-bb[i][1])/frame.shape[0])
                            if (bb[i][3]-bb[i][1])/frame.shape[0] > 0.25:
                                cropped = frame[bb[i][1]:bb[i]
                                                [3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet_config.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(
                                    -1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {
                                    images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(
                                    embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(
                                    predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(
                                    best_name, best_class_probabilities))

                                if best_class_probabilities > 0.4:
                                    cv2.rectangle(
                                        frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                else:
                                    name = "Unknown"

                except:
                    print("Dead")

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Stream the video frames to the web page"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/signin')
def signin():
    """Render the signin.html template"""
    return render_template('signin.html')


@app.route('/signup')
def signup():
    """Render the signup.html template"""
    return render_template('signup.html')


@app.route('/signup_image')
def signup_image():
    """Render the signup_image.html template"""
    return render_template('signup_image.html')


def generate_frames_signup():

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            pnet, rnet, onet = detect_face.create_mtcnn(
                sess, "mtcnn_facenet/src/align")

            people_detected = set()
            person_detected = collections.Counter()

            while (True):
                frame = video.read()

                frame = cv2.flip(frame, 1)

                bounding_boxes, _ = detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            if (bb[i][3]-bb[i][1])/frame.shape[0] > 0.25:
                                cv2.rectangle(
                                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                except:
                    print("Dead")

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed_signup')
def video_feed_signup():
    """Stream the video frames to the web page"""
    return Response(generate_frames_signup(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_signup', methods=['POST'])
def capture_image_signup():
    """Capture the current frame and save it to a file"""
    global image_counter

    if image_counter >= 10:
        return jsonify({'success': False})

    # Read the next video frame
    frame = video.read()
    # if not success:
    #     print("Video stream not available.")
    #     return "Video stream not available."

    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    tf.compat.v1.disable_eager_execution()

    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(
            sess, "mtcnn_facenet/src/align")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_boxes, _ = detect_face.detect_face(
            frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces = bounding_boxes.shape[0]
        if faces > 1:
            cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (255, 255, 255), thickness=1, lineType=2)
        elif faces > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces, 4), dtype=np.int32)
            for i in range(faces):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                if (bb[i][3]-bb[i][1])/frame.shape[0] > 0.25:
                    cv2.rectangle(
                        frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                    cropped_frame = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]

                    data = request.get_json()
                    name = data['name']
                    num_docs = collection.count_documents({})
                    new_id = f"user{num_docs + 1}"
                    print(new_id)
                    print(name)

                    # Create a folder with the name of the user in the faces directory
                    if not os.path.exists("mtcnn_facenet/Dataset"):
                        os.makedirs("mtcnn_facenet/Dataset")
                        if not os.path.exists("mtcnn_facenet/Dataset/Facedata"):
                            os.makedirs("mtcnn_facenet/Dataset/Facedata")
                            if not os.path.exists("mtcnn_facenet/Dataset/Facedata/raw"):
                                os.makedirs(
                                    "mtcnn_facenet/Dataset/Facedata/raw")
                    user_folder = os.path.join(
                        "mtcnn_facenet/Dataset/Facedata/raw", new_id)
                    if not os.path.exists(user_folder):
                        os.mkdir(user_folder)
                    print(user_folder)

                    cv2.imwrite(
                        f"{user_folder}/capture_image{image_counter+1}.png", frame)
                    print(f"Image {image_counter+1} saved successfully.")
                    image_counter += 1

    return jsonify({'success': True})
@app.route('/save_data', methods=['POST'])
def save_data():
    data = request.json
    name = data['name']
    position = data['position']
    password = data['password']
    gender = data['gender']
    today = datetime.today().strftime('%d/%m/%Y')
    num_docs = collection.count_documents({})
    new_id = f"user{num_docs + 1}"
    train_images = os.path.join(
        "mtcnn_facenet", "Dataset", "Facedata", "raw", new_id)

    # Create the folder and print the folder ID
    folder_name = new_id
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=file_metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f'Folder created with ID: {folder_id}')

    # Traverse the directory structure of the local folder and upload each file to Google Drive
    for dirpath, dirnames, filenames in os.walk(train_images):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            media = MediaFileUpload(file_path)
            file = service.files().create(body=file_metadata,
                                          media_body=media, fields='id').execute()
            print(f'File ID: {file.get("id")}')

    # Search for the folder by ID and print its name
    folder = service.files().get(fileId=folder_id, fields='id, name').execute()
    print(f'Folder name: {folder.get("name")}')

    # Search for the folder by name and print its ID
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = service.files().list(
        q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    if items:
        first_image_id = items[0]['id']
        print(f'Folder found with ID: {first_image_id}')
    else:
        print('Folder not found')

    align_dataset_mtcnn.preprocess()
    classifier.train_classify()
    # Save the data to MongoDB
    doc = {"_id": new_id, "name": name, "position": position,
           "password": password, "gender": gender, "type": "user",
           "date_in": today, "train_images": folder_id, "profile_image": first_image_id, }
    collection.insert_one(doc)

    print("re-train model\n")
    return "Data saved to MongoDB"


@app.route('/signin_manual')
def signin_manual():
    """Render the signup_image.html template"""
    return render_template('signin_manual.html')


@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    username = request.json['username']
    password = request.json['password']
    user = collection.find_one({'_id': username, 'password': password})
    if user:
        user_name = user.get('name')
        today = datetime.now()
        day = today.day
        month = today.month
        year = today.year
        login_data = logindata.find_one(
            {'userId': username, 'day': day, 'month': month, 'year': year})

        if login_data:
            time_out = datetime.now().strftime("%H:%M:%S")
            logindata.update_one({'userId': username, 'day': day, 'month': month, 'year': year}, {
                                 '$set': {'time_out': time_out}})
        else:
            time_in = datetime.now().strftime("%H:%M:%S")
            logindata.insert_one({'userId': username, 'name': user_name, 'day': day,
                                 'month': month, 'year': year, 'time_in': time_in, 'time_out': time_in})
        return jsonify({'success': True, 'redirect': '/welcome', 'username': username, 'password': password})
    else:
        return jsonify({'success': False, 'redirect': '/signup'})


@app.route('/welcome')
def welcome():
    """Render the signup_image.html template"""
    username = request.args.get('username')
    password = request.args.get('password')

    user = collection.find_one({'_id': username, 'password': password})
    if user:
        user_name = user.get('name')
        user_position = user.get('position')

    today = datetime.now()
    day = today.day
    month = today.month
    year = today.year
    login_data = logindata.find_one(
        {'userId': username, 'day': day, 'month': month, 'year': year})

    if login_data is not None:
        user_timeout = login_data.get('time_out')
        user_timein = login_data.get('time_in')

    formatted_date = today.strftime('%d/%m/%Y')

    return render_template('success.html', user_name=user_name, user_position=user_position,
                           user_timein=user_timein, user_timeout=user_timeout, date=formatted_date)


if __name__ == '__main__':
    app.run()