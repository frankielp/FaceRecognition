# Flask Website Timekeeping using Face Recognition (MTCNN+Facenet)

## Overview
This Flask website is a timekeeping application that utilizes face recognition techniques, specifically MTCNN (Multi-task Cascaded Convolutional Networks) and Facenet, to perform face detection and identification. It allows users to upload images or capture images using their webcam, and the system will recognize and identify the faces present in the images.

## Prerequisites
Before running this application, make sure you have the following prerequisites:
- Python 3.6 or later
- MTCNN and Facenet libraries installed
- Webcam or images to upload

## Installation & Usage
To install and run the application, follow these steps:

```bash
git clone https://github.com/frankielp/facerecognition
bash script.sh
```
To solely start the Flask website, run the following command:
```
python app.py
```
Once the application is running, you can access it in your web browser by navigating to `http://localhost:5000`.

1. Sign Up:
   - Click on the "Capture Images" button.
   - Allow the application to access your webcam.
   - Click the "Capture" button to capture an image.
   - Repeat the capture process for 10 images.
   - Fill the information form
   - Click the "Submit" button.

3. Sign In:
   - After uploading or capturing images, the system will perform face recognition.
   - Wait for the train process to complete and navigate to `signin` tab.
   - The recognized faces will be displayed with their corresponding names or labels and navigate to successful.



## Project Structure
The project structure is as follows:

```
.
├── LICENSE
├── README.md
├── app.py
├── client_secrets.json
├── credentials_store.json
├── model
│   ├── Dataset
│   │   └── Facedata
│   │       ├── processed
│   │       └── raw
│   ├── pretrained
│   ├── requirements.txt
│   ├── src
│   │   ├── align
│   │   ├── align_dataset_mtcnn.py
│   │   ├── calculate_filtering_metrics.py
│   │   ├── classifier.py
│   │   ├── compare.py
│   │   ├── decode_msceleb_dataset.py
│   │   ├── download_and_extract.py
│   │   ├── face_rec.py
│   │   ├── face_rec_flask.py
│   │   ├── facenet_config.py
│   │   ├── freeze_graph.py
│   │   ├── generative
│   │   ├── image_retrieve.py
│   │   ├── lfw.py
│   │   ├── models
│   │   ├── train_softmax.py
│   │   ├── train_tripletloss.py
│   │   └── validate_on_lfw.py
│   └── test
├── static
├── templates
├── script.sh
├── service_account.json
└── token.json

21 directories, 139 files
```

## Acknowledgements
This project utilizes the following libraries and technologies:
- Flask: A web framework for Python
- MTCNN: Multi-task Cascaded Convolutional Networks for face detection
- Facenet: A deep learning model for face recognition

The implementation of MTCNN and Facenet is based on existing open-source repositories and research papers. Special thanks to the contributors and authors of these resources.

Please note that this project is for educational purposes and should not be used for commercial or production purposes without proper consideration of security and privacy concerns.
