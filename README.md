# Face Tracking with OpenCV DNN & Caffe SSD
### Author
**Ranis Vakhitov** — 3rd-year student, Peter the Great St. Petersburg Polytechnic University (SPbPU)

### Real-Time Face Detection and Tracking Using a Caffe-Based SSD Neural Network in C++
---

## About the Project

A real-time face tracking application built in C++ with OpenCV. Faces are detected in each webcam frame using a pre-trained Caffe SSD (Single Shot Detector) model with a ResNet-10 backbone, and bounding boxes are drawn live on the video feed. The application also supports attaching custom user labels (name, age) to detected faces, read from a simple text-based user registry.

**Detection model:** `res10_300x300_ssd_iter_140000_fp16` — a Caffe SSD face detector with a ResNet-10 backbone, loaded via OpenCV's `dnn` module (`readNetFromCaffe`), using `deploy.prototxt` for the network architecture and the `.caffemodel` file for pre-trained weights.

**Pipeline:**
1. Capture frames from the webcam at 1280×720.
2. Preprocess each frame into a blob (`blobFromImage`, 300×300, mean subtraction `(104.0, 177.0, 123.0)`).
3. Run a forward pass through the SSD network to get detections.
4. Filter detections by confidence threshold (`> 0.5`).
5. Draw bounding boxes around detected faces and overlay associated user info as text.
6. Exit the loop on `Esc` key press.

---

## User Registry

Before tracking starts, the program collects user entries (name, age) via console input and writes them to `users.txt`. This file is read back and its contents are displayed as on-screen labels next to detected faces during tracking, then cleared at the end of the session.

```cpp
struct User {
    string name;
    int age;
};
```

- `writeUserToFile` — appends a user entry to the registry file.
- `readUserInfo` — reads all lines from the registry into a vector of strings.
- `clearFile` — truncates the registry file after the session ends.

---

## Face Detection Logic

```cpp
Net net = readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000_fp16.caffemodel");
Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
net.setInput(blob);
Mat detections = net.forward();
```

Each detection row contains `[batch_id, class_id, confidence, x1, y1, x2, y2]`, where coordinates are normalized to `[0, 1]` and are rescaled to the frame's actual width and height before drawing.

---

## Installation & Usage

### Requirements
- C++ compiler with C++11 support or later
- OpenCV (built with the `dnn`, `objdetect`, `imgproc`, and `highgui` modules)
- A connected webcam

### Build
```bash
# Clone the repository
git clone https://github.com/nyassky/FaceRecognition.git
cd FaceRecognition

# Compile (example using g++, adjust OpenCV include/lib paths as needed)
g++ main.cpp -o FaceRecognition `pkg-config --cflags --libs opencv4`
```

### Run
```bash
./FaceRecognition
```

Make sure `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000_fp16.caffemodel` are in the same directory as the executable. On launch, the program will prompt for the number of users and their name/age, then start the webcam feed with live face detection. Press `Esc` to stop.

---

## Project Structure

```
.
├── main.cpp                                      # Application entry point & detection loop
├── deploy.prototxt.txt                           # SSD network architecture definition
├── res10_300x300_ssd_iter_140000_fp16.caffemodel # Pre-trained SSD weights
└── users.txt                                     # Runtime-generated user registry (cleared after each run)
```

---

## Stack

![C++](https://img.shields.io/badge/C%2B%2B-11%2B-00599C)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8)
![Caffe](https://img.shields.io/badge/Model-Caffe%20SSD-orange)
