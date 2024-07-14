#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::dnn;

struct User {
    string name;
    int age;
};

int faceTracking(const std::vector<string>& userInfo) {
    User user1;
    VideoCapture cap(0); // ��������� ����������� �� ���������

    // ���� �� ������� �������, ������� �� ���������
    if (!cap.isOpened()) {
        cout << "Can't open the video camera!" << endl;
        cin.get(); // ����, ���� ����� ������ ����� �������
        return -1;
    }

    // ����������� ������ � ������ �������
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    double width = cap.get(CAP_PROP_FRAME_WIDTH);
    double height = cap.get(CAP_PROP_FRAME_HEIGHT);
    cout << "---------------------------------------------" << endl;
    cout << "Video resolution: " << width << " x " << height << endl;
    cout << "---------------------------------------------" << endl;

    // ������� ���� � ������ "My Camera Feed"
    string windowName = "My Camera Feed";
    namedWindow(windowName);

    // �������� ������ ���������
    Net net = readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000_fp16.caffemodel");
    if (net.empty()) {
        cerr << "Failed to load network" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        bool bSuccess = cap.read(frame); // ��������� ����� ���� �� �����

        // ��������� ����, ���� �� ������� ��������� ����
        if (!bSuccess) {
            cout << "ERROR: Video camera is not open!" << endl;
            cin.get(); // ����, ���� ����� ������ ����� �������
            break;
        }

        // ���������� ����������� ��� ���������
        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
        net.setInput(blob);

        // ��������� ����������� �������������
        Mat detections = net.forward();
        if (detections.empty()) {
            cerr << "Failed to get network output" << endl;
            continue;
        }

        Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        // ��������� �����������
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.5) {
                //��������� ���������� ��������������, ��������� confidence � ������ �����.
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 255), 4);

                // ���������� ����� ����� � ��������������� ����
                int baseline = 0;
                for (size_t j = 0; j < userInfo.size(); j++) {
                    Size textSize = getTextSize(userInfo[j], FONT_HERSHEY_COMPLEX, 0.7, 1, &baseline);
                    Point textOrg(x1, y1 - (j * textSize.height + 5));
                    putText(frame, userInfo[j], textOrg, FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 255, 255), 1);

                }
            }
        }

        imshow(windowName, frame);

        // ���� 10 ��, ���� �� ����� ������ �����-���� �������
        // ���� ������ ������� 'Esc', ������� �� �����
        // ���� ������ ����� ������ �������, ���������� ����
        // ���� � ������� 10 �� �� ���� ������ �� ����� �������, ���������� ����
        if (waitKey(33) == 27) {
            cout << "---------------------------------------------" << endl;
            cout << "The user pressed the Esc. Stopping the video" << endl;
            cout << "---------------------------------------------" << endl;
            break;
        }
    }
    return 0;
}

void writeUserToFile(const User& user, const string& file) {
    ofstream out(file, ios::app);

    if (!out.is_open()) {
        cout << "ERROR: File is not open!" << endl;
    }

    out << "Name: " << user.name << endl;
    out << "Age: " << user.age << endl;

    out.close();
}

std::vector<string> readUserInfo(const string& file) {
    std::vector<string> userInfo;
    ifstream in(file);
    string line;
    if (in.is_open()) {
        while (getline(in, line)) {
            userInfo.push_back(line);
        }
        in.close();
    }
    else {
        cout << "ERROR: Can't open the file!" << endl;
    }
    return userInfo;
}

void clearFile(const string& file) {
    ofstream out(file, ios::trunc);
    if (!out.is_open()) {
        cout << "ERROR: File is not open!" << endl;
    }
    out.close();
}

int main(int argc, char* argv[]) {
    int num;
    User user;
    cout << "How many users are there? ";
    cin >> num;
    for (int i = 0; i < num; i++) {
        cout << "Enter the user name: ";
        cin >> user.name;
        cout << "Enter the age of the user: ";
        cin >> user.age;
        cout << endl;
        writeUserToFile(user, "users.txt");
    }

    std::vector<string> userInfo = readUserInfo("users.txt");
    faceTracking(userInfo);
    clearFile("users.txt");

    return 0;
}
