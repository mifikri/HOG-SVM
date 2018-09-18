#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

// Set the file to write the resulting opencv hog classifier as YAML file
static string cvHOGFile = "/home/fikri/trainHOG2/genfiles/hog64.yaml";

/*static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }

    printf("%d \t", found_filtered.size());
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
        printf("%d,%d \n\t\t", r.x+(r.width/2), r.y+(r.height/2));
    }

    printf("\t");
    cout << "Jumlah Deteksi: " << found_filtered.size() << " | ";
}
*/
static void showDetections(const vector<Point>& found, Mat& imageData) {
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Point r = found[i];
        // Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
        rectangle(imageData, Rect(r.x-16, r.y-32, 32, 64), Scalar(64, 255, 64), 3);
    }
}

static void showDetections(const vector<Rect>& found, Mat& imageData) {
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i) {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++) {
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
}

static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageOri, Mat& imageData) {
    vector<Rect> found;
    Size padding(Size(8, 8));
    Size winStride(Size(8, 8));
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding, SCALE, 2);
    showDetections(found, imageOri);
}

int main(int argc, char** argv) {
    HOGDescriptor hog; // Use standard parameters here

    //processing time check
    unsigned long AAtime = 0, BBtime = 0;

    //frame number
    unsigned int frameNum = 0;

    //set the hitThreshold
    //const double hitThreshold = -1.00565; //-0.312493;
    const double hitThreshold = 0.415;

    //load the saved result opencv hog classifier
    hog.load(cvHOGFile);

    printf("Percobaan deteksi bola menggunakan HOG dengan kamera\n");
 
    //Size size(300,200);
    //VideoCapture cap("/home/fikri/video/Webcam/1.mp4"); // open the video
    //VideoCapture cap(1);
    //VideoCapture cap("/home/fikri/video/Webcam/2.mp4");
    if(!cap.isOpened()) { // check if we succeeded
        printf("Error opening video!\n");
        return EXIT_FAILURE;
    }

    Size size(560,315);
    namedWindow("HOG ball detection", CV_WINDOW_FREERATIO);

    Mat test, testGray;
    while ((cvWaitKey(33) & 255) != 27) {
        cap >> test; // get a new frame from video file
        resize(test,test,size);
        //blur(test,test,Size(7,7));
        //Mat matRotation = getRotationMatrix2D(Point(test.cols/2,test.rows/2),180,1);
        //warpAffine(test,test, matRotation,test.size());
        AAtime = getTickCount();
        printf("%d \t", frameNum);
        cvtColor(test, testGray, CV_BGR2GRAY);
        //imshow("Gray", testGray);
        detectTest(hog, hitThreshold, test, testGray);
        //time check
        BBtime = getTickCount();
        double s_time = (BBtime - AAtime) / getTickFrequency();
        double fps_time = 1/s_time;
        printf("%.5lf \t%.5lf \n", s_time, fps_time);
        imshow("HOG ball detection", test);
        frameNum++;
    }
    // </editor-fold>

    return EXIT_SUCCESS;
}
