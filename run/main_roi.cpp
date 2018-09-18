#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define LEBAR   32
#define TINGGI  32

#define JARAK       100
#define CAHAYA      75
#define SCALE       1.2

using namespace std;
using namespace cv;

// Set the file to write the features to
static string featuresFile = "/home/fikri/Alfarobi/putih/trainHOG-master/genfiles/features.dat";
// Set the file to write the SVM model to
static string svmModelFile = "/home/fikri/Alfarobi/putih/trainHOG-master/genfiles/svmlightmodel.dat";
// Set the file to write the resulting detecting descriptor vector to
static string descriptorVectorFile = "/home/fikri/Alfarobi/putih/trainHOG-master/genfiles/descriptorvector.dat";
// Set the file to write the resulting opencv hog classifier as YAML file
//static string cvHOGFile = "/home/fikri/Alfarobi/putih/hog/cvHOGClassifier64x64.yaml";
static string cvHOGFile32 = "/home/fikri/Alfarobi/putih/trainHOG-master/genfiles/cvHOGClassifier32.yaml";

Point2d bola;
Mat img_goall, img_ROI, imgField;
Point2d ballPos, last_ballPost;
Mat frame, frame2, frame_gray, frame_gray2, frame_gray1, frame_HSV, threshold_output, canny_output;

int thresh = 100;
int iLowH = 29;  int iHighH = 49;
int iLowS = 19;  int iHighS = 151;
int iLowV = 32; int iHighV = 200;

HOGDescriptor hog; // Use standard parameters here

//set the hitThreshold
    const double hitThreshold = -1.0565; // 32x32
//const double hitThreshold = -0.05565;  // 64x64
//  const double hitThreshold = -0.312493;

float computeMoment(Mat img)
{
    float image_moment;
    Moments moment = moments(img);
    image_moment = moment.m00;
    return image_moment;
}

Point2d computeCentroid(Mat img)
{
    Point2d centroid;
    centroid.x = moments(img).m10/moments(img).m00;
    centroid.y = moments(img).m01/moments(img).m00;
    cout << "adada"<<centroid.x << "ssss" << centroid.y <<endl;
    return centroid;
}

Mat ImageSegmentation(Mat input)
{
    static short blur_count=0;
    Mat output;

    cvtColor(input, output, CV_RGB2HSV);

    namedWindow("Control Green");
    createTrackbar("LowH      ", "Control Green", &iLowH, 179);
    createTrackbar("HighH     ", "Control Green", &iHighH, 179);
    createTrackbar("LowS      ", "Control Green", &iLowS, 255);
    createTrackbar("HighS     ", "Control Green", &iHighS, 255);
    createTrackbar("LowV      ", "Control Green", &iLowV, 255);
    createTrackbar("HighV     ", "Control Green", &iHighV, 255);

    inRange(output,Scalar(iLowH, iLowS,iLowV ), Scalar(iHighH, iHighS, iHighV),output); //putih - abu2

    if (blur_count <= 2) {
        blur(output, output, Size(3,3));
        blur_count++;
    }
    else {
        medianBlur(output, output, 3);
        blur_count = 0;
    }
    Mat element = getStructuringElement( 2, Size( 3, 3 ), Point( 1, 1 ) );
    morphologyEx( output, output, 2, element );
    return output;
}

//static void showDetections(const vector<Rect>& found, Mat& imageData) {
Point2d showDetections(const vector<Rect>& found, vector<Rect>& found_filtered, Mat& imageData) {

    static Point2d pos, last_pos;
    pos = Point2d(-1,-1);
    Point2d min, max;
    Mat imageField, imageROI;
    bool ballPosState;
    float momenOutputs, momentROI, rowsROI, colsROI, ROIArea;

    //    vector<Rect> found_filtered;
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
    i = 0;
//    for (i = 0; i < found_filtered.size(); i++) {
    if(found_filtered.size()>0){
        Rect r = found_filtered[i];
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);

        cout <<"rtl : "<<r.tl()<<" rbr: "<<r.br()<<endl;
        cout <<"r : "<<r<<endl;
        double radius = (r.br().x - r.tl().x)/2;
//        bola.x = r.x+(r.width/2);
//        bola.y = r.y+(r.height/2);
        bola.x = radius +r.x;
        bola.y = radius +r.y;

        cout<<"bola: "<<bola<<" rad : "<<radius<<endl;

//        if(bola == Point2d(-1,-1))
//    }
    printf("\t");
    //cout << "Jumlah Deteksi: " << found_filtered.size() << " | ";

    if(bola != Point2d(-1,-1)) {
        min.x = bola.x - 2.5*radius; max.x = bola.x + 2.5*radius;
        min.y = bola.y - 1.5*radius; max.y = bola.y + 1.5*radius;
        //min=r.tl();max=r.br();

        if(min.x < 0) min.x = 0;
        if(min.y < 0) min.y = 0;
        if(max.x > 320 )  max.x = 320;
        if(max.y > 240 ) max.y = 240;

    }
    else
    {
        min.x = -bola.x;
        max.x = -bola.x+1;
        min.y = -bola.y;
        max.y = -bola.y+1;
    }

//    imageROI = imageData.colRange(r.tl().x,r.br().x).rowRange(r.tl().y,r.br().y);
//    imageROI = imageData.colRange(r.x-r.width/2,r.x+r.width/2).rowRange(r.y-r.height/2,r.y+r.height/2);
    imageROI = frame.colRange(min.x,max.x).rowRange(min.y,max.y);

     cout << "min : " << min<< "max : " << max << endl;
    imshow("imageROI",imageROI);
//    rectangle(imageData, Point(r.tl().x,r.br().x),Point(r.tl().y,r.br().y), Scalar(255,0,0),1);
    rectangle(imageData, min,max, Scalar(255,0,0),1);
//    rectangle(imageData, Point(100,100),Point(255,255), Scalar(0,255,0),1);

    //rectangle(imageData, min,max, Scalar(255,0,0),1);
   // cout << "min : " << min << "max : " << max << endl;
    imageField = ImageSegmentation(imageROI);
    imshow("imageROISegmented",imageField);

    momentROI = computeMoment(imageField);
    //cout << "momentROI = " << momentROI << endl;
   // int x= momentROI.m10;
    //int y=momentROI.m01;
    ROIArea = 255*imageROI.rows*imageROI.cols;
    //cout<<"ROI Area : "<<ROIArea<<endl;


    float percentMoment = momentROI/(ROIArea);
    cout << "percentMoment = " << percentMoment << endl;

    if(percentMoment < 0.2 )
    {
        ballPosState = false;
        pos = Point2d(-1,-1);
    }
    else
    {
        ballPosState = true;
        pos = bola;
        //pos.x = last_pos.x + 0.3 * (centroidOutputs.x - last_pos.x); // low pass filter
        //pos.y = last_pos.y + 0.2 * (centroidOutputs.y - last_pos.y);
        last_pos = pos;
        cout<<"pos : "<<pos<<endl;
        cout<<"last pos : "<<last_pos<<endl;


    }}
    else
    {pos = Point2d(-1,-1);}

    return pos;
}

Point2d detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageOri, Mat& imageData) {
    vector<Rect> found;
    vector<Rect> found_filtered;
    cvtColor(imageOri, imageData, CV_BGR2GRAY);


    Size padding(Size(8, 8));
    Size winStride(Size(8, 8));
    hog.detectMultiScale(imageData, found, hitThreshold, winStride, padding, SCALE, 2);
    showDetections(found, found_filtered, imageOri);

}

/*Point2d detectBall(Mat inputt)
{
    Mat inputt_gray;
    imshow("Gray", inputt_gray);
    bola = detectTest(hog, hitThreshold, inputt, inputt_gray);
    cout<<"bola : "<<bola<<endl;

    return bola;
}


Mat findROI(Mat inputz, Point2d pos_nofil, size_t i)
{
    Point2d min, max;

    if(pos_nofil != Point2d(-1,-1)) {
        min.x = pos_nofil.x - 5*found_filtered[i].width/2; max.x = pos_nofil.x + 5*found_filtered[i].width/2;
        min.y = pos_nofil.y - 4*found_filtered[i].height/2; max.y = pos_nofil.y + 4*found_filtered[i].height/2;

        if(min.x < 0) min.x = 0;
        if(min.y < 0) min.y = 0;
        if(max.x > found_filtered[i].width)  max.x = found_filtered[i].width;
        if(max.y > found_filtered[i].height) max.y = found_filtered[i].height;
    }
    else
    {
        min.x = -pos_nofil.x;
        max.x = -pos_nofil.x+1;
        min.y = -pos_nofil.y;
        max.y = -pos_nofil.y+1;
    }
    img_ROI = inputz.colRange(min.x,max.x).rowRange(min.y,max.y);

    return img_ROI;
}*/


/*Point2d cv_getPos(const Mat input, Mat output, Point2d last_pos)
{
    static Point2d pos;
    Point2d centre;
    Point2d bolaa;
    pos = Point2d(-1,-1);
    bool ballPosState;
    Mat imageROI,imageField;
    float momenOutputs, momentROI, rowsROI, colsROI, ROIArea;
    size_t i;
    Mat outputs;

    bolaa = detectBall(input);
    cout<<"bolaa : "<<bolaa<<endl;

    imageROI = findROI(input, bolaa, 0);
    imshow("imageROI",imageROI);

    imageField = ImageSegmentation(imageROI);
    imshow("imageROISegmented",imageField);

    ROIArea = 255*imageROI.rows*imageROI.cols;
    cout<<"ROI Area : "<<ROIArea<<endl;

    momentROI = computeMoment(imageField);
    cout << "momentROI = " << momentROI << endl;

    float percentMoment = momentROI/(ROIArea);
    cout << "percentMoment = " << percentMoment << endl;

    if(percentMoment < 0.2 )
    {
        ballPosState = false;
        pos = Point2d(-1,-1);
    }
    else
    {
        ballPosState = true;
        pos = bolaa;
        //pos.x = last_pos.x + 0.3 * (centroidOutputs.x - last_pos.x); // low pass filter
        //pos.y = last_pos.y + 0.2 * (centroidOutputs.y - last_pos.y);
        last_pos = pos;
        cout<<"pos : "<<pos<<endl;
        cout<<"last pos : "<<last_pos<<endl;
    }
    return pos;
}
*/

int main(int argc, char** argv) {

    //processing time check
    unsigned long AAtime = 0, BBtime = 0;

    //frame number
    unsigned int frameNum = 0;

    //load the saved result opencv hog classifier
  //  hog.save(cvHOGFile);
    hog.load(cvHOGFile32);

    printf("Percobaan deteksi bola menggunakan HOG dengan kamera\n");
    printf("Deskriptor      : %d,%d\n", TINGGI, LEBAR);
    printf("Threshold       : %.5lf\n", hitThreshold);
    printf("Jarak           : %d cm\n", JARAK);
    printf("Pencahayaan     : %d lux\n", CAHAYA);
    printf("Scale           : %.1lf\n", SCALE);
    printf("Obstacle        : 2 robot\n\n");

    //VideoCapture cap("/home/fikri/Alfarobi/putih/trainHOG-master/video/Webcam/2016-06-20-162724.mp4");
    //VideoCapture cap(1);
    VideoCapture cap("/home/fikri/Alfarobi/putih/trainHOG-master/video/webcam2.mp4");
    if(!cap.isOpened()) { // check if we succeeded
        printf("Error opening video!\n");
        return EXIT_FAILURE;
    }
//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    printf("Frame ke- \tTotal Deteksi \tTitik Tengah \tWaktu Deteksi(sec) \tFPS \n");

    while ((cvWaitKey(33) & 255) != 27) {
        cap >> frame; // get a new frame from video file
//        Mat matRotation = getRotationMatrix2D(Point(test.cols/2,test.rows/2),180,1);
//        warpAffine(test,test, matRotation,test.size());
        AAtime = getTickCount();
        printf("%d \t", frameNum);
        //ballPos = cv_getPos(frame, frame_gray, last_ballPost);

        detectTest(hog, hitThreshold, frame, frame_gray);

        //time check
        BBtime = getTickCount();
        double s_time = (BBtime - AAtime) / getTickFrequency();
        double fps_time = 1/s_time;
        printf("%.5lf \t%.5lf \n", s_time, fps_time);
        imshow("HOG ball detection", frame);
        frameNum++;


    }
    // </editor-fold>

    return EXIT_SUCCESS;
}
