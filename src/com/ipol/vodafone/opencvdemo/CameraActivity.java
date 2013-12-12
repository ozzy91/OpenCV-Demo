package com.ipol.vodafone.opencvdemo;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.ipol.vodafone.opencvdemo.filter.GoalDetectionFilter;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

public class CameraActivity extends Activity implements CvCameraViewListener2 {

	private static final String TAG = "OpenCV Demo";

	private CameraBridgeViewBase mOpenCvCameraView;
	private GoalDetectionFilter goalFilter;

	private int mTotalSquare = 0;
	private long mSeconds;
	private Mat rgbImage;
	private Vector<DrawnContours> mDrawnContours = new Vector<DrawnContours>();
	private ArrayList<MatOfPoint> contours;
	List<MatOfPoint> squares = new ArrayList<MatOfPoint>();
	List<MatOfPoint2f> squares2f = new ArrayList<MatOfPoint2f>();
	private double maxArea = 0;
	private MatOfPoint maxTempPoint;
	private MatOfPoint secondMaxPoint;

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				goalFilter = new GoalDetectionFilter();
				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		DisplayManager.INSTANCE.init(this);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		setContentView(R.layout.activity_camera);
		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
		mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
		mOpenCvCameraView.setCvCameraViewListener(this);
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_7, this, mLoaderCallback);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
	}

	public void onCameraViewStopped() {
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
//		Mat frame = inputFrame.rgba();
//		if (goalFilter != null)
//			goalFilter.processFrame(inputFrame.rgba());
//		return goalFilter.processFrame(frame);
		return goalFilter.processFrame(inputFrame.rgba());
	}

	public Mat edgeDetect(Mat src) {
		Mat pyr = new Mat();
		Mat timing = new Mat();

		Imgproc.pyrDown(src, pyr, new Size(src.width() / 2, src.height() / 2));
		Imgproc.pyrUp(pyr, timing, src.size());

		Mat blurred = new Mat();
		timing.copyTo(blurred);
		// src.copyTo(blurred);
		Log.v(TAG, "Blurred Matrix! : " + blurred.total());

		Imgproc.medianBlur(src, blurred, 9);
		Log.v(TAG, "Median Blur Done!");

		Mat gray0 = new Mat(blurred.size(), blurred.type());
		Imgproc.cvtColor(gray0, gray0, Imgproc.COLOR_RGB2GRAY);
		Mat gray = new Mat();

		squares.clear();
		squares2f.clear();
		Log.v(TAG, "Gray0 Matrix! : " + gray0.total());
		Log.v(TAG, "Gray Matrix! : " + gray.total());

		// List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

		// find squares in every color plane of the image
		mDrawnContours.clear();

		for (int c = 0; c < 3; c++) {
			Log.v(TAG, "Mix Channels Started! : " + gray0.total());
			int ch[] = { c, 0 };
			MatOfInt fromto = new MatOfInt(ch);
			List<Mat> blurredlist = new ArrayList<Mat>();
			List<Mat> graylist = new ArrayList<Mat>();

			blurredlist.add(0, timing);
			graylist.add(0, gray0);

			Core.mixChannels(blurredlist, graylist, fromto);
			gray0 = graylist.get(0);
			Log.v(TAG, "Mix Channels Done! : " + gray0.total());
			// try several threshold levels
			int threshold_level = 11;
			for (int l = 0; l < threshold_level; l++) {
				// Use Canny instead of zero threshold level!
				// Canny helps to catch squares with gradient shading
				Log.v(TAG, "Threshold Level: " + l);

				if (l == 0) {
					Imgproc.Canny(gray0, gray, 50, 5); //

					// Dilate helps to remove potential holes between edge
					// segments
					// Imgproc.dilate(gray, gray, Mat.ones(new Size(3,3),0));

					Imgproc.dilate(gray, gray, Mat.ones(new Size(3, 3), 0));

				} else {
					int thresh = (l + 1) * 255 / threshold_level;
					// int thresh = 50;
					Imgproc.threshold(gray0, gray, thresh, 255, Imgproc.THRESH_TOZERO);
				}

				Log.v(TAG, "Canny (or Thresholding) Done!");
				Log.v(TAG, "Gray Matrix (after)! : " + gray.total());
				contours = new ArrayList<MatOfPoint>();
				// Find contours and store them in a list
				Imgproc.findContours(gray, contours, new Mat(), 1, 2);
				Log.v(TAG, "Contours Found!");

				MatOfPoint2f approx = new MatOfPoint2f();
				MatOfPoint2f mMOP2f1 = new MatOfPoint2f();
				MatOfPoint mMOP = new MatOfPoint();
				for (int i = 0; i < contours.size(); i++) {
					contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
					Imgproc.approxPolyDP(mMOP2f1, approx, Imgproc.arcLength(mMOP2f1, true) * 0.02, true);
					approx.convertTo(mMOP, CvType.CV_32S);

					if (approx.rows() == 4 && Math.abs(Imgproc.contourArea(approx)) > 1000
							&& Imgproc.isContourConvex(mMOP)) {

						Log.v(TAG, "Passes Conditions! " + approx.size().toString());
						double maxcosine = 0;
						Point[] list = approx.toArray();

						for (int j = 2; j < 5; j++) {
							double cosine = Math.abs(angle(list[j % 4], list[j - 2], list[j - 1]));
							maxcosine = Math.max(maxcosine, cosine);
						}

						if (maxcosine < 0.3) {
							MatOfPoint temp = new MatOfPoint();
							approx.convertTo(temp, CvType.CV_32S);
							squares.add(temp);
							squares2f.add(approx);
							DrawnContours drawnContour = new DrawnContours();
							drawnContour.setIndex(i);
							mDrawnContours.add(drawnContour);
							double area = Math.abs(Imgproc.contourArea(temp));
							if (maxArea < area) {
								maxArea = area;
								secondMaxPoint = maxTempPoint;
								maxTempPoint = temp;

							}
							// Rect rect= Imgproc.boundingRect(temp);
							//
							// Core.rectangle(src, rect.tl() , new
							// Point(rect.width, rect.height), new
							// Scalar(0,125,255));
							Imgproc.drawContours(src, contours, i, new Scalar(0, 0, 255));
						}
					}

				}
				Log.v(TAG, "Squares Added to List! : " + squares.size());
			}
		}

		Log.i(TAG, "::EdgeDetect:" + "squares2f.size():" + squares2f.size());
		// if(maxTempPoint!=null){
		// Rect rect= Imgproc.boundingRect(maxTempPoint);
		// Core.rectangle(src, new Point(rect.x, rect.y) , new Point(rect.width,
		// rect.height), new Scalar(0,125,255));
		// if(secondMaxPoint!=null){
		// Rect rect1= Imgproc.boundingRect(maxTempPoint);
		// Core.rectangle(src, new Point(rect1.x, rect1.y) , new
		// Point(rect1.width, rect1.height), new Scalar(0,125,255));
		// }
		// }

		return src;
	}

	double angle(Point pt1, Point pt2, Point pt0) {
		double dx1 = pt1.x - pt0.x;
		double dy1 = pt1.y - pt0.y;
		double dx2 = pt2.x - pt0.x;
		double dy2 = pt2.y - pt0.y;
		return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
	}
}
