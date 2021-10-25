#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/highgui/highgui.hpp>

void imageReceiveCallback(const sensor_msgs::ImageConstPtr& msg) {
  try {
    cv::imshow("View image message", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(1);
  } catch (const cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "image_subscriber_test_node");
  cv::namedWindow("View image message");

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber subscriber =
      it.subscribe("/camera/image_raw", 1, imageReceiveCallback);

  ros::spin();

  cv::destroyAllWindows();

  return 0;
}
