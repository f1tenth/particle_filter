#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include "darknet_ros_msgs/BoundingBoxes.h"
#include <opencv2/highgui/highgui.hpp>

class VideoVisualizer {
public:
  VideoVisualizer():
      nh_(ros::NodeHandle()),
      it_(nh_),
      current_frame_(){
    img_sub_ =
        it_.subscribe("/camera/image_raw", 1, &VideoVisualizer::imageReceiveCallback, this);
    bbox_sub_ = nh_.subscribe("/pf/bbox", 1, &VideoVisualizer::bboxReceiveCallback, this);
  }

  void imageReceiveCallback(const sensor_msgs::ImageConstPtr& msg){
    current_frame_ = cv_bridge::toCvShare(msg, "bgr8")->image;   
    vis(); 
  }

  void bboxReceiveCallback(const darknet_ros_msgs::BoundingBoxes& msg){
    for (const auto& bbox : msg.bounding_boxes) {
      if (!current_frame_.empty()) {
        cv::Point p1(int(bbox.xmin), int(bbox.ymin));
        cv::drawMarker(current_frame_, p1, 255);
        vis();
      }
    }
  }

  void vis() {
    cv::imshow("View image message", current_frame_);
    cv::waitKey(1);
  }

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber img_sub_;
  ros::Subscriber bbox_sub_;
  cv::Mat current_frame_;
};



int main(int argc, char** argv) {
  ros::init(argc, argv, "image_subscriber_test_node");
  cv::namedWindow("View image message");
  VideoVisualizer VV;
  ros::spin();

  cv::destroyAllWindows();

  return 0;
}
