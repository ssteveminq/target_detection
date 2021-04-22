#ifndef CONTEXT_CLOUD
#define CONTEXT_CLOUD

#include <string>
#include <iostream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <ros/ros.h>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <math.h>
#include <villa_yolocloud/BboxInfo.h>
#include <villa_yolocloud/ObjectInfo.h>
#include <villa_yolocloud/ObjectInfoArray.h>
#include <XmlRpcValue.h>
#include <yaml-cpp/yaml.h>

struct ImageBoundingBox {
    float x;
    float y;
    int width;
    int height;
    std::string label;
};

class ContextCloud {
private:
    // Hardcoded intrinsic parameters for the HSR xtion
    float intrinsic_sx = 535.2900990271;
    float intrinsic_sy = 535.2900990271;
    float intrinsic_cx = 320.0000000000;
    float intrinsic_cy = 240.0000000000;

public:
    pcl::PointCloud<pcl::PointXYZL>::Ptr objects;
    std::vector<std::string> labels;
    std::map<std::string, std::vector<uint32_t> > labels_to_idset;
    std::map<uint32_t, pcl::PointXYZL > id_to_position;
    std::map<uint32_t, ImageBoundingBox > id_to_bbox;
    std::map<uint32_t, float> id_to_depth;
    std::map<uint32_t, ros::Time> id_to_time;
    std::vector<std::string> TargetSet;
    std::map<std::string, int> current_list;


    ContextCloud()
        : objects(new pcl::PointCloud<pcl::PointXYZL>) {
        
    }

    int addObject(const ImageBoundingBox &bbox, const cv::Mat &rgb_image, const cv::Mat &depth_image, const Eigen::Affine3f &camToMap);

    pcl::PointCloud<pcl::PointXYZL>::Ptr sliceCloud(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);

    int getObjectsize();


};

#endif // CONTEXT_CLOUD
