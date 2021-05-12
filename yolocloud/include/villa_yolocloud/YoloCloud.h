#ifndef YOLOCLOUD_H
#define YOLOCLOUD_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

struct ImageBoundingBox {
    float x;
    float y;
    int width;
    int height;
    std::string label;
};

class YoloCloud {
private:
    // Hardcoded intrinsic parameters for the HSR xtion
    float intrinsic_sx = 535.2900990271;
    float intrinsic_sy = 535.2900990271;
    float intrinsic_cx = 320.0000000000;
    float intrinsic_cy = 240.0000000000;

public:
    pcl::PointCloud<pcl::PointXYZL>::Ptr objects;
    std::vector<std::string> labels;
    std::map<std::string, uint32_t> labels_to_id;

    YoloCloud()
        : objects(new pcl::PointCloud<pcl::PointXYZL>) {
    }

    int addObject(const ImageBoundingBox &bbox, const cv::Mat &rgb_image, const cv::Mat &depth_image, const Eigen::Affine3f &camToMap);

    pcl::PointCloud<pcl::PointXYZL>::Ptr sliceCloud(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);

};

#endif // YOLOCLOUD_H
