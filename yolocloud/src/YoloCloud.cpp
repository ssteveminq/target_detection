#include "villa_yolocloud/YoloCloud.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>

using namespace cv;
using namespace std;

const float NEIGHBOR_THRESH = 0.1;

/*
 * Adds object
 * Returns index in point cloud or -1 if not added
 */
int YoloCloud::addObject(const ImageBoundingBox &bbox,
        const Mat &rgb_image,
        const Mat &depth_image,
        const Eigen::Affine3f &camToMap) {
    if (bbox.y + bbox.height > rgb_image.rows
        || bbox.x + bbox.width > rgb_image.cols) {
        //std::cout << bbox.x << std::endl;
        //std::cout << bbox.y << std::endl;
        //std::cout << bbox.width << std::endl;
        //std::cout << bbox.height << std::endl;
        //throw std::runtime_error("Invalid bbox");
        return -1;
    }

    // define bounding rectangle
    Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);

    Mat mask; // segmentation result (4 possible values)
    Mat bgModel, fgModel; // the models (internally used)

    // GrabCut segmentation
    grabCut(rgb_image,    // input image
                mask,   // segmentation result
                rect,// rectangle containing foreground 
                bgModel, fgModel, // models
                1,        // number of iterations
                GC_INIT_WITH_RECT); // use rectangle

#ifdef DEBUG
    Mat image_mask(mask.size(), CV_8UC1);
    compare(mask, GC_PR_FGD, image_mask, CMP_EQ);
    Mat foreground(rgb_image.size(), CV_8UC3, Scalar(255,255,255));
    rgb_image.copyTo(foreground, image_mask);

    // draw rectangle on original image
    Mat image(rgb_image.size(), CV_8UC3);
    rgb_image.copyTo(image);
    cv::rectangle(image, rect, Scalar(255,255,255), 1);
    namedWindow("Image");
    imshow("Image", image);

    // display result
    namedWindow("Segmented Image");
    imshow("Segmented Image", foreground);

    waitKey(0);
#endif

    // TODO: Use something better than the average. Grab biggest peak, or verify depth is unimodal first
    float depth = 0.;
    int count = 0;
    for (size_t y = bbox.y; y < bbox.y + bbox.height; y++) {
        for (size_t x = bbox.x; x < bbox.x + bbox.width; x++) {
            uint8_t mask_val = mask.at<uint8_t>(y, x);
            float cur_depth = depth_image.at<float>(y, x);
            if (cur_depth != 0 && !isnan(cur_depth) && (mask_val == GC_PR_FGD || mask_val == GC_FGD)) {
                depth += cur_depth;
                count++;
            }
        }
    }
    depth /= count;

    // No valid depth, so return
    if (count == 0) {
        return -1;
    }

    // If the label does not yet exist, add it
    auto it = labels_to_id.find(bbox.label);
    uint32_t id;
    if (it == labels_to_id.end()) {
        id = labels.size();
        labels_to_id.insert({bbox.label, id});
        labels.push_back(bbox.label);
    } else {
        id = it->second;
    }

    // Compute 3d point in the world
    float x_center = bbox.x + bbox.width/2.;
    float y_center = bbox.y + bbox.height/2.;
    float camera_x = (x_center - intrinsic_cx) * (1.0 / intrinsic_sx) * depth;
    float camera_y = (y_center - intrinsic_cy) * (1.0 / intrinsic_sy) * depth;
    float camera_z = depth;
    Eigen::Vector3f world = camToMap * Eigen::Vector3f(camera_x, camera_y, camera_z);

    pcl::PointXYZL point;
    point.x = world(0);
    point.y = world(1);
    point.z = world(2);
    point.label = id;

    if (!objects->points.empty()) {
        // Check if point already exists (or close enough one already exists)
        pcl::KdTree<pcl::PointXYZL>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZL>);
        tree->setInputCloud(objects);
        std::vector<int> nn_indices(1);
        std::vector<float> nn_dists(1);
        if (!objects->empty()
            && tree->nearestKSearch(point, 1, nn_indices, nn_dists)) {
            if (objects->points[nn_indices[0]].label == point.label && nn_dists[0] <= NEIGHBOR_THRESH) {
              // TODO: Average with neighbors
                // We don't want to add this as it is probably a duplicate
                return -1;
            }
        }
    }

    objects->push_back(point);
    return objects->size() - 1;
}

pcl::PointCloud<pcl::PointXYZL>::Ptr YoloCloud::sliceCloud(float x_min, float x_max,
                 float y_min, float y_max,
                 float z_min, float z_max) {

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PassThrough<pcl::PointXYZL> pass;

    pass.setInputCloud(objects);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x_min, x_max);
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(y_min, y_max);
    pass.filter(*cloud_filtered);

    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min, z_max);
    pass.filter(*cloud_filtered);

    return cloud_filtered;
}

