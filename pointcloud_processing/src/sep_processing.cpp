#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <pointcloud_processing_msgs/ObjectInfo.h>
#include <pointcloud_processing_msgs/ObjectInfoArray.h>
#include <pointcloud_processing_msgs/fov_positions.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/CameraInfo.h>
#include <vision_msgs/Detection2DArray.h>

#include <math.h> 

// Approximate time policy
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
using namespace message_filters;

// Darknet detection
#include <darknet_ros_msgs/BoundingBoxes.h>

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>

// typedefs
typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::Ptr CloudPtr;

typedef int ObjectClassID;
typedef std::string ObjectClassName;
typedef std::map<ObjectClassName, ObjectClassID> ObjectsMap;

const int QUEUE_SIZE = 10;
std::string DARKNET_TOPIC;
std::string RETINANET_TOPIC;
std::string PCL_TOPIC;
std::string FOV_TOPIC;
std::string TARGET_FRAME;
std::string TARGET_CLASS;
bool VISUAL;
bool PCL_VISUAL;


// ROS Nodehandle
ros::NodeHandle *nh;

// Publishers
ros::Publisher detected_objects_pub;

// Initialize transform listener
tf::TransformListener* lst;
tf2_ros::Buffer* pbuffer;

// caches for callback data
darknet_ros_msgs::BoundingBoxes current_boxes;
pointcloud_processing_msgs::fov_positions current_fov;
sensor_msgs::CameraInfo camera_info;

//map
ObjectsMap object_classes;

bool received_first_message=false;
bool received_first_message_bbox =false;
bool received_first_message_cloud =false;
bool received_fov_region=false;

typedef struct
{
    int32_t x;
    int32_t y;
} PixelCoords;

void BBoxCb(const darknet_ros_msgs::BoundingBoxesConstPtr& input_detection)
{
    //ROS_INFO("bounding_box callback");
    current_boxes = *input_detection;

    received_first_message_bbox =true;
}


void FoVRegionCb(const pointcloud_processing_msgs::fov_positionsConstPtr& fov_region_)
{
    //ROS_INFO("fov_regions callback");
    current_fov=*fov_region_;
    received_fov_region=true;
}


void CameraInfoCb(const sensor_msgs::CameraInfoConstPtr msg)
{
    camera_info = *msg;
}


/**
 * @brief Convert a point in the camera frame to (u,v) pixel coordinates
 * 
 * @param point The cartesian point to convert
 * @param camera_info A 3x3 matrix intrinsic to the camera.
 * 
 * @return A pair containing the (x,y) pixel coordinates.
 */
inline PixelCoords poseToPixel(const PointType &point,
                               const sensor_msgs::CameraInfo &camera_info)
{
    PixelCoords result;

    result.x = camera_info.K[0]*point.x + camera_info.K[2]*point.z;
    result.y = camera_info.K[4]*point.y + camera_info.K[5]*point.z;

    return result;
}


std::vector<PixelCoords> convertCloudToPixelCoords(const CloudPtr cloud,
                                                   const sensor_msgs::CameraInfo &camera_info)
{
    std::vector<PixelCoords> output;
    output.reserve(cloud->size());

    for (const PointType &point : cloud->points)
    {
        output.push_back( poseToPixel(point, camera_info) );
    }

    return output;
}


ObjectClassID getObjectID(const ObjectClassName class_name, const ObjectsMap &map)
{
    ObjectClassID class_id;

    try
    {
        class_id = map.at(class_name);
    }
    catch(const std::out_of_range& e)
    {
        ROS_ERROR("getObjectID() - No class ID found for name %s", class_name.c_str());
        std::cerr << e.what() << '\n';
        return ObjectClassID(NAN);
    }

    return class_id;
}


ObjectsMap convertClassesMap(std::map<std::string, std::string> input)
{
    ObjectsMap output;

    for (const std::map<std::string, std::string>::value_type &pair : input)
    {
        output[pair.first] = std::stoi(pair.second);
    }

    return output;
}


CloudPtr filterPointsInFoV(const CloudPtr input,
                           const std::vector<PixelCoords> &pixel_coordinates,
                           const int xmin,
                           const int xmax,
                           const int ymin,
                           const int ymax)
{
    pcl::PointIndices::Ptr indices_in_bbox;
    indices_in_bbox->indices.reserve(input->size());

    for (int i = 0; i < pixel_coordinates.size(); ++i)
    {
        if (pixel_coordinates[i].x < xmin &&
            pixel_coordinates[i].x < xmax &&
            pixel_coordinates[i].y < ymin &&
            pixel_coordinates[i].y < ymax)
        {
            indices_in_bbox->indices.push_back(i);
        }
    }

    indices_in_bbox->indices.shrink_to_fit();

    CloudPtr cloud_in_bbox(new Cloud);
    pcl::ExtractIndices<PointType> camera_fov_filter;

    // Extract the inliers  of the ROI
    camera_fov_filter.setInputCloud(input);
    camera_fov_filter.setIndices(indices_in_bbox);
    camera_fov_filter.setNegative(false);
    camera_fov_filter.filter(*cloud_in_bbox);

    return cloud_in_bbox;
}


CloudPtr filterPointsInBox(const CloudPtr input,
                           const std::vector<PixelCoords> &pixel_coordinates,
                           const int xmin,
                           const int xmax,
                           const int ymin,
                           const int ymax)
{
    pcl::PointIndices::Ptr indices_in_bbox;
    indices_in_bbox->indices.reserve(input->size());

    for (int i = 0; i < pixel_coordinates.size(); ++i)
    {
        if (pixel_coordinates[i].x < xmin &&
            pixel_coordinates[i].x < xmax &&
            pixel_coordinates[i].y < ymin &&
            pixel_coordinates[i].y < ymax)
        {
            indices_in_bbox->indices.push_back(i);
        }
    }

    indices_in_bbox->indices.shrink_to_fit();

    CloudPtr cloud_in_bbox(new Cloud);
    pcl::ExtractIndices<PointType> camera_fov_filter;

    // Extract the inliers  of the ROI
    camera_fov_filter.setInputCloud(input);
    camera_fov_filter.setIndices(indices_in_bbox);
    camera_fov_filter.setNegative(false);
    camera_fov_filter.filter(*cloud_in_bbox);

    return cloud_in_bbox;
}


void pointCloudCb(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
{
    received_first_message_cloud = true;

    if (!received_fov_region)
    {
        return;
    }

    const ros::Time now = ros::Time::now();

    double confidence_threshold;
    nh->param("confidence_threshold", confidence_threshold, 0.75);

    //ROS_INFO("cloud callback");
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf2_listener(tf_buffer);
    received_first_message = true;

    // Initialize containers for clouds
    CloudPtr cloud(new Cloud);
    CloudPtr cloud_target(new Cloud);

    // Initialize container for object poses
    geometry_msgs::PoseArray target_poses;
    target_poses.header.stamp = now;
    target_poses.header.frame_id = input_cloud->header.frame_id;

    // Initialize container for centroids' markers
    visualization_msgs::Marker centroid_target_list;

    centroid_target_list.header.frame_id = input_cloud->header.frame_id;
    centroid_target_list.type = visualization_msgs::Marker::SPHERE_LIST;
    centroid_target_list.color.a = 1.0;
    centroid_target_list.color.r = 1.0;
    centroid_target_list.action = visualization_msgs::Marker::ADD;

    centroid_target_list.scale.x = 0.05;
    centroid_target_list.scale.y = 0.05;
    centroid_target_list.scale.z = 0.05;

    // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
    pcl::fromROSMsg(*input_cloud, *cloud);

    // Initialize container for inliers of the ROI
    pcl::PointIndices::Ptr inliers_camera_fov(new pcl::PointIndices());

    //check pcl and rgb frames are using same frame_id 
    if (input_cloud->header.frame_id != current_fov.header.frame_id)
    {
        ROS_INFO("frame is not same! cloud : %s , fov_region: %s", input_cloud->header.frame_id.c_str(), current_fov.header.frame_id.c_str());
        return;
    }

    inliers_camera_fov->indices.reserve(cloud->size());

    //Check the 3D FOV region to gather if xyz of point cloud data are within 3D FOV region:
    //checking two points current_fov.position[0] // current_fov.positions[1]
    for (int k = 0; k < cloud->points.size(); k++)
    {
        if( (abs(cloud->points[k].y-current_fov.positions[0].y) <1.0) && (abs(cloud->points[k].z-current_fov.positions[0].z) <1.0)) 
        {
            if( (abs(cloud->points[k].y-current_fov.positions[1].y) <2.0) && (abs(cloud->points[k].y-current_fov.positions[1].z) <2.0)) 
            {
                if( ((cloud->points[k].x-current_fov.positions[0].x) >0.5) && (abs(cloud->points[k].x-current_fov.positions[1].x) <3.0))
                {
                    inliers_camera_fov->indices.push_back(k);
                }
            }
        }
    }

    inliers_camera_fov->indices.shrink_to_fit();

    if (inliers_camera_fov->indices.empty())
    {
        ROS_WARN("No pointcloud data found within the camera field of view.");
        return;
    }


    // -------------------Extraction of points in the camera FOV------------------------------
    // Create the filtering object
    CloudPtr cloud_filtered(new Cloud);
    pcl::ExtractIndices<PointType> camera_fov_filter;

    // Extract the inliers  of the ROI
    camera_fov_filter.setInputCloud(cloud);
    camera_fov_filter.setIndices(inliers_camera_fov);
    camera_fov_filter.setNegative(false);
    camera_fov_filter.filter(*cloud_filtered);

    // ----------------------Voxel Downsampling----------------------------------
    pcl::VoxelGrid<PointType> sor_voxelgrid;
    sor_voxelgrid.setInputCloud(cloud_filtered);
    sor_voxelgrid.setLeafSize(0.05, 0.05, 0.05); //size of the grid
    sor_voxelgrid.filter(*cloud_filtered);

    // ---------------------StatisticalOutlierRemoval--------------------
    // Remove statistical outliers from the downsampled cloud
    pcl::StatisticalOutlierRemoval<PointType> sor_noise;
    sor_noise.setInputCloud(cloud_filtered);
    sor_noise.setMeanK(50);
    sor_noise.setStddevMulThresh(1.0);
    sor_noise.filter(*cloud_filtered);

    // remove NaN points from the cloud
    CloudPtr nanfiltered_cloud(new Cloud);
    std::vector<int> rindices;
    pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, rindices);

    // output
    vision_msgs::Detection2DArray detected_objects;
    detected_objects.header.stamp = now;
    detected_objects.header.frame_id = input_cloud->header.frame_id;
    detected_objects.detections.reserve(current_boxes.bounding_boxes.size());

    // produce pixel-space coordinates
    const std::vector<PixelCoords> pixel_coordinates = convertCloudToPixelCoords(cloud_filtered, camera_info);

    /////////////////////////////////////////////////////////////
    for(const darknet_ros_msgs::BoundingBox &box : current_boxes.bounding_boxes)
    {
        // do we meet the threshold for a confirmed detection?
        if (box.probability >= confidence_threshold)
        {            
            // ----------------------Extract points in the bounding box-----------
            const CloudPtr cloud_in_bbox = filterPointsInBox(cloud_filtered,
                                                             pixel_coordinates,
                                                             box.xmin,
                                                             box.xmax,
                                                             box.ymin,
                                                             box.ymax);
            
            // ----------------------Compute centroid-----------------------------
            Eigen::Vector4f centroid_out;
            pcl::compute3DCentroid(*cloud_in_bbox, centroid_out); 

            // add to the output
            vision_msgs::Detection2D object;
            object.bbox.center.x = (box.xmax + box.xmin)/2;
            object.bbox.center.y = (box.ymax + box.ymin)/2;
            object.bbox.size_x = box.xmax - box.xmin;
            object.bbox.size_y = box.ymax - box.ymin;

            vision_msgs::ObjectHypothesisWithPose hypothesis;
            hypothesis.id = getObjectID(box.Class, object_classes);
            hypothesis.score = box.probability;
            hypothesis.pose.pose.position.x = centroid_out[0];
            hypothesis.pose.pose.position.y = centroid_out[1];
            hypothesis.pose.pose.position.z = centroid_out[2];
            hypothesis.pose.pose.orientation.w = 1;

            object.results.push_back(hypothesis);

            detected_objects.detections.push_back(object);
        }
    }

    // publish results
    detected_objects_pub.publish(detected_objects);

    ROS_INFO("Pointcloud processed");
}



int
main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "pointcloud_processing");
    nh = new ros::NodeHandle;

    nh->param("DARKNET_TOPIC", DARKNET_TOPIC, {"/darknet_ros/bounding_boxes"});
    nh->param("RETINANET_TOPIC", RETINANET_TOPIC, {"/retina_ros/bounding_boxes"});
    nh->param("PCL_TOPIC", PCL_TOPIC, {"/pointcloud_transformer/output_pcl2"});
    nh->param("FOV_TOPIC", FOV_TOPIC, {"/fov_regions"});
    //nh->param("PCL_TOPIC", PCL_TOPIC, {"/points2"});
    nh->param("TARGET_FRAME", TARGET_FRAME, {"odom"});
    nh->param("TARGET_CLASS", TARGET_CLASS, {"chair"});
    nh->param("VISUAL", VISUAL, {true});
    nh->param("PCL_VISUAL", PCL_VISUAL, {true});
    
    std::map<std::string, std::string> temp_map;
    if (!nh->getParam("object_classes", temp_map))
    {
        ROS_ERROR("Failed to load dictionary parameter 'object_classes'.");
        return 1;
    }

    try
    {
        object_classes = convertClassesMap(temp_map);
    } catch(std::invalid_argument ex)
    {
        ROS_FATAL("Invalid object_classes parameter.");
        return 1;
    }

    // Initialize subscribers to darknet detection and pointcloud
    ros::Subscriber bbox_sub = nh->subscribe<darknet_ros_msgs::BoundingBoxes>(DARKNET_TOPIC, 100, BBoxCb); 
    ros::Subscriber cloud_sub = nh->subscribe<sensor_msgs::PointCloud2>(PCL_TOPIC, 100, pointCloudCb);
    ros::Subscriber fovregion_sub = nh->subscribe<pointcloud_processing_msgs::fov_positions>(FOV_TOPIC, 100, FoVRegionCb);
    ros::Subscriber camera_info_sub = nh->subscribe<sensor_msgs::CameraInfo>("camera_info", 100, CameraInfoCb);

    // Initialize transform listener
    //tf::TransformListener listener(ros::Duration(10));
    //lst = &listener;
    //tf2_ros::Buffer tf_buffer(ros::Duration(100));
    //pbuffer = &tf_buffer;

    // Synchronize darknet detection with pointcloud
    //sync.registerCallback(boost::bind(&cloud_cb, _1, _2));

    // Create a ROS publisher for the output point cloud
    detected_objects_pub = nh->advertise<vision_msgs::Detection2DArray>("detected_objects", 1);

    ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && (!received_first_message_bbox && !received_first_message_cloud)) {
        ROS_WARN_THROTTLE(2, "Waiting for image, depth and detections messages...");
        ros::spinOnce();
    }
        
    // Spin
    ros::spin ();

    return 0;
}
