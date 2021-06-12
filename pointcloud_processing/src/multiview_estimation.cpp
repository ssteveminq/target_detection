#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <pointcloud_processing_msgs/ObjectInfo.h>
#include <pointcloud_processing_msgs/ObjectInfoArray.h>
#include <pointcloud_processing_msgs/fov_positions.h>
#include <pointcloud_processing_msgs/ClusterArray.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/CameraInfo.h>
#include <vision_msgs/Detection2DArray.h>
#include <geometry_msgs/TransformStamped.h>

//#include <cmath.h> 
#include <math.h> 

// Approximate time policy
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
using namespace message_filters;

// Darknet detection
#include <darknet_ros_msgs/BoundingBoxes.h>
//Odom
#include <nav_msgs/Odometry.h>

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
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl_ros/transforms.h> 
#include <pcl/features/normal_3d.h>
#include <pointcloud_processing_msgs/get_targetpose.h>
#include <pointcloud_processing_msgs/register_scene.h>

#define UNKNOWN_OBJECT_ID -1


// typedefs
typedef pcl::PointXYZRGB PointType;
// typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef geometry_msgs::Point gPoint;

typedef Cloud::Ptr CloudPtr;

typedef int ObjectClassID;
typedef std::string ObjectClassName;
typedef std::map<ObjectClassName, ObjectClassID> ObjectsMap;
typedef std::map<int, geometry_msgs::PoseArray> PoseMap;
typedef std::map<int, geometry_msgs::PoseArray>::iterator PoseMapIter;

// the optical frame of the RGB camera (not the link frame)
std::string rgb_optical_frame_;


const int QUEUE_SIZE = 10;
const int MAX_DEPTH=3;
const int MIN_DIST=0.5;

std::string DARKNET_TOPIC;
std::string RETINANET_TOPIC;
std::string PCL_TOPIC;
std::string FOV_TOPIC;
std::string TARGET_FRAME;
std::string TARGET_CLASS;
bool VISUAL;
bool PCL_VISUAL;
bool debug_lidar_viz;


pointcloud_processing_msgs::ClusterArray Cloudbag;
pointcloud_processing_msgs::ClusterArray CloudFOVbag;
pointcloud_processing_msgs::ClusterArray segmentbag;
sensor_msgs::PointCloud2 registered_pcl;
sensor_msgs::PointCloud2 registered_pcl_filtered;
PoseMap seg_poses_map;
std::vector<geometry_msgs::PoseStamped> robot_poses;
geometry_msgs::PoseStamped target_pose;
geometry_msgs::PoseStamped previous_target_pose;
std::vector<geometry_msgs::Point> bboxpoints;
std::vector<std::vector<geometry_msgs::Point>> registered_bboxpoints;
visualization_msgs::Marker registered_bbox_vertices;
visualization_msgs::Marker registered_bbox_lines;
//std::vector<int> seg_idxset;


// ROS Nodehandle
ros::NodeHandle *nh;

// Publishers
ros::Publisher detected_objects_pub;
ros::Publisher lidar_fov_pub_;
ros::Publisher lidar_bbox_pub_;
ros::Publisher lidar_temp_pub_;
ros::Publisher pcl_center_pub;
ros::Publisher inplane_pub;
ros::Publisher outplane_pub;
ros::Publisher clusters_pub;
ros::Publisher pcl_map_pub;
ros::Publisher pcl_registered_pub;
ros::Publisher seg1_pub;
ros::Publisher seg2_pub;
ros::Publisher seg3_pub;
ros::Publisher segposes_pub;
ros::Publisher bbox_marker_pub;
ros::Publisher bboxline_marker_pub;

// Initialize transform listener
tf::TransformListener* lst;
tf2_ros::Buffer* pbuffer;
//tf2_ros::TransformListener tf_listener_;
tf::StampedTransform transform_;

// caches for callback data
darknet_ros_msgs::BoundingBoxes current_boxes_;
pointcloud_processing_msgs::fov_positions current_fov;
sensor_msgs::CameraInfo camera_info;
geometry_msgs::PoseStamped robot_pose;

//map
ObjectsMap object_classes;

CloudPtr cur_cloud;
CloudPtr cur_cloud_fov;
CloudPtr cur_cloud_bbox;
CloudPtr cur_cloud_map;

bool received_first_message=false;
bool received_first_message_bbox =false;
bool received_first_message_cloud =false;
bool received_fov_region=false;
bool received_robot_pose=false;
bool first_registered=true;
bool firstly_called=true;
bool target_obtained=false;
bool bool_cloud_fov=false;
bool received_camera_info=false;


void crossProduct_Point(const gPoint p1, const gPoint p2, gPoint& output)
{
   output.x = p1.y * p2.z - p1.z * p2.y;
   output.y = -(p1.x * p2.z - p1.z * p2.x);
   output.z = p1.x * p2.y - p1.y * p2.x;
}

void normalize_Point(gPoint& p1)
{
   double norm_ =sqrt(pow(p1.x,2)+pow(p1.y,2)+pow(p1.z,2));
   p1.x = p1.x/norm_;
   p1.y = p1.y/norm_;
   p1.z = p1.z/norm_;
}

double dotProduct_Point(const gPoint p1, const gPoint p2)
{
    return (p1.x*p2.x+p1.y*p2.y+p1.z*p2.z);
}

class Frustum
{
 /******
    A class for Frustum ( Field of View volume)
 ******/

public:
    //std::vector<gPoint> pointset; //8points
    std::vector<std::vector<double>> plane_coeffset; //8 coefficients (a,b,c,d)

public:
    /*
    Front plane : flt-0, flb-4, frt-2, frb-6
    Rear plane : rlt-1, rlb-5, rrt-3, rrb-7
    make 6 planes with three points inside plane
    store coefficient of six planes
    */
    Frustum(std::vector<gPoint> pointset){

        if(pointset.size()!=8)
        {
            ROS_WARN("frustum requires more points!");
        }
        else{
            plane_coeffset.resize(5);
            /* the direction vector is towards inside of the volume of frustum */
            plane_coeffset[0] = getPlanewithTreepoints(pointset[0], pointset[2], pointset[1]); //top plane
            plane_coeffset[1] = getPlanewithTreepoints(pointset[6], pointset[4], pointset[7]); //bottom plane
            plane_coeffset[2] = getPlanewithTreepoints(pointset[0], pointset[1], pointset[4]); //left plane
            plane_coeffset[3] = getPlanewithTreepoints(pointset[6], pointset[7], pointset[2]); //right plane
            plane_coeffset[4] = getPlanewithTreepoints(pointset[0], pointset[4], pointset[2]); //front plane
            //plane_coeffset[5] = getPlanewithTreepoints(pointset[3], pointset[5], pointset[1]); //rear plane
        }
    }

    std::vector<double> getPlanewithTreepoints(const gPoint p1, const gPoint p2, const gPoint p3)
    {
        geometry_msgs::Point v1,v2;
        v1.x=p3.x-p1.x;
        v1.y=p3.y-p1.y;
        v1.z=p3.z-p1.z;

        v2.x=p2.x-p1.x;
        v2.y=p2.y-p1.y;
        v2.z=p2.z-p1.z;

        //cross-product
        geometry_msgs::Point cp_;
        crossProduct_Point(v1, v2, cp_);
        normalize_Point(cp_); //normal vector (a,b,c)
        double d = -(dotProduct_Point(cp_, p3)); //plane constant d

        //plane (a, b, c, d)
        std::vector<double> plane_coeff(4,0.0);
        plane_coeff[0]=cp_.x; plane_coeff[1]=cp_.y;
        plane_coeff[2]=cp_.z; plane_coeff[3]=d;

        return plane_coeff;

    }
    /* check query point within the volume of frustum - using dot product with normal vectors of five planes */
    bool querypointInFrustum(const gPoint query)
    {
        //ROS_INFO_STREAM("queryfunction-plane_coeffsize : " <<plane_coeffset.size());
        bool IsInside = true;
        double val_=0.0;
        gPoint normal_;
        for(size_t i(0);i<plane_coeffset.size();i++)
        {
            normal_.x = plane_coeffset[i][0];
            normal_.y = plane_coeffset[i][1];
            normal_.z = plane_coeffset[i][2];
            //ax+by+cz+d 
            val_ = dotProduct_Point(normal_, query)+plane_coeffset[i][3];
            if (val_<0)
                return false; //outside frustum
        }
        return true;       //insde Frustum
    }

    ~Frustum()
    {
        plane_coeffset.clear();
    }

 

};




typedef struct
{
    int32_t x;
    int32_t y;
    double z;
} PixelCoords;

void bBoxCb(const darknet_ros_msgs::BoundingBoxesConstPtr& input_detection)
{
    current_boxes_ = *input_detection;
    received_first_message_bbox =true;
}


void foVRegionCb(const pointcloud_processing_msgs::fov_positionsConstPtr& fov_region_)
{
    //ROS_INFO("fov_regions callback");
    current_fov=*fov_region_;
    received_fov_region=true;
}


void cameraInfoCb(const sensor_msgs::CameraInfoConstPtr msg)
{
    camera_info = *msg;
    received_camera_info=true;
}

void odomCb(const nav_msgs::OdometryConstPtr msg)
{
    received_robot_pose= true;
    robot_pose.header.stamp = ros::Time::now();
    robot_pose.header.frame_id = "walrus/map";
    robot_pose.pose=msg->pose.pose;
}


inline PointType normalizePoint(PointType input)
{    
    const double denominator = std::sqrt(input.x * input.x +
                                         input.y * input.y +
                                         input.z * input.z);

    input.x /= denominator;
    input.y /= denominator;
    input.z /= denominator;

    return input;
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

    const PointType norm_point = normalizePoint(point);
    double dist_point =sqrt(pow(point.x,2)+pow(point.y,2)+pow(point.z,2));

    result.x = camera_info.K[0]*norm_point.x + camera_info.K[2]*norm_point.z;
    result.y = camera_info.K[4]*norm_point.y + camera_info.K[5]*norm_point.z;
    //result.z = norm_point.z;
    result.z = dist_point; //mk

    //ROS_INFO_STREAM("Cartesian: " << point);
    //ROS_INFO_STREAM("Pixelspace: " << result.x << " " << result.y << " " << result.z);

    return result;
}


/**
 * @brief Convert bounding box vertices( pixel coordinates ) into point in the camera frame 
 * 
 * @param points The cartesian point to convert
 * @param depth  minimum depth of point cloud data / 
 * @param box    bounding box info from object recognition algorithm
 * @param camera_info A 3x3 matrix intrinsic to the camera.
 * 
 * @return an array of bounding box points in the camera frame  
 * sequence (xmin, ymin)-d1,d2,  (xmax, ymin)-d1,d2,  (xmin, ymax)-d1,d2 (xmax, ymax)-d1,d2
 */

void  PixelToPose(std::vector<geometry_msgs::Point>& points, double depth_, 
                              const darknet_ros_msgs::BoundingBox & box,
                              const sensor_msgs::CameraInfo &camera_info)
{
    points.clear();
    geometry_msgs::Point point3d;
    double depth_max = depth_+7.5;
    //const PointType norm_point = normalizePoint(point);
    //ROS_INFO_STREAM("b_xmin: "<<box.xmin <<", b_ymin: "<<box.ymin<<std::endl;);
    //ROS_INFO_STREAM("K[0]: "<<camera_info.K[0] <<", K[2]: "<<camera_info.K[2]<<std::endl;);
    //ROS_INFO_STREAM("b_w_x: "<<point3d.x <<"b_w_y: "<<point3d.y<<std::endl;);
    point3d.x = depth_/camera_info.K[0]*(box.xmin-camera_info.K[2]);
    point3d.y = depth_/camera_info.K[4]*(box.ymin-camera_info.K[5]);
    point3d.z = depth_;
    points.push_back(point3d);
    point3d.z = depth_max;
    points.push_back(point3d);
    point3d.x = depth_/camera_info.K[0]*(box.xmax-camera_info.K[2]);
    point3d.y = depth_/camera_info.K[4]*(box.ymin-camera_info.K[5]);
    point3d.z = depth_;
    points.push_back(point3d);
    point3d.z = depth_max;
    points.push_back(point3d);
    point3d.x = depth_/camera_info.K[0]*(box.xmin-camera_info.K[2]);
    point3d.y = depth_/camera_info.K[4]*(box.ymax-camera_info.K[5]);
    point3d.z = depth_;
    points.push_back(point3d);
    point3d.z = depth_max;
    points.push_back(point3d);
    point3d.x = depth_/camera_info.K[0]*(box.xmax-camera_info.K[2]);
    point3d.y = depth_/camera_info.K[4]*(box.ymax-camera_info.K[5]);
    point3d.z = depth_;
    points.push_back(point3d);
    point3d.z = depth_max;
    points.push_back(point3d);

    return;
}

void addbboxlines(visualization_msgs::Marker& bbox_lines, std::vector<geometry_msgs::Point>& bboxpoints)
{

    for(const geometry_msgs::Point p_ : bboxpoints)
    {
        bbox_lines.points.push_back(p_);
    }


    bbox_lines.points.push_back(bboxpoints[0]);
    bbox_lines.points.push_back(bboxpoints[2]);

    bbox_lines.points.push_back(bboxpoints[0]);
    bbox_lines.points.push_back(bboxpoints[4]);

    bbox_lines.points.push_back(bboxpoints[2]);
    bbox_lines.points.push_back(bboxpoints[6]);

    bbox_lines.points.push_back(bboxpoints[4]);
    bbox_lines.points.push_back(bboxpoints[6]);

    bbox_lines.points.push_back(bboxpoints[1]);
    bbox_lines.points.push_back(bboxpoints[3]);

    bbox_lines.points.push_back(bboxpoints[1]);
    bbox_lines.points.push_back(bboxpoints[5]);

    bbox_lines.points.push_back(bboxpoints[3]);
    bbox_lines.points.push_back(bboxpoints[7]);

    bbox_lines.points.push_back(bboxpoints[5]);
    bbox_lines.points.push_back(bboxpoints[7]);


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
        //ROS_ERROR("getObjectID() - No class ID found for name %s", class_name.c_str());
        //std::cerr << e.what() << '\n';
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


/*
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

    CloudPtr cloud_in_fov(new Cloud);
    pcl::ExtractIndices<PointType> camera_fov_filter;

    // Extract the inliers  of the ROI
    camera_fov_filter.setInputCloud(input);
    camera_fov_filter.setIndices(indices_in_bbox);
    camera_fov_filter.setNegative(false);
    camera_fov_filter.filter(*cloud_in_fov);

    if (debug_lidar_viz) {
        sensor_msgs::PointCloud2 pc2;
        pcl::toROSMsg(*cloud_in_fov, pc2);
        lidar_fov_pub_.publish(pc2);
    }

    return cloud_in_fov;
}
*/

/**
 * @brief Extract from a pointcloud those points that are within the FoV of the camera.
 * @param input The input pointcloud
 * @param pixel_coordinates A vector of pixelspace coordinates. These correspond by index
 *                          with the points ins input
 * @param height The pixel height of the camera
 * @param width The pixel width of the camera
 * @return A pointcloud containing only the points within the camera FoV.
 */
CloudPtr filterPointsInFoV(const CloudPtr input,
                           const std::vector<PixelCoords> &pixel_coordinates,
                           const int height,
                           const int width)
{
    pcl::PointIndices::Ptr indices_in_fov(new pcl::PointIndices());
    indices_in_fov->indices.reserve(input->size());

    for (int i = 0; i < pixel_coordinates.size(); ++i)
    {
        if (pixel_coordinates[i].z > 0 &&
            pixel_coordinates[i].x >= 0 &&
            pixel_coordinates[i].x <= width &&
            pixel_coordinates[i].y >= 0 &&
            pixel_coordinates[i].y <= height)
        {
            indices_in_fov->indices.push_back(i);
        }
    }

    CloudPtr cloud_in_fov(new Cloud);
    pcl::ExtractIndices<PointType> camera_fov_filter;

    // Extract the inliers  of the ROI
    camera_fov_filter.setInputCloud(input);
    camera_fov_filter.setIndices(indices_in_fov);
    camera_fov_filter.setNegative(false);
    camera_fov_filter.filter(*cloud_in_fov);

    return cloud_in_fov;
}


CloudPtr filterPointsInFrustum(const CloudPtr input, std::vector<gPoint> bboxpoints)
{
    pcl::PointIndices::Ptr indices_in_volume(new pcl::PointIndices());
    indices_in_volume->indices.reserve(input->size());
    //indices_in_fov->indices.reserve(input->size());
    Frustum boxvolume(bboxpoints);
    ROS_INFO_STREAM("frustrum created");
    for(int k(0);k<input->points.size();k++)
    {
        gPoint query_p;
        query_p.x = input->points[k].x;
        query_p.y = input->points[k].y;
        query_p.z = input->points[k].z;
        if(boxvolume.querypointInFrustum(query_p))
        {
            ROS_INFO_STREAM("k--"<<k);
            indices_in_volume->indices.push_back(k);
        }
    }
    if(indices_in_volume->indices.size()<1)
            ROS_INFO_STREAM("no points in volume");

    
    ROS_INFO_STREAM("actual filter");
    CloudPtr cloud_in_fov(new Cloud);
    pcl::ExtractIndices<PointType> camera_fov_filter;

    ROS_INFO_STREAM("actual filter-2");
    // Extract the inliers  of the ROI
    camera_fov_filter.setInputCloud(input);
    camera_fov_filter.setIndices(indices_in_volume);
    camera_fov_filter.setNegative(false);
    camera_fov_filter.filter(*cloud_in_fov);


    return cloud_in_fov;
}




CloudPtr filterPointsInBox(const CloudPtr input,
                           const std::vector<PixelCoords> &pixel_coordinates,
                           const int xmin,
                           const int xmax,
                           const int ymin,
                           const int ymax,
                           double &min_depth_)
{
    pcl::PointIndices::Ptr indices_in_bbox(new pcl::PointIndices());
    indices_in_bbox->indices.reserve(input->size());

    //ROS_INFO_STREAM("BBox: " << xmin << " " << xmax << " " << ymin << " " << ymax);

    min_depth_ =100.0;

    for (int i = 0; i < pixel_coordinates.size(); ++i)
    {
        //ROS_INFO_STREAM("Coords: " << pixel_coordinates[i].x << " " << pixel_coordinates[i].y << " " << pixel_coordinates[i].z);
        if (pixel_coordinates[i].x > xmin &&
            pixel_coordinates[i].x < xmax &&
            pixel_coordinates[i].y > ymin &&
            pixel_coordinates[i].y < ymax &&
            pixel_coordinates[i].z > 0)
        {
            //ROS_INFO_STREAM("Coords: " << pixel_coordinates[i].x << " " << pixel_coordinates[i].y << " " << pixel_coordinates[i].z);
            indices_in_bbox->indices.push_back(i);
        }

        if(pixel_coordinates[i].z<min_depth_)
            min_depth_=pixel_coordinates[i].z;
    }
    //ROS_INFO_STREAM("num bbox indices: " << indices_in_bbox->indices.size());
    //ROS_INFO_STREAM("mininum depth: " << min_depth_);

    indices_in_bbox->indices.shrink_to_fit();
    CloudPtr cloud_in_bbox(new Cloud);
    pcl::ExtractIndices<PointType> camera_fov_filter;

    // Extract the inliers  of the ROI
    camera_fov_filter.setInputCloud(input);
    camera_fov_filter.setIndices(indices_in_bbox);
    camera_fov_filter.setNegative(false);
    camera_fov_filter.filter(*cloud_in_bbox);

    if (debug_lidar_viz) {
        sensor_msgs::PointCloud2 pc2;
        pcl::toROSMsg(*cloud_in_bbox, pc2);
        lidar_bbox_pub_.publish(pc2);
    }

    return cloud_in_bbox;
}


bool register_scene(pointcloud_processing_msgs::register_scene::Request &req, pointcloud_processing_msgs::register_scene::Response &res)
{
    firstly_called=true;
    geometry_msgs::TransformStamped transform_pcl;
    sensor_msgs::PointCloud2 cloud_in;
    sensor_msgs::PointCloud2 cloud_out;
    geometry_msgs::PointStamped point_in;
    geometry_msgs::PointStamped point_out;
    std::vector<geometry_msgs::Point> registered_bboxpoints_temp;
    tf2_ros::TransformListener tf2_listener(*pbuffer);

    //current bounding box =>bboxpoints == registered_bboxpoints_temp
     for(int i(0);i<bboxpoints.size();i++ )
        {
            ROS_INFO_STREAM("bboxpoint size: "<<bboxpoints.size());
            //transform to the map frame
            try{
                point_in.point = bboxpoints[i];
                point_in.header.frame_id = "walrus/realsense_front_color_optical_frame";
                point_in.header.stamp = ros::Time(0);
                point_out = pbuffer->transform(point_in, "walrus/map");
                registered_bboxpoints_temp.push_back(point_out.point);
                registered_bbox_vertices.points.push_back(point_out.point); //save box vertices for visualization
                //registered_bboxpoints.push_back(point_out.point);
                //ROS_INFO_STREAM("point_out. x: "<<point_out.point.x <<"point_out.y : " <<point_out.point.y<<std::endl;);
            }

            catch (tf2::TransformException ex){
                ROS_ERROR("%s",ex.what());
            }
        }

    CloudPtr cloud_(new Cloud);
    if(!first_registered )
    {
        //filter the current registered pc using new bounding box w.r.t map frame 
        pcl::fromROSMsg(registered_pcl_filtered, *cloud_);
        const CloudPtr cloud_in_bbox_filtered =filterPointsInFrustum(cloud_, registered_bboxpoints_temp); //map_frame
        pcl::toROSMsg(*cloud_in_bbox_filtered , registered_pcl_filtered); //save filted pcl to "registered_pcl_filtered"
    }

    //register point_cloud points in fov w.r.t world frame 
   try{
       //register current fov pointcloud
       if(bool_cloud_fov)
       {
           pcl::toROSMsg(*cur_cloud_bbox, cloud_in);
           CloudFOVbag.clusters.push_back(cloud_in);
       }

       //tranfrom current pcl to map frame
       transform_pcl = pbuffer->lookupTransform("walrus/map", "walrus/realsense_front_color_optical_frame", ros::Time(0), ros::Duration(2.0));
       pcl::PCLPointCloud2 pcl_pc2;
       pcl::toROSMsg(*cur_cloud_bbox, cloud_in);
       tf2::doTransform(cloud_in, cloud_out, transform_pcl);
       Cloudbag.clusters.push_back(cloud_out);
       //save point cloud w.r.t map frame
       if(first_registered)
       {
           registered_pcl_filtered= cloud_out;
           first_registered=false;
       }
       else //filter incoming pointcloud with registered_boxes
       {
           //convert current registered_pcl_map(sensor_msgs::PoinCloud2) to pcl pointcloud and pclPointcloud2
           pcl::PCLPointCloud2 pcl2_registered;
           CloudPtr cloud_registered(new Cloud);
           pcl::fromROSMsg(registered_pcl_filtered, *cloud_registered);
           pcl::toPCLPointCloud2( *cloud_registered,pcl2_registered);

           //convert current pocintlcoud_fov(map frame) to  pcl pointcloud : cloud_fov_map
           CloudPtr cloud_fov_map_(new Cloud);
           pcl::fromROSMsg(cloud_out, *cloud_fov_map_);
           pcl::PCLPointCloud2 pcl2_fov;
           for(size_t j(0); j<registered_bboxpoints.size();j++ )
           {
               ROS_INFO_STREAM("j: "<<j);
               cloud_fov_map_=filterPointsInFrustum(cloud_fov_map_, registered_bboxpoints[j]); //map_frame
               //ROS_INFO_STREAM("size of pointcloud after filterwithFrustum: " <<cur_cloud_bbox->points.size());
               pcl::toPCLPointCloud2( *cloud_fov_map_,pcl2_fov);
               pcl::concatenatePointCloud(pcl2_registered, pcl2_fov,pcl2_registered); //this function requires PCLPointcloud2 format
           }
           pcl::fromPCLPointCloud2(pcl2_registered,*cloud_fov_map_); //convert
           pcl::toROSMsg(*cloud_fov_map_, registered_pcl_filtered); //save filted pcl to "registered_pcl_filtered"
           //ROS_INFO_STREAM("concatenate done");
       }

       robot_poses.push_back(robot_pose);
         //register bounding box points in world frame 
       registered_bboxpoints.push_back(registered_bboxpoints_temp);
        addbboxlines(registered_bbox_lines, registered_bboxpoints_temp);


           
       res.is_registered=true;
       ROS_INFO("service done");
       return true;
   }
   catch (tf2::TransformException ex){
       ROS_ERROR("%s",ex.what());

       res.is_registered=false;
       ROS_INFO("service failed");
       return false;
   }


    // Merge metadata
    // MergedCloud.width += Cloud1.width;
    //
    // // Re-size the merged data array to make space for the new points
    // uint64_t OriginalSize = MergedCloud.data.size();
    // MergedCloud.data.resize(MergedCloud.data.size() + Cloud1.data.size());
    //
    // // Copy the new points from Cloud1 into the second half of the MergedCloud array
    // std::copy(
    //   Cloud1.data.begin(),
    //     Cloud1.data.end(),
    //       MergedCloud.data.begin() + OriginalSize);
    //
    ////merge
       /*
       registered_pcl.header= cloud_out.header;
       registered_pcl.header.frame_id = "walrus/map";
       if(first_registered)
       {
           registered_pcl=cloud_out;
           ROS_INFO("firstly registered- size: %d ", registered_pcl.data.size());
           first_registered=false;
       }
       else{
           sensor_msgs::PointCloud2 cloud_tmp;
           cloud_tmp=registered_pcl;
           uint32_t OriginalSize = registered_pcl.data.size();
           registered_pcl.width = registered_pcl.data.size()+cloud_out.width;
           ROS_INFO("original size: %d, new_pcl_size: %d, total_width: %d", OriginalSize, cloud_out.width, registered_pcl.width);
           registered_pcl.data.resize(registered_pcl.width);
           //std::copy(
                   //cloud_tmp.data.begin(),
                   //cloud_tmp.data.end(),
                   //registered_pcl.data.begin());
           for(int i(0);i<cloud_tmp.width;i++)
                   registered_pcl.data[i]=cloud_tmp.data[i];

           for(int i(0);i<cloud_out.width;i++)
                   registered_pcl.data[i+OriginalSize]=cloud_out.data[i];
               
           //std::copy(
                   //cloud_out.data.begin(),
                   //cloud_out.data.end(),
                   //registered_pcl.data.begin() + OriginalSize);
       }
       */
       //uint64_t OriginalSize = registered_pcl.data.size();
       //ROS_INFO("original size: %d, new_pcl_size: %d, total_width: %d", OriginalSize, cloud_out.width, registered_pcl.width);
       //registered_pcl.data.resize(registered_pcl.data.size() + cloud_out.data.size());
       //registered_pcl.data.resize(registered_pcl.width);
       //std::copy(
               //cloud_out.data.begin(),
               //cloud_out.data.end(),
               //registered_pcl.data.begin() + OriginalSize);

       //registered_pcl.is_dense=true;
       //pcl_map_pub.publish(registered_pcl);

    return true;
}


void segmentationpointClouds()
{
    if (Cloudbag.clusters.size()>0)
    {
      segmentbag.clusters.clear();
      for(int i(0);i<Cloudbag.clusters.size();i++)
      {
          CloudPtr cloud(new Cloud);
          pcl::fromROSMsg(Cloudbag.clusters[i], *cloud);

          // ----------------------Voxel Downsampling----------------------------------
          CloudPtr cloud_downsampled(new Cloud);
          pcl::VoxelGrid<PointType> sor_voxelgrid;
          sor_voxelgrid.setInputCloud(cloud);
          sor_voxelgrid.setLeafSize(0.03, 0.03, 0.03); //size of the grid
          sor_voxelgrid.filter(*cloud_downsampled);

          //---------------------StatisticalOutlierRemoval--------------------
          // Remove statistical outliers from the downsampled cloud
          CloudPtr cloud_denoised(new Cloud);
          pcl::StatisticalOutlierRemoval<PointType> sor_noise;
          sor_noise.setInputCloud(cloud_downsampled);
          sor_noise.setMeanK(50);
          sor_noise.setStddevMulThresh(1.5);
          sor_noise.filter(*cloud_denoised);

          /*
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_inplane        (new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_outplane       (new pcl::PointCloud<pcl::PointXYZRGB>);


           pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
           pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices ());

          // Create the segmentation object
          pcl::SACSegmentation<pcl::PointXYZRGB> seg;
          // Optional
          seg.setOptimizeCoefficients (true);
          // Mandatory
          seg.setModelType (pcl::SACMODEL_PLANE);
          seg.setMethodType (pcl::SAC_RANSAC);
          seg.setDistanceThreshold (0.1);
          //seg.setRadiusLimits(0, 0.2);

          seg.setInputCloud (cloud_denoised);
          seg.segment (*inliers_plane, *coefficients);

          // Create the filtering object
          pcl::ExtractIndices<pcl::PointXYZRGB> extract_plane;

          // Extract the inliers
          extract_plane.setInputCloud (cloud_denoised);
          extract_plane.setIndices (inliers_plane);
          extract_plane.setNegative (false);
          extract_plane.filter (*cloud_filtered_inplane);
            
      // Extract the outliers

          extract_plane.setNegative (true);
          extract_plane.filter (*cloud_filtered_outplane);

          pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud_filtered = new pcl::PointCloud<pcl::PointXYZRGB>;
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrFiltered (xyz_cloud_filtered);

          //pcl::PassThrough<pcl::PointXYZRGB> pass;
          //pass.setInputCloud (cloud_filtered_inplane);
          //pass.setFilterFieldName ("x");
          //pass.setFilterLimits (0.2, 10.0);
          //pass.setFilterLimitsNegative (true);
          //pass.filter(*xyzCloudPtrFiltered);
          */

          //cur_cloud=xyzCloudPtrFiltered;
           pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal> ());
          // Normal estimation

          pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
          ne.setInputCloud (cloud_downsampled);

          pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_n (new pcl::search::KdTree<pcl::PointXYZRGB>());
          ne.setSearchMethod (tree_n);
          ne.setRadiusSearch (0.1);
          ne.compute (*cloud_normals);

          // Creating the kdtree object for the search method of the extraction
          pcl::KdTree<pcl::PointXYZRGB>::Ptr tree_ec  (new pcl::KdTreeFLANN<pcl::PointXYZRGB> ());
          tree_ec->setInputCloud (cloud_downsampled);
          
          // Extracting Euclidean clusters using cloud and its normals
          std::vector<pcl::PointIndices> cluster_indices_;
          const float tolerance = 0.5f; // 50cm tolerance in (x, y, z) coordinate system
          const double eps_angle = 30* (M_PI / 180.0); // 30 degree tolerance in normals
          const unsigned int min_cluster_size = 150;
         
          pcl::extractEuclideanClusters (*cloud_downsampled, *cloud_normals, tolerance, tree_ec, cluster_indices_, eps_angle, min_cluster_size);
          std::cout<<"segmentation size: "<<cluster_indices_.size()<<std::endl;
          //seg_idxset.push_back(cluster_indices_.size());

         // Saving the clusters in separate pcd files
          int j = 0;
          pcl::PCLPointCloud2 outputPCL;
          sensor_msgs::PointCloud2 output_tmp;
          geometry_msgs::PoseArray seg_poses;
          seg_poses.header.stamp=ros::Time::now();
          seg_poses.header.frame_id=Cloudbag.clusters[0].header.frame_id;
          for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices_.begin (); it != cluster_indices_.end (); ++it)
          {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
            for (const auto &index : it->indices)
              cloud_cluster->push_back ((*cloud_downsampled)[index]); 

           //pcl::PointCloud<pcl::PointXYZRGB>::Ptr nanfiltered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
           //std::vector<int> rindices;
           //pcl::removeNaNFromPointCloud(*cloud_cluster,*nanfiltered_cloud, rindices);

            cloud_cluster->width = cloud_cluster->size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;


            pcl::toPCLPointCloud2( *cloud_cluster,outputPCL);
            Eigen::Vector4f centroid_tmp;
            geometry_msgs::Pose centroid_pos;
            pcl::compute3DCentroid(*cloud_cluster, centroid_tmp); 
            centroid_pos.position.x=centroid_tmp[0];
            centroid_pos.position.y=centroid_tmp[1];
            centroid_pos.position.z=centroid_tmp[2];
            centroid_pos.orientation.w=1.0;
            seg_poses.poses.push_back(centroid_pos);
        
         
            pcl_conversions::fromPCL(outputPCL, output_tmp);
            output_tmp.header= seg_poses.header;
            //output_tmp.header.stamp=ros::Time::now();
            //output_tmp.header.frame_id=Cloudbag.clusters[0].header.frame_id;
            segmentbag.clusters.push_back(output_tmp);
            j++;
          }
          seg_poses_map.insert({i, seg_poses});

      }


/*
      // Create the KdTree object for the search method of the extraction
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
      //tree->setInputCloud (xyzCloudPtrFiltered);
      tree->setInputCloud (cloud_denoised);



	  // create the extraction object for the clusters
	  std::vector<pcl::PointIndices> cluster_indices;
	  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	  // specify euclidean cluster parameters
	  ec.setClusterTolerance (0.1); // 10cm
	  ec.setMinClusterSize (200);
	  ec.setMaxClusterSize (30000);
	  ec.setSearchMethod (tree);
	  ec.setInputCloud (cloud_denoised);
	  // exctract the indices pertaining to each cluster and store in a vector of pcl::PointIndices
	  ec.extract (cluster_indices);

      sensor_msgs::PointCloud2 output_tmp;
      pcl::PCLPointCloud2 outputPCL;
	  pointcloud_processing_msgs::ClusterArray CloudClusters;

      int cnt=0;
      int min_idx = 0;
	  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {

        // create a new clusterData message object
        // create a pcl object to hold the extracted cluster
        pcl::PointCloud<pcl::PointXYZRGB> *cluster = new pcl::PointCloud<pcl::PointXYZRGB>;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterPtr (cluster);

        // now we are in a vector of indices pertaining to a single cluster.
        // Assign each point corresponding to this cluster in xyzCloudPtrPassthroughFiltered a specific color for identification purposes
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          clusterPtr->points.push_back(xyzCloudPtrFiltered->points[*pit]);

        }
        // convert to pcl::PCLPointCloud2
        pcl::toPCLPointCloud2( *clusterPtr ,outputPCL);
        Eigen::Vector4f centroid_tmp;
        pcl::compute3DCentroid(*clusterPtr, centroid_tmp); 
        // Convert to ROS data type
        pcl_conversions::fromPCL(outputPCL, output_tmp);
        output_tmp.header.frame_id=Cloudbag.clusters[0].header.frame_id;

        // add the cluster to the array message
        //clusterData.cluster = output;
        CloudClusters.clusters.push_back(output_tmp);
        if(cnt==0)
        {
            seg1_pub.publish(output_tmp);
        }
        else if(cnt==1)
        {
            seg2_pub.publish(output_tmp);
        }
        else if(cnt==2)
        {
            seg3_pub.publish(output_tmp);
        }
        else{
        
            ROS_INFO("more than 3 clusters");
        
        }
        cnt++;
      }
      //CloudPtr cloud_min(new Cloud);
      Eigen::Vector4f centroid_seg;
      clusters_pub.publish(CloudClusters);

      //sensor_msgs::PointCloud2 min_output;
      //pcl::toROSMsg(CloudClusters.clusters[min_idx], min_output);
      //pcl::fromROSMsg(CloudClusters.clusters[min_idx], *cloud_min);
      //pcl::compute3DCentroid(*cloud_min, centroid_seg); 
      //geometry_msgs::PoseStamped pcl_center;
        //pcl_center.header.frame_id = input_cloud->header.frame_id;
        //pcl_center.pose.position.x= min_x;
        //pcl_center.pose.position.y= min_y;
        //pcl_center.pose.position.z= min_z;
        //pcl_center.pose.orientation.w = 1;
        //pcl_center_pub.publish(pcl_center);


      //sensor_msgs::PointCloud2 output;
      //sensor_msgs::PointCloud2 output2;
      //pcl::toROSMsg(*cloud_filtered_inplane, output);
      //pcl::toROSMsg(*xyzCloudPtrFiltered, output2);

       //Set output frame as the input frame
      //output.header.frame_id=input_cloud->header.frame_id;
      //output2.header.frame_id=input_cloud->header.frame_id;

       //Publish output
      //inplane_pub.publish (output);
      //outplane_pub.publish (output2);
    

    //sensor_msgs::PointCloud2 pc2_temp;
    //pcl::toROSMsg(*cloud_denoised, pc2_temp);
    //pc2_temp.header.frame_id = input_cloud->header.frame_id;
    //lidar_temp_pub_.publish(pc2_temp);

    //Eigen::Vector4f centroid_out;
    //pcl::compute3DCentroid(*cloud_denoised, centroid_out); 

    // add to the output
    
    */
    }

}

bool calculate_overlapedregion()
{


}

bool calculate_targetposes(pointcloud_processing_msgs::get_targetpose::Request &req, pointcloud_processing_msgs::get_targetpose::Response &res)
{

    CloudPtr cloud_(new Cloud);
    pcl::fromROSMsg(registered_pcl_filtered, *cloud_);
    Eigen::Vector4f centroid_out;
    pcl::compute3DCentroid(*cloud_, centroid_out); 

    // add to the output
    geometry_msgs::PoseStamped pcl_center;
    pcl_center.header.frame_id = "walrus/map";
    pcl_center.header.stamp = ros::Time::now();
    pcl_center.pose.position.x= centroid_out[0];
    pcl_center.pose.position.y= centroid_out[1];
    pcl_center.pose.position.z= centroid_out[2];
    pcl_center.pose.orientation.w = 1;
    previous_target_pose = pcl_center;
    target_obtained=true;

    res.target_pose= pcl_center;



    return true;
}


//get target poses from dataset
bool calculate_targetposes_old(pointcloud_processing_msgs::get_targetpose::Request &req, pointcloud_processing_msgs::get_targetpose::Response &res)
{
    segmentationpointClouds();
    std::map<int, geometry_msgs::PoseArray> opt_poses;
    
    //among segmentation --get the overlapped case
    std::map< int, std::vector< std::vector<double> > > score_mapV;
    //if there is only one registered, take the closest segment from the robot
    if(seg_poses_map.size()<1)
    {
        return false;
    }
    else if(seg_poses_map.size()==1)
    {
        //when there is only one snapshot, we will chooose the closest segmentation from the robot
        PoseMapIter it = seg_poses_map.begin();
        std::vector<geometry_msgs::PoseStamped>::iterator piter= robot_poses.begin();
        std::vector<double> dist_set;
        dist_set.resize(it->second.poses.size());
        double temp_dist=0.0;
        for(int j(0); j< it->second.poses.size();j++)
        {
            geometry_msgs::Pose rpose=(*piter).pose;
            temp_dist=pow((rpose.position.x-(it->second).poses[j].position.x),2);
            temp_dist+=pow((rpose.position.y-(it->second).poses[j].position.y),2);
            temp_dist=sqrt(temp_dist);
            dist_set[j]=temp_dist;
        }
        auto minIt = std::min_element(dist_set.begin(), dist_set.end());
        int min_idx = minIt-dist_set.begin();
        target_pose.header.stamp = ros::Time::now(); 
        //target_pose.header.frame_id = it->second.header.frame_id;
        target_pose.header.frame_id = "walrus/map";
        target_pose.pose=it->second.poses[min_idx];
        res.target_pose=target_pose;
        pcl_center_pub.publish(target_pose);
        previous_target_pose.header.frame_id = "walrus/map";
        previous_target_pose.pose=target_pose.pose;
        previous_target_pose.header=target_pose.header;
        target_obtained=true;
        return true;
    }
    else{
        //take the first segmented group as a baseline, 
        PoseMapIter fit = seg_poses_map.begin();
        int pose_size= fit->second.poses.size(); //first registered segments
        PoseMapIter it = seg_poses_map.begin();
        for(it ; it!=seg_poses_map.end(); it++)
        {
            if(it==seg_poses_map.begin())
            {
                if(firstly_called) //only save the opt_pose
                {
                    for(int i(0);i<it->second.poses.size();i++)
                    {
                        geometry_msgs::PoseArray opt_pos;
                        opt_pos.poses.push_back(it->second.poses[i]);
                        opt_poses.insert({i,opt_pos});
                    }
                    firstly_called=false;
                }
                else
                {
                    continue;
                }
            }
            else
            {
                //std::vector< std::vector<double> > score_map;
                //score_map.resize(pose_size);
                for(int k(0);k<pose_size;k++)//criteria pose
                {
                    double min_dist=100.0;
                    int min_index=0;
                    for(int j(0); j< it->second.poses.size();j++) //current pose from segmented
                    {
                        double temp_dist=0.0;
                        
                        geometry_msgs::Pose spose=fit->second.poses[k];
                        temp_dist=pow((spose.position.x-(it->second).poses[j].position.x),2);
                        temp_dist+=pow((spose.position.y-(it->second).poses[j].position.y),2);
                        temp_dist=sqrt(temp_dist);

                        if(temp_dist <min_dist)
                        {
                            min_dist=temp_dist;
                            min_index=j;
                        }
                        //score_map[k].push_back(temp_dist);
                    }
                    if(min_dist<MIN_DIST)//when the segmation is within a range from the previous iteration
                    {
                        ROS_INFO("added K: %d, J: %d", k, min_index);
                        std::map<int,geometry_msgs::PoseArray>::iterator it_ = opt_poses.find(k);
                        it_->second.poses.push_back(it->second.poses[min_index]);
                        res.target_pose.pose = it->second.poses[min_index];
                        target_pose.header.stamp = ros::Time::now(); 
                        //target_pose.header.frame_id = "walrus/map"it->second.header.frame_id;
                        target_pose.header.frame_id = "walrus/map";
                        target_pose=res.target_pose;
                        previous_target_pose.pose=target_pose.pose;
                        previous_target_pose.header=target_pose.header;
                        previous_target_pose.header.frame_id = "walrus/map";
                        pcl_center_pub.publish(target_pose);
                        target_obtained=true;
  
                    }
                }
                //score_mapV.insert({it->first, score_map});
            }
        }
    }

}

bool transformPointCloud2(sensor_msgs::PointCloud2ConstPtr cloud_ptr, sensor_msgs::PointCloud2 &pointcloud,
                          const std::string target_frame)
{
    geometry_msgs::TransformStamped transform;
    try
    {
      transform = pbuffer->lookupTransform(target_frame,
                                             tf2::getFrameId(*cloud_ptr),
                                             ros::Time(0),
                                             ros::Duration(0.1));
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("%s",ex.what());
      return false;
    }

    tf2::doTransform(*cloud_ptr, pointcloud, transform);
    
    return true;
}

void pointCloudCb(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
{
    received_first_message_cloud = true;
    const ros::Time now = ros::Time::now();
    double confidence_threshold;
    nh->param("confidence_threshold", confidence_threshold, 0.75);

    //ROS_INFO("cloud callback");
    received_first_message = true;

    // Initialize containers for clouds
    CloudPtr cloud(new Cloud);
    CloudPtr cloud_target(new Cloud);

    // Initialize container for object poses
    geometry_msgs::PoseArray target_poses;
    target_poses.header.stamp = now;
    target_poses.header.frame_id = input_cloud->header.frame_id;

    // Initialize container for centroids' markers
    visualization_msgs::Marker bbox_vertices, bbox_lines;

    bbox_vertices.header.frame_id = bbox_lines.header.frame_id = input_cloud->header.frame_id;
    bbox_vertices.header.stamp = bbox_lines.header.stamp = ros::Time::now();
    bbox_vertices.type = visualization_msgs::Marker::SPHERE_LIST;
    bbox_vertices.color.a = 1.0;
    bbox_vertices.color.r = 1.0;
    bbox_vertices.action = visualization_msgs::Marker::ADD;

    bbox_vertices.scale.x = 0.5;
    bbox_vertices.scale.y = 0.5;
    bbox_vertices.scale.z = 0.5;

    bbox_lines.type = visualization_msgs::Marker::LINE_LIST;
    bbox_lines.color.a = 1.0;
    bbox_lines.color.r = 1.0;
    bbox_lines.action = visualization_msgs::Marker::ADD;

    bbox_lines.scale.x = 0.15;



    // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
    pcl::fromROSMsg(*input_cloud, *cloud);
    cur_cloud=cloud;

    sensor_msgs::PointCloud2 output_cloud;

    //process for saving pcl in optical frame
    // transform the pointcloud into the RGB optical frame
    if (tf2::getFrameId(*input_cloud) != rgb_optical_frame_)
    {
        if (!transformPointCloud2(input_cloud, output_cloud, rgb_optical_frame_))
        {
            return;
        }  
    }

    pcl::fromROSMsg(*input_cloud, *cloud);

    //CloudPtr nanfiltered_cloud(new Cloud);
    CloudPtr cloud_nan_filtered(new Cloud);
    std::vector<int> rindices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud_nan_filtered, rindices);

    //current pose and camera paramters --> pixel coordinates
    const std::vector<PixelCoords> pixel_coordinates = convertCloudToPixelCoords(cloud_nan_filtered, camera_info);
    // -------------------Extraction of points in the camera FOV------------------------------
    cur_cloud_fov = filterPointsInFoV(cloud_nan_filtered, pixel_coordinates, camera_info.height, camera_info.width);
    if(cur_cloud_fov!=NULL)
    {
        bool_cloud_fov = true;
    }

    const std::vector<PixelCoords> pixel_coordinates_fov = convertCloudToPixelCoords(cur_cloud_fov, camera_info);
    for(const darknet_ros_msgs::BoundingBox &box : current_boxes_.bounding_boxes)
    {
        const ObjectClassID id = getObjectID(box.Class, object_classes);
        // do we meet the threshold for a confirmed detection?
        if (box.probability >= confidence_threshold && id != UNKNOWN_OBJECT_ID)
        {            
            // ----------------------Extract points in the bounding box-----------
            double min_depth=0.0;
            if(box.Class=="person")
            {
                cur_cloud_bbox = filterPointsInBox(cur_cloud_fov,
                                                             pixel_coordinates_fov,
                                                             box.xmin,
                                                             box.xmax,
                                                             box.ymin,
                                                             box.ymax,
                                                             min_depth);

                //convert current bbox points(pixel coordinates) into world coordinates
                PixelToPose(bboxpoints, min_depth, box, camera_info);

                //create bbox_vertices for visualization
                for(const geometry_msgs::Point p_ : bboxpoints)
                    bbox_vertices.points.push_back(p_);

                //create bbox_edges for visualization
                addbboxlines(bbox_lines, bboxpoints);

                if (debug_lidar_viz)
                {
                    sensor_msgs::PointCloud2 pc2;
                    pcl::toROSMsg(*cur_cloud_bbox, pc2);
                    lidar_bbox_pub_.publish(pc2);
                    //bbox_marker_pub.publish(bbox_vertices);
                    //bboxline_marker_pub.publish(bbox_lines);
                }
            }
            // ----------------------Compute centroid-----------------------------
            //Eigen::Vector4f centroid_out;
            //pcl::compute3DCentroid(*cloud_in_bbox, centroid_out); 

            // add to the output
            //vision_msgs::Detection2D object;
            //object.bbox.center.x = (box.xmax + box.xmin)/2;
            //object.bbox.center.y = (box.ymax + box.ymin)/2;
            //object.bbox.size_x = box.xmax - box.xmin;
            //object.bbox.size_y = box.ymax - box.ymin;

            //vision_msgs::ObjectHypothesisWithPose hypothesis;
            //hypothesis.id = id;
            //hypothesis.score = box.probability;
            //hypothesis.pose.pose.position.x = centroid_out[0];
            //hypothesis.pose.pose.position.y = centroid_out[1];
            //hypothesis.pose.pose.position.z = centroid_out[2];
            //hypothesis.pose.pose.orientation.w = 1;

            //object.results.push_back(hypothesis);

            //detected_objects.detections.push_back(object);
        }
    }





    /*

    // ----------------------Voxel Downsampling----------------------------------
    CloudPtr cloud_downsampled(new Cloud);
    pcl::VoxelGrid<PointType> sor_voxelgrid;
    sor_voxelgrid.setInputCloud(cloud);
    sor_voxelgrid.setLeafSize(0.1, 0.1, 0.1); //size of the grid
    sor_voxelgrid.filter(*cloud_downsampled);

     //---------------------StatisticalOutlierRemoval--------------------
    // Remove statistical outliers from the downsampled cloud
    CloudPtr cloud_denoised(new Cloud);
    pcl::StatisticalOutlierRemoval<PointType> sor_noise;
    sor_noise.setInputCloud(cloud_downsampled);
    sor_noise.setMeanK(50);
    sor_noise.setStddevMulThresh(1.0);
    sor_noise.filter(*cloud_denoised);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_inplane        (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_outplane       (new pcl::PointCloud<pcl::PointXYZRGB>);


       pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
       pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices ());

      // Create the segmentation object
      pcl::SACSegmentation<pcl::PointXYZRGB> seg;
      // Optional
      seg.setOptimizeCoefficients (true);
      // Mandatory
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setDistanceThreshold (0.1);
      //seg.setRadiusLimits(0, 0.2);

      seg.setInputCloud (cloud_denoised);
      seg.segment (*inliers_plane, *coefficients);

      // Create the filtering object
      pcl::ExtractIndices<pcl::PointXYZRGB> extract_plane;

      // Extract the inliers
      extract_plane.setInputCloud (cloud_denoised);
      extract_plane.setIndices (inliers_plane);
      extract_plane.setNegative (false);
      extract_plane.filter (*cloud_filtered_inplane);
        
  // Extract the outliers

      extract_plane.setNegative (true);
      extract_plane.filter (*cloud_filtered_outplane);

      pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud_filtered = new pcl::PointCloud<pcl::PointXYZRGB>;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrFiltered (xyz_cloud_filtered);

      pcl::PassThrough<pcl::PointXYZRGB> pass;
      pass.setInputCloud (cloud_filtered_inplane);
      pass.setFilterFieldName ("x");
      pass.setFilterLimits (0.2, 10.0);
      //pass.setFilterLimitsNegative (true);
      pass.filter(*xyzCloudPtrFiltered);

      //cur_cloud=xyzCloudPtrFiltered;
      */
      /*
	      // Create the KdTree object for the search method of the extraction
	  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
      tree->setInputCloud (xyzCloudPtrFiltered);
	  //tree->setInputCloud (cloud_denoised);

	  // create the extraction object for the clusters
	  std::vector<pcl::PointIndices> cluster_indices;
	  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
	  // specify euclidean cluster parameters
	  ec.setClusterTolerance (0.1); // 10cm
	  ec.setMinClusterSize (200);
	  ec.setMaxClusterSize (20000);
	  ec.setSearchMethod (tree);
	  ec.setInputCloud (xyzCloudPtrFiltered);
	  // exctract the indices pertaining to each cluster and store in a vector of pcl::PointIndices
	  ec.extract (cluster_indices);

      sensor_msgs::PointCloud2 output_tmp;
      pcl::PCLPointCloud2 outputPCL;
	  pointcloud_processing_msgs::ClusterArray CloudClusters;

      int cnt=0;
      double min_depth=100.0;
      double min_x=0.0;
      double  min_y=0.0;
       double  min_z =0.0;
      int min_idx = 0;
	  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {

        // create a new clusterData message object

        // create a pcl object to hold the extracted cluster
        pcl::PointCloud<pcl::PointXYZRGB> *cluster = new pcl::PointCloud<pcl::PointXYZRGB>;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterPtr (cluster);

        // now we are in a vector of indices pertaining to a single cluster.
        // Assign each point corresponding to this cluster in xyzCloudPtrPassthroughFiltered a specific color for identification purposes
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          clusterPtr->points.push_back(xyzCloudPtrFiltered->points[*pit]);

        }

        // convert to pcl::PCLPointCloud2
        pcl::toPCLPointCloud2( *clusterPtr ,outputPCL);
        Eigen::Vector4f centroid_tmp;
        pcl::compute3DCentroid(*clusterPtr, centroid_tmp); 
        if (centroid_tmp[0]<min_depth)
          {
              min_depth = centroid_tmp[0];
              min_x=centroid_tmp[0];
              min_y=centroid_tmp[1];
              min_z=centroid_tmp[2];

              min_idx=cnt;
          }

        // Convert to ROS data type
        pcl_conversions::fromPCL(outputPCL, output_tmp);
        output_tmp.header.frame_id=input_cloud->header.frame_id;

        // add the cluster to the array message
        //clusterData.cluster = output;
        CloudClusters.clusters.push_back(output_tmp);
        if(cnt==0)
        {
            seg1_pub.publish(output_tmp);
        }
        else if(cnt==1)
        {
            seg2_pub.publish(output_tmp);
        }
        else if(cnt==2)
        {
            seg3_pub.publish(output_tmp);
        }
        else{
        
            ROS_INFO("more than 3 clusters");
        
        }
        cnt++;
    

      }
      //double min_depth=100.0;
      //int min_idx = 0;
      
      //CloudPtr cloud_min(new Cloud);
      Eigen::Vector4f centroid_seg;
      //for(int k(0);k<CloudClusters.clusters.size();k++)
      //{
          //pcl::fromROSMsg(CloudClusters.clusters[k], *cloud_min);
          //pcl::compute3DCentroid(*cloud_min, centroid_seg); 
          //if (centroid_seg[0]<min_depth)
          //{
              //min_depth = centroid_seg[0];
              //min_idx=k;
          //}
      //}



      clusters_pub.publish(CloudClusters);

      sensor_msgs::PointCloud2 min_output;
      //pcl::toROSMsg(CloudClusters.clusters[min_idx], min_output);
      //pcl::fromROSMsg(CloudClusters.clusters[min_idx], *cloud_min);
      //pcl::compute3DCentroid(*cloud_min, centroid_seg); 
      geometry_msgs::PoseStamped pcl_center;
        pcl_center.header.frame_id = input_cloud->header.frame_id;
        pcl_center.pose.position.x= min_x;
        pcl_center.pose.position.y= min_y;
        pcl_center.pose.position.z= min_z;
        pcl_center.pose.orientation.w = 1;
        pcl_center_pub.publish(pcl_center);
        */


            /*
      sensor_msgs::PointCloud2 output;
      sensor_msgs::PointCloud2 output2;
      pcl::toROSMsg(*cloud_filtered_inplane, output);
      pcl::toROSMsg(*xyzCloudPtrFiltered, output2);

      // Set output frame as the input frame
      output.header.frame_id=input_cloud->header.frame_id;
      output2.header.frame_id=input_cloud->header.frame_id;

      // Publish output
      inplane_pub.publish (output);
      outplane_pub.publish (output2);
    

    sensor_msgs::PointCloud2 pc2_temp;
    pcl::toROSMsg(*cloud_denoised, pc2_temp);
    pc2_temp.header.frame_id = input_cloud->header.frame_id;
    lidar_temp_pub_.publish(pc2_temp);

    Eigen::Vector4f centroid_out;
    pcl::compute3DCentroid(*cloud_denoised, centroid_out); 
    */

    // add to the output
    //geometry_msgs::PoseStamped pcl_center;
    //pcl_center.header.frame_id = input_cloud->header.frame_id;
    //pcl_center.pose.position.x= centroid_out[0];
    //pcl_center.pose.position.y= centroid_out[1];
    //pcl_center.pose.position.z= centroid_out[2];
    //pcl_center.pose.orientation.w = 1;

    //pcl_center_pub.publish(pcl_center);


    // remove NaN points from the cloud
    
    //ROS_INFO("Pointcloud processed");
}



int
main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "pointcloud_processing");
    //pose_est_srv pose_manager;
    nh = new ros::NodeHandle();

    nh->param("DARKNET_TOPIC", DARKNET_TOPIC, {"/darknet_ros/bounding_boxes"});
    nh->param("RETINANET_TOPIC", RETINANET_TOPIC, {"/retina_ros/bounding_boxes"});
    //nh->param("PCL_TOPIC", PCL_TOPIC, {"/pointcloud_transformer/output_pcl2"});
    //nh->param("PCL_TOPIC", PCL_TOPIC, {"/sep_processing_node/lidar_bbox"});
    nh->param("PCL_TOPIC", PCL_TOPIC, {"/sep_processing_node/lidar_fov"});
    nh->param("FOV_TOPIC", FOV_TOPIC, {"/fov_regions"});
    //nh->param("PCL_TOPIC", PCL_TOPIC, {"/points2"});
    nh->param("TARGET_FRAME", TARGET_FRAME, {"odom"});
    nh->param("TARGET_CLASS", TARGET_CLASS, {"chair"});
    nh->param("VISUAL", VISUAL, {true});
    nh->param("PCL_VISUAL", PCL_VISUAL, {true});
    nh->param("debug_lidar_viz", debug_lidar_viz, {true});

    if (argc != 2)
    {
        ROS_INFO("usage: rosrun pointcloud_processing pose_estimation_service rgb_frame");
        return 1;
    }
    rgb_optical_frame_ = std::string(argv[1]);

    std::map<std::string, std::string> temp_map;
    //if (!nh->hasParam("/sep_processing_node/object_classes"))
    //{
        //ROS_ERROR("Failed to load dictionary parameter 'object_classes'.");
        //return 1;
    //}
    //nh->getParam("/sep_processing_node/object_classes", temp_map);
    temp_map.insert(std::make_pair("person", "1"));


    try
    {
        object_classes = convertClassesMap(temp_map);
    } catch(std::invalid_argument ex)
    {
        ROS_FATAL("Invalid object_classes parameter.");
        return 1;
    }

    // Initialize subscribers to darknet detection and pointcloud
    ros::Subscriber bbox_sub = nh->subscribe<darknet_ros_msgs::BoundingBoxes>(DARKNET_TOPIC, 100, bBoxCb); 
    ros::Subscriber cloud_sub = nh->subscribe<sensor_msgs::PointCloud2>(PCL_TOPIC, 100, pointCloudCb);
    ros::Subscriber camera_info_sub = nh->subscribe<sensor_msgs::CameraInfo>("camera/color/camera_info", 100, cameraInfoCb);
    ros::Subscriber odom_sub= nh->subscribe<nav_msgs::Odometry>("walrus/odom", 100, odomCb);

    ros::ServiceServer register_service = nh->advertiseService("/register_scenes",  register_scene);
    ros::ServiceServer estimation_service = nh->advertiseService("/get_targetpose",  calculate_targetposes);

    // Initialize transform listener
    //tf::TransformListener listener(ros::Duration(20));
    //lst = &listener;
    
    tf2_ros::Buffer tf_buffer(ros::Duration(100));
    pbuffer = &tf_buffer;
    tf2_ros::TransformListener tf2_listener(tf_buffer);
    //tf2_ros::TransformListener tf_listener_;

    // Synchronize darknet detection with pointcloud
    //sync.registerCallback(boost::bind(&cloud_cb, _1, _2));

    // Create a ROS publisher for the output point cloud
    detected_objects_pub = nh->advertise<vision_msgs::Detection2DArray>("detected_objects", 1);
    pcl_center_pub = nh->advertise<geometry_msgs::PoseStamped>("target_center", 1);


    //Create marker for registered box and linesarker registered_bbox_vertices;
    registered_bbox_vertices.header.frame_id = "walrus/map";
    registered_bbox_vertices.header.stamp =  ros::Time::now();
    registered_bbox_vertices.type = visualization_msgs::Marker::SPHERE_LIST;
    registered_bbox_vertices.color.a = 1.0;
    registered_bbox_vertices.color.r = 1.0;
    registered_bbox_vertices.action = visualization_msgs::Marker::ADD;

    registered_bbox_vertices.scale.x = 0.5;
    registered_bbox_vertices.scale.y = 0.5;
    registered_bbox_vertices.scale.z = 0.5;

    registered_bbox_vertices.pose.orientation.w = 1.0;
    registered_bbox_lines.header.frame_id = "walrus/map";
    registered_bbox_lines.header.stamp =  ros::Time::now();
    registered_bbox_lines.type = visualization_msgs::Marker::LINE_LIST;
    registered_bbox_lines.color.a = 1.0;
    registered_bbox_lines.color.r = 1.0;
    registered_bbox_lines.action = visualization_msgs::Marker::ADD;
    registered_bbox_lines.scale.x = 0.15;

    registered_pcl.header.frame_id = "walrus/map";



    if (debug_lidar_viz) {
        lidar_fov_pub_ = nh->advertise<sensor_msgs::PointCloud2>("lidar_fov", 1);
        lidar_temp_pub_ = nh->advertise<sensor_msgs::PointCloud2>("lidar_temp", 1);
        lidar_bbox_pub_ = nh->advertise<sensor_msgs::PointCloud2>("lidar_bbox", 1);

        inplane_pub= nh->advertise<sensor_msgs::PointCloud2> ("pcl_inplane", 1);
        outplane_pub= nh->advertise<sensor_msgs::PointCloud2> ("pcl_outplane", 1);
        clusters_pub= nh->advertise<pointcloud_processing_msgs::ClusterArray> ("segmented_pcl", 1);
        seg1_pub= nh->advertise<sensor_msgs::PointCloud2> ("pcl_seg1", 1);
        seg2_pub= nh->advertise<sensor_msgs::PointCloud2> ("pcl_seg2", 1);
        seg3_pub= nh->advertise<sensor_msgs::PointCloud2> ("pcl_seg3", 1);
        segposes_pub= nh->advertise<geometry_msgs::PoseArray> ("seg_poses", 1);
        pcl_map_pub=  nh->advertise<sensor_msgs::PointCloud2> ("pcl_registered_map", 1);
        bbox_marker_pub= nh->advertise<visualization_msgs::Marker> ("bbox_marker", 1);
        bboxline_marker_pub= nh->advertise<visualization_msgs::Marker> ("bbox_line_marker", 1);
        //pcl_registered_pub
    }

    ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && (!received_first_message_bbox && !received_first_message_cloud)) {
        ROS_WARN_THROTTLE(2, "Waiting for image, depth and detections messages...");
        ros::spinOnce();
    }

    ros::Rate rate(10.0);
    while(ros::ok())
    {
        geometry_msgs::TransformStamped transform_pcl;
    
       try{
           transform_pcl = tf_buffer.lookupTransform("walrus/map", "walrus/realsense_front_color_optical_frame", ros::Time(), ros::Duration(7.0));
       }
       catch (tf2::TransformException ex){
           ROS_ERROR("%s",ex.what());
       }

       //segmentationpointClouds();
       
       bbox_marker_pub.publish(registered_bbox_vertices);
       bboxline_marker_pub.publish(registered_bbox_lines);

       registered_pcl_filtered.header.stamp= ros::Time::now();
       registered_pcl_filtered.header.frame_id="walrus/map";
       pcl_map_pub.publish(registered_pcl_filtered);

       //ROS_INFO("published");


       for(int cnt(0);cnt <Cloudbag.clusters.size(); cnt++)
       {
       
           sensor_msgs::PointCloud2 output_tmp=Cloudbag.clusters[cnt];
           if(cnt==0)
            {
                seg1_pub.publish(output_tmp);
            }
            else if(cnt==1)
            {
                seg2_pub.publish(output_tmp);
            }
            else if(cnt==2)
            {
                seg3_pub.publish(output_tmp);
            }
            else{
                if(cnt>3)
                    ROS_INFO("more than 3 clusters");
            }
       }

       if(target_obtained)
           pcl_center_pub.publish(previous_target_pose);
  
        ros::spinOnce();
        rate.sleep();

    }
        
    // Spin
    ros::spin ();

    return 0;
}
