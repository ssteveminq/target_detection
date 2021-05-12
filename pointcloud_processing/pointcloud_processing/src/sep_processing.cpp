#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <visualization_msgs/Marker.h>
#include <pointcloud_processing_msgs/ObjectInfo.h>
#include <pointcloud_processing_msgs/ObjectInfoArray.h>
#include <pointcloud_processing_msgs/fov_positions.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
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

const int QUEUE_SIZE = 10;
std::string DARKNET_TOPIC;
std::string RETINANET_TOPIC;
std::string PCL_TOPIC;
std::string FOV_TOPIC;
std::string TARGET_FRAME;
std::string TARGET_CLASS;
bool VISUAL;
bool PCL_VISUAL;




// Initialize publishers
ros::Publisher pub_target;
ros::Publisher pub_target_centroid;
ros::Publisher pub_target_poses;
ros::Publisher pub_target_point;
ros::Publisher pub_target_list;
ros::Publisher pub_ObjectInfos;

// Initialize transform listener
tf::TransformListener* lst;
tf2_ros::Buffer* pbuffer;

// Set fixed reference frame
std::string fixed_frame = "odom";
darknet_ros_msgs::BoundingBoxes current_boxes;
pointcloud_processing_msgs::fov_positions current_fov;

//map
std::map<std::string, pointcloud_processing_msgs::ObjectInfo> labels_to_obj;
std::vector<std::string> target_string;
std::map<std::string, std::deque<float>> depth_buffer;
bool is_occluder = false;
bool is_target = false;
bool received_first_message=false;
bool received_first_message_bbox =false;
bool received_first_message_cloud =false;
bool received_fov_region=false;
std::string occ_label = "unknown";

//If _name is in target_strings, return true
bool Is_target(std::string _name)
{
    bool Is_target = false;
    for(size_t i(0);i< target_string.size();i++)
    {
        if(target_string[i]==_name)
        {
            return true;
        }
    }
    return false;
}

void 
bbox_cb (const darknet_ros_msgs::BoundingBoxesConstPtr& input_detection)
{
    //ROS_INFO("bounding_box callback");
    current_boxes = *input_detection;

    received_first_message_bbox =true;

}

void 
fovregion_cb (const pointcloud_processing_msgs::fov_positionsConstPtr& fov_region_)
{
    //ROS_INFO("fov_regions callback");
    current_fov=*fov_region_;
    received_fov_region=true;
}


void
pcloud_cb (const sensor_msgs::PointCloud2ConstPtr& input_cloud)
{
    //ROS_INFO("cloud_callback");
    received_first_message_cloud =true;

     //ROS_INFO("cloud callback");
      tf2_ros::Buffer tf_buffer;
      tf2_ros::TransformListener tf2_listener(tf_buffer);
      received_first_message = true;
      //Initialize boolean variables
      is_target= false;

      // Initialize containers for clouds
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud                  (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_target           (new pcl::PointCloud<pcl::PointXYZRGB>);

      // Initialize container for object poses
      geometry_msgs::PoseArray target_poses;
      target_poses.header.stamp = ros::Time::now();
      target_poses.header.frame_id = input_cloud->header.frame_id;

      // Initialize container for centroids' markers
      visualization_msgs::Marker centroid_target_list ;

      centroid_target_list.header.frame_id =  fixed_frame;
      centroid_target_list.type = 7;
      centroid_target_list.color.a = 1.0;
      centroid_target_list.color.r = 1.0;
      centroid_target_list.action =  0;

      centroid_target_list.scale.x = 0.05;
      centroid_target_list.scale.y =0.05;
      centroid_target_list.scale.z = 0.05;

      pointcloud_processing_msgs::ObjectInfoArray objectsarray;

      // Initialize container for auxiliary clouds
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_roi            (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_voxelgrid      (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_sor            (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_xyz            (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_inplane        (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_outplane       (new pcl::PointCloud<pcl::PointXYZRGB>);

      // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
      std::vector<int> indices_gf;
      pcl::fromROSMsg( *input_cloud, *cloud);

      //Gather PCL indices
      if(received_fov_region)
      {
          //check pcl and rgb frames are using same frame_id 
          if(input_cloud->header.frame_id!=current_fov.header.frame_id)
          {
              ROS_INFO("frame is not same! cloud : %s , fov_region: %s", input_cloud->header.frame_id.c_str(), current_fov.header.frame_id.c_str());
          }

          //Check the 3D FOV region to gather if xyz of point cloud data are within 3D FOV region:
          //checking two points current_fov.position[0] // current_fov.positions[1]
          for(int k(0); k< cloud->points.size();k++)
          {
            
             if( (abs(cloud->points[k].y-current_fov.positions[0].y) <1.0) && (abs(cloud->points[k].z-current_fov.positions[0].z) <1.0)) 
             {
             
                 if( (abs(cloud->points[k].y-current_fov.positions[1].y) <2.0) && (abs(cloud->points[k].y-current_fov.positions[1].z) <2.0)) 
                 {
                     if( ((cloud->points[k].x-current_fov.positions[0].x) >0.5) && (abs(cloud->points[k].x-current_fov.positions[1].x) <3.0)) 
                         indices_gf.push_back(k);
                 
                 }
                 //printf("k: %d, x: %.2lf, y: %.2lf, z: %.2lf, \n", k, cloud->points[k].x, cloud->points[k].y, cloud->points[k].z);
             }
          }
      }
      //ROS_INFO("indices.size() = %d", indices_gf.size());


      // Get cloud width and height
      int width = input_cloud->width;
      int height = input_cloud->height;
      ROS_INFO("input_cloud width: %d, height: %d", width, height);
      int pcl_size = width*height;

    /////////////////////////////////////////////////////////////
    for(int i(0); i<current_boxes.bounding_boxes.size(); i++){
    
    // Unwrap darknet detection
    //ros::Time secs =ros::Time::now();
    std::string object_name = current_boxes.bounding_boxes[i].Class;
    int xmin                = current_boxes.bounding_boxes[i].xmin;
    int xmax                = current_boxes.bounding_boxes[i].xmax;
    int ymin                = current_boxes.bounding_boxes[i].ymin;
    int ymax                = current_boxes.bounding_boxes[i].ymax;
    float probability = current_boxes.bounding_boxes[i].probability;
    if(probability<0.6)
        return;
    ROS_INFO("object_name: %s, xmin: %d, xmax: %d, ymin: %d, ymax: %d", object_name.c_str(), xmin, xmax, ymin, ymax );
    //object_name="bottle";

    // -------------------ROI extraction------------------------------

    // Initialize container for inliers of the ROI
    pcl::PointIndices::Ptr inliers_roi (new pcl::PointIndices());


    inliers_roi->indices = indices_gf;

    // Create the filtering ROI object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract_roi;
    // Extract the inliers  of the ROI
    extract_roi.setInputCloud (cloud);
    extract_roi.setIndices (inliers_roi);
    extract_roi.setNegative (false);
    extract_roi.filter (*cloud_filtered_roi);

    //Eigen::Vector4f centroid_out;
    //pcl::compute3DCentroid(*cloud_filtered_roi,centroid_out); 

    //ROS_INFO("here-1.1");
    //extract_roi.filter (*cloud);
    //ROS_INFO("here-1.2");
    
    //ROS_INFO("here-1.2");
    // ----------------------VoxelGrid----------------------------------
    // Perform the downsampling
    pcl::VoxelGrid<pcl::PointXYZRGB> sor_voxelgrid;
    sor_voxelgrid.setInputCloud (cloud_filtered_roi);
    sor_voxelgrid.setLeafSize (0.05, 0.05, 0.05); //size of the grid
    sor_voxelgrid.filter (*cloud_filtered_voxelgrid);
    //ROS_INFO("here-1.2");


   // ---------------------StatisticalOutlierRemoval--------------------
   // Create the 
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_noise;
   // Remove noise
   //sor_noise.setInputCloud (cloud_filtered_roi);
   sor_noise.setInputCloud (cloud_filtered_voxelgrid);
   sor_noise.setMeanK (50);
   sor_noise.setStddevMulThresh (1.0);
   sor_noise.filter (*cloud_filtered_sor);

   //pcl::IndicesPtr indices_xyz(new std::vector <int>);
   //pcl::PassThrough<pcl::PointXYZRGB> pass;
   //pass.setInputCloud(cloud_filtered_sor);
   //pass.setFilterFieldName("z");
   //pass.setFilterLimits(0.5,5.0);
   //pass.filter(*indices_xyz);
    //ROS_INFO("here-1.3");

   //pcl::ExtractIndices<pcl::PointXYZRGB> extract_roi2;
    // Extract the inliers  of the ROI
    //extract_roi2.setInputCloud (cloud_filtered_sor);
    //extract_roi2.setIndices (indices_xyz);
    //extract_roi2.setNegative (false);
    //extract_roi2.filter (*cloud_filtered_sor);
  

  //ROS_INFO("here-2");
   //remove NaN points from the cloud
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr nanfiltered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
   std::vector<int> rindices;
   pcl::removeNaNFromPointCloud(*cloud_filtered_sor,*nanfiltered_cloud, rindices);

   ROS_INFO("here-1.4");
     
  // ----------------------Compute centroid-----------------------------
  Eigen::Vector4f centroid_out;
  pcl::compute3DCentroid(*cloud_filtered_sor,centroid_out); 

  geometry_msgs::PointStamped centroid_rel;
  geometry_msgs::PointStamped centroid_abs;
  centroid_abs.header.frame_id=fixed_frame;
  centroid_rel.header.frame_id = input_cloud->header.frame_id;
  //ROS_INFO("cloud frame id: %s", input_cloud->header.frame_id.c_str());
  centroid_abs.header.stamp = ros::Time(0);
  centroid_rel.header.stamp = ros::Time(0);

  centroid_rel.point.x = centroid_out[0];
  centroid_rel.point.y = centroid_out[1];
  centroid_rel.point.z = centroid_out[2];
  ROS_INFO("rel-x: %.2lf, y: %.2lf, z: %.2lf",centroid_out[0], centroid_out[1],centroid_out[2]);

  
  //lst->waitForTransform(input_cloud->header.frame_id, fixed_frame, ros::Time(0), ros::Duration(2.0));
  //try{lst->transformPoint(fixed_frame,centroid_rel, centroid_abs);}
  //ROS_INFO("succedd");}
  //catch (tf::TransformException& ex) {
      //ROS_INFO("transform error #1") ;
  //}

   //ROS_INFO("here-1.6");
  // ---------------Create detected object reference frame-------------
  geometry_msgs::PoseStamped object_pose;

  object_pose.header.stamp= input_cloud->header.stamp;
  object_pose.header.frame_id = input_cloud->header.frame_id;
  object_pose.pose.position.x = centroid_rel.point.x;
  object_pose.pose.position.y = centroid_rel.point.y;
  object_pose.pose.position.z = centroid_rel.point.z;

   ROS_INFO("here-1.7");
  pointcloud_processing_msgs::ObjectInfo cur_obj;
  cur_obj.x = xmin;
  cur_obj.y = ymin;
  cur_obj.width = xmax-xmin;
  cur_obj.height = ymax-ymin;
  cur_obj.label = object_name;
  cur_obj.last_time = ros::Time::now();
  cur_obj.average_depth=0.0;
  cur_obj.no_observation=false;
  cur_obj.center = centroid_abs.point;
  cur_obj.depth_change = false;
  cur_obj.occ_objects = false;
  if(Is_target(object_name))
      cur_obj.is_target=true;
  else
      cur_obj.is_target=false;
      
  // ---------------Store resultant cloud and centroid------------------
  ROS_INFO("object_name: %s", object_name.c_str());
  if (object_name == TARGET_CLASS){
     //pub_target_point.publish(centroid_rel);
     ROS_INFO("%s", TARGET_CLASS.c_str());
    *cloud_target += *cloud_filtered_sor;
    centroid_target_list.points.push_back(centroid_rel.point);
    target_poses.poses.push_back(object_pose.pose);
    is_target = true;
  }
  else{

      ROS_INFO("non-target object %s is detectedd", object_name.c_str());
  }

   //ROS_INFO("here-1.6");
    // Else if the label exist, check the position
    auto it = labels_to_obj.find(object_name);
    if (it == labels_to_obj.end()) {
        labels_to_obj.insert({object_name, cur_obj});
    }
    else
    {
        //update new information
        it->second= cur_obj;
    }


  objectsarray.objectinfos.push_back(cur_obj);
  }

  //publish data
  pub_ObjectInfos.publish(objectsarray);
  // Create a container for the result data.
  //sensor_msgs::PointCloud2 output_bottle;

  // Convert pcl::PointCloud to sensor_msgs::PointCloud2
  //pcl::toROSMsg(*cloud_target,output_target;

  // Set output frame as the input frame
  //output_target.header.frame_id           = input_cloud->header.frame_id;

  // Publish the data.
  //pub_target.publish (output_target;

  // Publish markers
  pub_target_centroid.publish (centroid_target_list);

  // Publish poses
  pub_target_poses.publish(target_poses);

  ROS_INFO("Pointcloud processed");


}



int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pointcloud_processing");
  ros::NodeHandle nh;

  nh.param("DARKNET_TOPIC", DARKNET_TOPIC, {"/darknet_ros/bounding_boxes"});
  nh.param("RETINANET_TOPIC", RETINANET_TOPIC, {"/retina_ros/bounding_boxes"});
  nh.param("PCL_TOPIC", PCL_TOPIC, {"/pointcloud_transformer/output_pcl2"});
  nh.param("FOV_TOPIC", FOV_TOPIC, {"/fov_regions"});
  //nh.param("PCL_TOPIC", PCL_TOPIC, {"/points2"});
  nh.param("TARGET_FRAME", TARGET_FRAME, {"odom"});
  nh.param("TARGET_CLASS", TARGET_CLASS, {"chair"});
  nh.param("VISUAL", VISUAL, {true});
  nh.param("PCL_VISUAL", PCL_VISUAL, {true});

  target_string.push_back(TARGET_CLASS);

  // Initialize subscribers to darknet detection and pointcloud
  ros::Subscriber bbox_sub;
  ros::Subscriber cloud_sub;
  ros::Subscriber fovregion_sub;
  bbox_sub = nh.subscribe<darknet_ros_msgs::BoundingBoxes>(DARKNET_TOPIC, 100, bbox_cb); 
  cloud_sub =nh.subscribe<sensor_msgs::PointCloud2>(PCL_TOPIC, 100, pcloud_cb );
  fovregion_sub = nh.subscribe<pointcloud_processing_msgs::fov_positions>(FOV_TOPIC, 100, fovregion_cb );

  // Initialize transform listener
  //tf::TransformListener listener(ros::Duration(10));
  //lst = &listener;
  //tf2_ros::Buffer tf_buffer(ros::Duration(100));
  //pbuffer = &tf_buffer;

  // Synchronize darknet detection with pointcloud
  //sync.registerCallback(boost::bind(&cloud_cb, _1, _2));

  // Create a ROS publisher for the output point cloud
  pub_target = nh.advertise<sensor_msgs::PointCloud2> ("pcl_target", 1);

  // Create a ROS publisher for the output point cloud centroid markers
  pub_target_centroid = nh.advertise<visualization_msgs::Marker> ("target_centroids", 1);

  //Create a ROS publisher for the detected  points
  pub_target_point= nh.advertise<geometry_msgs::PointStamped> ("/target_center", 1);

  // Create a ROS publisher for the detected poses
  pub_target_poses = nh.advertise<geometry_msgs::PoseArray> ("/target_poses", 1);

  pub_ObjectInfos = nh.advertise<pointcloud_processing_msgs::ObjectInfoArray> ("ObjectInfos", 1);

  ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && (!received_first_message_bbox && !received_first_message_cloud)) {
        ROS_WARN_THROTTLE(2, "Waiting for image, depth and detections messages...");
        ros::spinOnce();
    }





  // Spin
  ros::spin ();
}
