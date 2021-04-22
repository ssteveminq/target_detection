#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <visualization_msgs/Marker.h>
#include <pointcloud_processing_msgs/handle_position.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>

// Initialize publishers
ros::Publisher pub_door;
ros::Publisher pub_door_centroid;
ros::Publisher pub_door_poses;
ros::Publisher pub_handle;
ros::Publisher pub_handle_centroid;
ros::Publisher pub_handle_poses;
ros::Publisher pub_cabinetdoor;
ros::Publisher pub_cabinetdoor_centroid;
ros::Publisher pub_cabinetdoor_poses;
ros::Publisher pub_refrigeratordoor;
ros::Publisher pub_refrigeratordoor_centroid;
ros::Publisher pub_handle_centroid_list;
ros::Publisher pub_refrigeratordoor_poses;

// Initialize transform listener
tf::TransformListener* lst;

// Set fixed reference frame
std::string fixed_frame = "map";


void 
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const darknet_ros_msgs::BoundingBoxesConstPtr& input_detection)
{

  // Initialize containers for clouds
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud                  (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_door             (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_handle           (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cabinetdoor      (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_refrigeratordoor (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Handle orientation
  std::vector<std::string> handle_orientation;

  // Initialize container for object poses
  geometry_msgs::PoseArray door_poses,handle_poses,cabinetdoor_poses,refrigeratordoor_poses;
  handle_poses.header.stamp = door_poses.header.stamp = cabinetdoor_poses.header.stamp = refrigeratordoor_poses.header.stamp = ros::Time::now();
  handle_poses.header.frame_id = door_poses.header.frame_id = cabinetdoor_poses.header.frame_id = refrigeratordoor_poses.header.frame_id = fixed_frame;

  // Initialize container for centroids' markers
  visualization_msgs::Marker centroid_door_list, centroid_handle_list, centroid_cabinetdoor_list, centroid_refrigeratordoor_list;

  centroid_door_list.header.frame_id = centroid_handle_list.header.frame_id = centroid_cabinetdoor_list.header.frame_id = centroid_refrigeratordoor_list.header.frame_id = fixed_frame;

  centroid_door_list.type = centroid_handle_list.type = centroid_cabinetdoor_list.type = centroid_refrigeratordoor_list.type = 7;

  centroid_door_list.color.a = centroid_handle_list.color.a = centroid_cabinetdoor_list.color.a = centroid_refrigeratordoor_list.color.a = 1.0;

  centroid_door_list.color.r = centroid_handle_list.color.r = centroid_cabinetdoor_list.color.r = centroid_refrigeratordoor_list.color.r = 1.0;

  centroid_door_list.action = centroid_handle_list.action = centroid_cabinetdoor_list.action = centroid_refrigeratordoor_list.action = 0;

  centroid_door_list.scale.x = centroid_handle_list.scale.x = centroid_cabinetdoor_list.scale.x = centroid_refrigeratordoor_list.scale.x = 0.05;

  centroid_door_list.scale.y = centroid_handle_list.scale.y = centroid_cabinetdoor_list.scale.y = centroid_refrigeratordoor_list.scale.y = 0.05;

  centroid_door_list.scale.z = centroid_handle_list.scale.z = centroid_cabinetdoor_list.scale.z = centroid_refrigeratordoor_list.scale.z = 0.05;

  // Initialize message with handle detections
  pointcloud_processing_msgs::handle_position handle_list;

  // Initialize container for auxiliary clouds
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_roi            (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_voxelgrid      (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_sor            (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_inplane        (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_outplane       (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
  pcl::fromROSMsg( *input_cloud, *cloud);

  // Get cloud width and height
  int width = input_cloud->width;
  int height = input_cloud->height;

  // Number of objects detected
  int num_boxes = input_detection->bounding_boxes.size();

  // For each object detected
  for(int i(0); i<num_boxes; i++){
    
    // Unwrap darknet detection
    std::string object_name = input_detection->bounding_boxes[i].Class;
    int xmin                = input_detection->bounding_boxes[i].xmin;
    int xmax                = input_detection->bounding_boxes[i].xmax;
    int ymin                = input_detection->bounding_boxes[i].ymin;
    int ymax                = input_detection->bounding_boxes[i].ymax;


    // ---------------Determine handle orientation--------------------
    if (object_name=="handle"){
      if ((xmax-xmin)>=(ymax-ymin)){
        handle_orientation.push_back("horizontal");
      }
      else{
        handle_orientation.push_back("vertical");
      }
    }

    // -------------------ROI extraction------------------------------

    // Initialize container for inliers of the ROI
    pcl::PointIndices::Ptr inliers_roi (new pcl::PointIndices ());

    // Get inliers
    std::vector<int> indices;
    for (int column(xmin); column<=xmax; column++){
      for (int row(ymin); row<=ymax; row++){
        // Pixel coordinates to pointcloud index
        indices.push_back(row*width+column);
      }
    }

    inliers_roi->indices = indices;

    // Create the filtering ROI object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract_roi;
    // Extract the inliers  of the ROI
    extract_roi.setInputCloud (cloud);
    extract_roi.setIndices (inliers_roi);
    extract_roi.setNegative (false);
    extract_roi.filter (*cloud_filtered_roi);
    
    
    // ----------------------VoxelGrid----------------------------------

    // Perform the downsampling
    pcl::VoxelGrid<pcl::PointXYZRGB> sor_voxelgrid;
    sor_voxelgrid.setInputCloud (cloud_filtered_roi);
    sor_voxelgrid.setLeafSize (0.01, 0.01, 0.01); //size of the grid
    sor_voxelgrid.filter (*cloud_filtered_voxelgrid);

   // Exception
   if (cloud_filtered_voxelgrid->points.size() < 3){
     if (object_name=="handle"){handle_orientation.pop_back();}
   break;
   }

    
   // ---------------------StatisticalOutlierRemoval--------------------

   // Create the filtering object
   pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_noise;
   // Remove noise
   sor_noise.setInputCloud (cloud_filtered_voxelgrid);
   sor_noise.setMeanK (50);
   sor_noise.setStddevMulThresh (1.0);
   sor_noise.filter (*cloud_filtered_sor);


   // ---------------------RANSAC_PlanarSegmentation--------------------

   // Initialize containers for plane coefficients and inliers of the plane
   pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
   pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices ());

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);

  // Segment the largest planar component from the remaining cloud
  seg.setInputCloud (cloud_filtered_sor);
  seg.segment (*inliers_plane, *coefficients);

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZRGB> extract_plane;

  // Extract the inliers
  extract_plane.setInputCloud (cloud_filtered_sor);
  extract_plane.setIndices (inliers_plane);
  extract_plane.setNegative (false);
  extract_plane.filter (*cloud_filtered_inplane);
    
  // Extract the outliers

  extract_plane.setNegative (true);
  extract_plane.filter (*cloud_filtered_outplane);


  // ----------------------Compute normal vector-----------------------

  // Initialize container for vectors
  geometry_msgs::Vector3Stamped normal_vector_rel;
  geometry_msgs::Vector3Stamped normal_vector_abs;

  normal_vector_rel.header.stamp = ros::Time::now();
  normal_vector_rel.header.frame_id = coefficients->header.frame_id;
  normal_vector_rel.vector.x = coefficients->values[0];
  normal_vector_rel.vector.y = coefficients->values[1];
  normal_vector_rel.vector.z = coefficients->values[2];

  // make sure the normal vector is pointing inwards the object

  if (normal_vector_rel.vector.z < 0){
    normal_vector_rel.vector.x = -normal_vector_rel.vector.x;
    normal_vector_rel.vector.y = -normal_vector_rel.vector.y;    
    normal_vector_rel.vector.z = -normal_vector_rel.vector.z;
  }

  // Transform the normal vector to fixed reference frame
  lst->waitForTransform(fixed_frame, input_cloud->header.frame_id, ros::Time::now(), ros::Duration(4.0));
  lst->transformVector(fixed_frame,normal_vector_rel, normal_vector_abs);
  //try{lst->transformVector(fixed_frame,normal_vector_rel, normal_vector_abs);}
  //catch (tf::TransformException& ex) {}

  // ----------------------Compute centroid-----------------------------

  // Centroid of cloud without plane
  Eigen::Vector4f centroid_out;
  pcl::compute3DCentroid(*cloud_filtered_outplane,centroid_out); 

  // Centroid of plane cloud
  Eigen::Vector4f centroid_in;
  pcl::compute3DCentroid(*cloud_filtered_inplane,centroid_in); 
  

  // ----------------Set attributes of centroid marker------------------

  geometry_msgs::PointStamped centroid_rel;
  geometry_msgs::PointStamped centroid_abs;
  centroid_rel.header.frame_id = input_cloud->header.frame_id;
  centroid_rel.header.stamp = ros::Time::now();

  if (object_name == "handle"){
    centroid_rel.point.x = centroid_out[0];
    centroid_rel.point.y = centroid_out[1];
    centroid_rel.point.z = centroid_out[2];
  }
  else{
    centroid_rel.point.x = centroid_in[0];
    centroid_rel.point.y = centroid_in[1];
    centroid_rel.point.z = centroid_in[2];
  }

  // Transform centroid to fixed reference frame
  lst->waitForTransform(fixed_frame, input_cloud->header.frame_id, ros::Time::now(), ros::Duration(4.0));
  lst->transformPoint(fixed_frame,centroid_rel, centroid_abs);
  //try{lst->transformPoint(fixed_frame,centroid_rel, centroid_abs);}
  //catch (tf::TransformException& ex) {}


  // ---------------Create detected object reference frame-------------

  geometry_msgs::PoseStamped object_pose;
  
  // Project normal vector in the floor plane
  normal_vector_abs.vector.z = 0.0;

  // Vertical vector
  geometry_msgs::Vector3 vertical_vector;
  vertical_vector.x = 0.0; vertical_vector.y = 0.0; vertical_vector.z = 1.0;
  
  // normalize vectors
  float module_normal = sqrt(normal_vector_abs.vector.x*normal_vector_abs.vector.x + normal_vector_abs.vector.y*normal_vector_abs.vector.y);
  normal_vector_abs.vector.x = normal_vector_abs.vector.x/module_normal; normal_vector_abs.vector.y = normal_vector_abs.vector.y/module_normal; 

  // do cross product
  geometry_msgs::Vector3 cross_p;
  
  cross_p.x = normal_vector_abs.vector.y;
  cross_p.y = -normal_vector_abs.vector.x;
  cross_p.z = 0.0;


  // calculate determinant
  //float determinant = -cross_p.x*normal_vector_abs.vector.y+normal_vector_abs.vector.x*cross_p.y;

  // make sure the orientation of the base is correct
  //cross_p.x = cross_p.x*determinant; cross_p.y = cross_p.y*determinant;

  // compute rotation matrix
  tf::Matrix3x3 rotation_matrix(0.0,cross_p.x,normal_vector_abs.vector.x,0.0,cross_p.y,normal_vector_abs.vector.y,1.0,0.0,0.0);

  // compute quaternion
  tf::Quaternion quaternion;
  rotation_matrix.getRotation(quaternion);

  // set orientation of the object
  tf::Stamped<tf::Transform> object_orientation(tf::Transform(quaternion), ros::Time::now(), fixed_frame);
  tf::poseStampedTFToMsg(object_orientation,object_pose);

  object_pose.pose.position.x = centroid_abs.point.x;
  object_pose.pose.position.y = centroid_abs.point.y;
  object_pose.pose.position.z = centroid_abs.point.z;

  // ---------------Store resultant cloud and centroid------------------
  if (object_name == "door"){
    *cloud_door += *cloud_filtered_inplane;
    centroid_door_list.points.push_back(centroid_abs.point);
    door_poses.poses.push_back(object_pose.pose);
  }
  else if (object_name == "handle"){
    *cloud_handle += *cloud_filtered_outplane;
    centroid_handle_list.points.push_back(centroid_abs.point);
    handle_list.position.push_back(centroid_abs);
    handle_poses.poses.push_back(object_pose.pose);
  }
  else if (object_name == "cabinet door"){
    *cloud_cabinetdoor += *cloud_filtered_inplane;
    centroid_cabinetdoor_list.points.push_back(centroid_abs.point);
    cabinetdoor_poses.poses.push_back(object_pose.pose);
  }
  else{
    *cloud_refrigeratordoor += *cloud_filtered_inplane;
    centroid_refrigeratordoor_list.points.push_back(centroid_abs.point);
    refrigeratordoor_poses.poses.push_back(object_pose.pose);
  }

  }


  //--------Assign to each handle its corresponding door pose-----------

  // Number of handles in detection
  int num_handles = handle_poses.poses.size();
  
  // Number of doors in detection
  int num_doors = door_poses.poses.size();
  
  // Number of cabinet doors in detection
  int num_cabinetdoors = cabinetdoor_poses.poses.size();
  
  // Number of refrigerator doors in detection
  int num_refrigeratordoors = refrigeratordoor_poses.poses.size();

  float distance;


  // For each handle
  for (int i(0); i<num_handles; i++){

    geometry_msgs::Point handle_position = handle_poses.poses[i].position;

    geometry_msgs::Pose chosen_pose = handle_poses.poses[i];
    
    float distance_min = 1.0;

    //Check all doors
    for (int j(0); j<num_doors; j++){

      geometry_msgs::Point door_position = door_poses.poses[j].position;
      distance = sqrt((handle_position.x-door_position.x)*(handle_position.x-door_position.x)+(handle_position.y-door_position.y)*(handle_position.y-door_position.y)+(handle_position.z-door_position.z)*(handle_position.z-door_position.z));

      if (distance<distance_min){
	chosen_pose = door_poses.poses[j];
        distance_min = distance;
      }
    }

    //Check all cabinet doors
    for (int j(0); j<num_cabinetdoors; j++){

      geometry_msgs::Point cabinetdoor_position = cabinetdoor_poses.poses[j].position;
      distance = sqrt((handle_position.x-cabinetdoor_position.x)*(handle_position.x-cabinetdoor_position.x)+(handle_position.y-cabinetdoor_position.y)*(handle_position.y-cabinetdoor_position.y)+(handle_position.z-cabinetdoor_position.z)*(handle_position.z-cabinetdoor_position.z));

      if (distance<distance_min){
	chosen_pose = cabinetdoor_poses.poses[j];
        distance_min = distance;
      }
    }

    //Check all refrigerator doors
    for (int j(0); j<num_refrigeratordoors; j++){

      geometry_msgs::Point refrigeratordoor_position = refrigeratordoor_poses.poses[j].position;
      distance = sqrt((handle_position.x-refrigeratordoor_position.x)*(handle_position.x-refrigeratordoor_position.x)+(handle_position.y-refrigeratordoor_position.y)*(handle_position.y-refrigeratordoor_position.y)+(handle_position.z-refrigeratordoor_position.z)*(handle_position.z-refrigeratordoor_position.z));

      if (distance<distance_min){
	chosen_pose = refrigeratordoor_poses.poses[j];
        distance_min = distance;
      }
    }

   if (handle_orientation[i]=="horizontal"){

     tf::Quaternion auxiliary_quaternion;

     tf::quaternionMsgToTF(chosen_pose.orientation, auxiliary_quaternion);
    
     tf::Matrix3x3 original(auxiliary_quaternion);

     // Rotation of 90 degrees on z axis, first column is the second and the second is -first
     tf::Matrix3x3 rotated(original[0][1],-original[0][0],original[0][2],original[1][1],-original[1][0],original[1][2],original[2][1],-original[2][0],original[2][2]);
     rotated.getRotation(auxiliary_quaternion);

     tf::quaternionTFToMsg(auxiliary_quaternion,chosen_pose.orientation);
   }

    handle_poses.poses[i].orientation=chosen_pose.orientation;

  }

  // Create a container for the result data.
  sensor_msgs::PointCloud2 output_door;
  sensor_msgs::PointCloud2 output_handle;
  sensor_msgs::PointCloud2 output_cabinetdoor;
  sensor_msgs::PointCloud2 output_refrigeratordoor;

  // Convert pcl::PointCloud to sensor_msgs::PointCloud2
  pcl::toROSMsg(*cloud_door,output_door);
  pcl::toROSMsg(*cloud_handle,output_handle);
  pcl::toROSMsg(*cloud_cabinetdoor,output_cabinetdoor);
  pcl::toROSMsg(*cloud_refrigeratordoor,output_refrigeratordoor);

  // Set output frame as the input frame
  output_door.header.frame_id             = input_cloud->header.frame_id;
  output_handle.header.frame_id           = input_cloud->header.frame_id;
  output_cabinetdoor.header.frame_id      = input_cloud->header.frame_id;
  output_refrigeratordoor.header.frame_id = input_cloud->header.frame_id;

  // Publish the data.
  pub_door.publish (output_door);
  pub_handle.publish (output_handle);
  pub_cabinetdoor.publish (output_cabinetdoor);
  pub_refrigeratordoor.publish (output_refrigeratordoor);

  // Publish markers
  pub_door_centroid.publish (centroid_door_list);
  pub_handle_centroid.publish (centroid_handle_list);
  pub_cabinetdoor_centroid.publish (centroid_cabinetdoor_list);
  pub_refrigeratordoor_centroid.publish (centroid_refrigeratordoor_list);

  // Publish handle position
  pub_handle_centroid_list.publish (handle_list);

  // Publish poses
  pub_door_poses.publish(door_poses);
  pub_handle_poses.publish(handle_poses);
  pub_cabinetdoor_poses.publish(cabinetdoor_poses);
  pub_refrigeratordoor_poses.publish(refrigeratordoor_poses);

  ROS_INFO("Pointcloud processed");
}


int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pointcloud_processing");
  ros::NodeHandle nh;

  // Initialize subscribers to darknet detection and pointcloud
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "hsrb/head_rgbd_sensor/depth_registered/rectified_points", 1);
  message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> sub_box(nh, "darknet_ros/bounding_boxes", 1);

  // Initialize transform listener
  tf::TransformListener listener(ros::Duration(10));
  lst = &listener;

  // Synchronize darknet detection with pointcloud
  typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_cloud, sub_box);
  sync.registerCallback(boost::bind(&cloud_cb, _1, _2));

  // Create a ROS publisher for the output point cloud
  pub_door = nh.advertise<sensor_msgs::PointCloud2> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/door", 1);
  pub_handle = nh.advertise<sensor_msgs::PointCloud2> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/handle", 1);
  pub_cabinetdoor = nh.advertise<sensor_msgs::PointCloud2> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/cabinet_door", 1);
  pub_refrigeratordoor = nh.advertise<sensor_msgs::PointCloud2> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/refrigerator_door", 1);

  // Create a ROS publisher for the output point cloud centroid markers
  pub_door_centroid = nh.advertise<visualization_msgs::Marker> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/door_centroid", 1);
  pub_handle_centroid = nh.advertise<visualization_msgs::Marker> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/handle_centroid", 1);
  pub_cabinetdoor_centroid = nh.advertise<visualization_msgs::Marker> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/cabinet_door_centroid", 1);
  pub_refrigeratordoor_centroid = nh.advertise<visualization_msgs::Marker> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/refrigerator_door_centroid", 1);

  // Create a ROS publisher for the detected poses
  pub_door_poses = nh.advertise<geometry_msgs::PoseArray> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/door_pose", 1);
  pub_handle_poses = nh.advertise<geometry_msgs::PoseArray> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/handle_pose", 1);
  pub_cabinetdoor_poses = nh.advertise<geometry_msgs::PoseArray> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/cabinet_door_pose", 1);
  pub_refrigeratordoor_poses = nh.advertise<geometry_msgs::PoseArray> ("hsrb/head_rgbd_sensor/depth_registered/rectified_points/refrigerator_door_pose", 1);

  // Create a ROS publisher for handle position
  pub_handle_centroid_list = nh.advertise<pointcloud_processing_msgs::handle_position> ("detected_handles", 1);

  // Spin
  ros::spin ();
}
