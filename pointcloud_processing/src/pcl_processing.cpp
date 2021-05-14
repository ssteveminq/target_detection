#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/Marker.h>
#include <pointcloud_processing_msgs/ObjectInfo.h>
#include <pointcloud_processing_msgs/ObjectInfoArray.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
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
std::string PCL_TOPIC;
std::string TARGET_FRAME;
bool VISUAL;
bool PCL_VISUAL;

// Initialize publishers
ros::Publisher pub_bottle;
ros::Publisher pub_bottle_centroid;
ros::Publisher pub_bottle_poses;
ros::Publisher pub_bottle_point;
ros::Publisher pub_cup_point;
ros::Publisher pub_cup;
ros::Publisher pub_cup_centroid;
ros::Publisher pub_cup_poses;
ros::Publisher pub_bottle_list;
ros::Publisher pub_ObjectInfos;



// Initialize transform listener
tf::TransformListener* lst;
tf2_ros::Buffer* pbuffer;
tf2_ros::StaticTransformBroadcaster* broad_caster;
  //static tf2_ros::StaticTransformBroadcaster static_broadcaster;

// Set fixed reference frame
std::string fixed_frame = "odom";
//std::string fixed_frame = "map";

//map
std::map<std::string, pointcloud_processing_msgs::ObjectInfo> labels_to_obj;
std::vector<std::string> target_string;
std::map<std::string, std::deque<float>> depth_buffer;
bool is_occluder = false;
bool is_bottle = false;
bool is_cup= false;
bool received_first_message =false;
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
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input_cloud, const darknet_ros_msgs::BoundingBoxesConstPtr& input_detection)
{

  //ROS_INFO("cloud callback");
  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf2_listener(tf_buffer);
  received_first_message = true;
  //Initialize boolean variables
  is_bottle=is_cup = false;

  // Initialize containers for clouds
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud                  (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_bottle           (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cup              (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Initialize container for object poses
  geometry_msgs::PoseArray bottle_poses,cup_poses;
  bottle_poses.header.stamp = cup_poses.header.stamp = ros::Time::now();
  bottle_poses.header.frame_id = cup_poses.header.frame_id = input_cloud->header.frame_id;

  // Initialize container for centroids' markers
  visualization_msgs::Marker centroid_bottle_list, centroid_cup_list ;

  centroid_bottle_list.header.frame_id = centroid_cup_list.header.frame_id = input_cloud->header.frame_id;
  centroid_bottle_list.type = centroid_cup_list.type = 7;
  centroid_bottle_list.color.a = centroid_cup_list.color.a = 1.0;
  centroid_bottle_list.color.r = centroid_cup_list.color.r = 1.0;
  centroid_bottle_list.action = centroid_cup_list.action = 0;

  centroid_bottle_list.scale.x = centroid_cup_list.scale.x = 0.05;
  centroid_bottle_list.scale.y = centroid_cup_list.scale.y = 0.05;
  centroid_bottle_list.scale.z = centroid_cup_list.scale.z = 0.05;

  pointcloud_processing_msgs::ObjectInfoArray objectsarray;

  // Initialize container for auxiliary clouds
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_roi            (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_voxelgrid      (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_sor            (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_xyz            (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_inplane        (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_outplane       (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Convert sensor_msgs::PointCloud2 to pcl::PointCloud
  pcl::fromROSMsg( *input_cloud, *cloud);

  // Get cloud width and height
  //int width = input_cloud->width;
  //int height = input_cloud->height;
    //ROS_INFO("input_cloud width: %d, height: %d", width, height);
  //int width = input_cloud->width;
  //int height = input_cloud->height;
  int width = cloud->width;
  int height = cloud->height;
  int pcl_size = width*height;
  if(width==1 ||height==1)
  {
      width = 640;
      height = 480;
  
  }
  ROS_INFO("input_cloud width: %d, height: %d", width, height);
  

  // Number of objects detected
  int num_boxes = input_detection->bounding_boxes.size();

  // For each object detected
  for(int i(0); i<num_boxes; i++){
    
    // Unwrap darknet detection
    //ros::Time secs =ros::Time::now();
    std::string object_name = input_detection->bounding_boxes[i].Class;
    int xmin                = input_detection->bounding_boxes[i].xmin;
    int xmax                = input_detection->bounding_boxes[i].xmax;
    int ymin                = input_detection->bounding_boxes[i].ymin;
    int ymax                = input_detection->bounding_boxes[i].ymax;
    float probability = input_detection->bounding_boxes[i].probability;
    //if(probability<0.6)
        //return;
    //ROS_INFO("object_name: %s, xmin: %d, xmax: %d, ymin: %d, ymax: %d", object_name.c_str(), xmin, xmax, ymin, ymax );
    //object_name="bottle";
    //ROS_INFO("object_name: %s, xmin: %d, xmax: %d, ymin: %d, ymax: %d, probability: %.2lf", object_name.c_str(), xmin, xmax, ymin, ymax, probability );
    if(probability<0.75)
        continue;
    else
        ROS_INFO("object_name: %s, xmin: %d, xmax: %d, ymin: %d, ymax: %d, probability: %.2lf", object_name.c_str(), xmin, xmax, ymin, ymax, probability );

    // -------------------ROI extraction------------------------------
    // Initialize container for inliers of the ROI
    pcl::PointIndices::Ptr inliers_roi (new pcl::PointIndices());

    // Get inliers
    std::vector<int> indices;
    for (int column(xmin); column<=xmax; column++){
      for (int row(ymin); row<=ymax; row++){
        // Pixel coordinates to pointcloud index
        //int idxx= row*width+column;
        int idxx= row*width+column;
        std::cout<<"idxx"<<idxx<<std::endl;
        if(pcl_size>idxx)
            indices.push_back(idxx);
      }
    }
    ROS_INFO("indices.size() = %lu", indices.size());

    inliers_roi->indices = indices;
    // Create the filtering ROI object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract_roi;
    // Extract the inliers  of the ROI
    extract_roi.setInputCloud (cloud);
    extract_roi.setIndices (inliers_roi);
    extract_roi.setNegative (false);
    extract_roi.filter (*cloud_filtered_roi);
    //extract_roi.filter (*cloud);
    //ROS_INFO("here-1.2");
    
    // ----------------------VoxelGrid----------------------------------
    // Perform the downsampling
    pcl::VoxelGrid<pcl::PointXYZRGB> sor_voxelgrid;
    sor_voxelgrid.setInputCloud (cloud_filtered_roi);
    sor_voxelgrid.setLeafSize (0.05, 0.05, 0.05); //size of the grid
    sor_voxelgrid.filter (*cloud_filtered_voxelgrid);
    ROS_INFO("here-1.2");


   // ---------------------StatisticalOutlierRemoval--------------------
   // Create the 
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_noise;
   // Remove noise
   //sor_noise.setInputCloud (cloud_filtered_roi);
   sor_noise.setInputCloud (cloud_filtered_voxelgrid);
   sor_noise.setMeanK (50);
   sor_noise.setStddevMulThresh (1.0);
   sor_noise.filter (*cloud_filtered_sor);



   /*
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices ());
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.05);
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
  //extract_plane.setNegative (true);
  //extract_plane.filter (*cloud_filtered_outplane);
  */

   pcl::IndicesPtr indices_xyz(new std::vector <int>);
   pcl::PassThrough<pcl::PointXYZRGB> pass;
   pass.setInputCloud(cloud_filtered_sor);
   //pass.setInputCloud(cloud_filtered_inplane);
   pass.setFilterFieldName("z");
   pass.setFilterLimits(0.0,2.5);
   pass.filter(*indices_xyz);

   //pcl::ExtractIndices<pcl::PointXYZRGB> extract_roi2;
    // Extract the inliers  of the ROI
    //extract_roi2.setInputCloud (cloud_filtered_sor);
    //extract_roi2.setIndices (indices_xyz);
    //extract_roi2.setNegative (false);
    //extract_roi2.filter (*cloud_filtered_sor);
  
   //remove NaN points from the cloud
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr nanfiltered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
   std::vector<int> rindices;
   //pcl::removeNaNFromPointCloud(*cloud_filtered_sor,*nanfiltered_cloud, rindices);
   pcl::removeNaNFromPointCloud(*cloud_filtered_sor,*nanfiltered_cloud, rindices);

   /*

   pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree(0.2f);
   octree.setInputCloud(cloud_filtered_sor);
   octree.addPointsFromInputCloud();

   pcl::PointXYZ closest;
   closest.x = 0;
   closest.y = 0;
   closest.z = 0;

   std::vector<int> pointIdxNKNSearch;
   std::vector<float> pointNKNSquaredDistance;

   if (octree.nearestKSearch(closest, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0)
   {
       std::cout << "No point detected" << std::endl;
       return;
   }

   // Generate voxel for creating object detection
   //
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> pointIdxVec;
   
    if (octree.voxelSearch(cloud_filtered_sor->points[pointIdxNKNSearch[0]], pointIdxVec) > 0)
    {
        for (std::size_t i = 0; i < pointIdxVec.size(); ++i)
        {
            obstacle_cloud->points.push_back(cloud_filtered_sor->points[pointIdxVec[i]]);
        }
    }
   
    if (obstacle_cloud->points.size() == 0)
    {
        std::cout << "The closest point has not been recorded to any obstacle voxel point" <<
            std::endl;
        return;
    }

    
    for(int j(0);j<obstacle_cloud->points.size(); j++)
    {
        ROS_INFO("x: %.2lf, y: %.2lf, z: %.2lf ", obstacle_cloud->points[j].x,
                obstacle_cloud->points[j].y, obstacle_cloud->points[j].z);
    
    }


    */
   


  // ----------------------Compute centroid-----------------------------
  Eigen::Vector4f centroid_out;
  pcl::compute3DCentroid(*cloud_filtered_sor,centroid_out); 

  geometry_msgs::PointStamped centroid_rel;
  geometry_msgs::PointStamped centroid_abs;
  centroid_abs.header.frame_id=TARGET_FRAME;
  centroid_rel.header.frame_id = input_cloud->header.frame_id;
  //ROS_INFO("cloud frame id: %s", input_cloud->header.frame_id.c_str());
  centroid_abs.header.stamp = ros::Time(0);
  centroid_rel.header.stamp = ros::Time(0);

  centroid_rel.point.x = centroid_out[0];
  centroid_rel.point.y = centroid_out[1];
  centroid_rel.point.z = centroid_out[2];
  //ROS_INFO("rel-x: %.2lf, y: %.2lf, z: %.2lf",centroid_out[0], centroid_out[1],centroid_out[2]);
  //ROS_INFO(")

  
  lst->waitForTransform(input_cloud->header.frame_id, TARGET_FRAME, ros::Time(0), ros::Duration(2.0));
  try{lst->transformPoint(TARGET_FRAME,centroid_rel, centroid_abs);}
  catch (tf::TransformException& ex) {
      ROS_INFO("transform error #1") ;
  }

  // ---------------Create detected object reference frame-------------
  geometry_msgs::PoseStamped object_pose;

  object_pose.header.stamp= input_cloud->header.stamp;
  object_pose.header.frame_id = input_cloud->header.frame_id;
  object_pose.pose.position.x = centroid_rel.point.x;
  object_pose.pose.position.y = centroid_rel.point.y;
  object_pose.pose.position.z = centroid_rel.point.z;

  geometry_msgs::TransformStamped object_transform;
  object_transform.header.stamp= input_cloud->header.stamp;
  object_transform.header.frame_id = TARGET_FRAME;
  object_transform.child_frame_id = "target_frame";
  object_transform.transform.translation.x = centroid_abs.point.x;
  object_transform.transform.translation.y = centroid_abs.point.y;
  object_transform.transform.translation.z = centroid_abs.point.z;
  object_transform.transform.rotation.x = 0.0;
  object_transform.transform.rotation.y = 0.0;
  object_transform.transform.rotation.z = 0.0;
  object_transform.transform.rotation.w = 1.0;
  broad_caster->sendTransform(object_transform);
  //object_transform.pose.position.x = centroid_abs.point.x;
  //object_transform.pose.position.y = centroid_abs.point.y;
  //object_transform.pose.position.z = centroid_abs.point.z;




  pointcloud_processing_msgs::ObjectInfo cur_obj;
  cur_obj.x = xmin;
  cur_obj.y = ymin;
  cur_obj.width = xmax-xmin;
  cur_obj.height = ymax-ymin;
  cur_obj.label = object_name;
  cur_obj.last_time = ros::Time::now();
  //cur_obj.average_depth=avg_depth;
  cur_obj.average_depth=0;
  cur_obj.no_observation=false;
  cur_obj.center = centroid_abs.point;
  cur_obj.depth_change = false;
  cur_obj.occ_objects = false;
  if(Is_target(object_name))
      cur_obj.is_target=true;
  else
      cur_obj.is_target=false;
      
  // ---------------Store resultant cloud and centroid------------------
  if (object_name == "bottle"){
     //pub_bottle_point.publish(centroid_rel);
     //ROS_INFO("bottle");
    *cloud_bottle += *cloud_filtered_sor;
    centroid_bottle_list.points.push_back(centroid_rel.point);
    bottle_poses.poses.push_back(object_pose.pose);
    is_bottle = true;
  }
  else if (object_name == "cup"){
    *cloud_cup += *cloud_filtered_sor;
    centroid_cup_list.points.push_back(centroid_rel.point);
    cup_poses.poses.push_back(object_pose.pose);
    is_cup = true;
  }
  else{

      ROS_INFO("non-target object %s is detectedd", object_name.c_str());
  }

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

  ///// pcl publisher ////
  // Create a container for the result data.
  if(PCL_VISUAL)
  {
      sensor_msgs::PointCloud2 output_bottle;
      //sensor_msgs::PointCloud2 output_cup;

      //Convert pcl::PointCloud to sensor_msgs::PointCloud2
      pcl::toROSMsg(*cloud_bottle,output_bottle);
      //pcl::toROSMsg(*cloud_cup,output_cup);

      //Set output frame as the input frame
      output_bottle.header.frame_id   = input_cloud->header.frame_id;
      //output_cup.header.frame_id      = input_cloud->header.frame_id;

      // Publish the data.
      pub_bottle.publish (output_bottle);
      //pub_cup.publish (output_cup);
  }
  ///// pcl publisher /////

  // Publish markers
  pub_bottle_centroid.publish (centroid_bottle_list);
  pub_cup_centroid.publish (centroid_cup_list);

  // Publish poses
  pub_bottle_poses.publish(bottle_poses);
  pub_cup_poses.publish(cup_poses);


  ROS_INFO("Pointcloud processed");
}


int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pointcloud_processing");
  ros::NodeHandle nh;

  nh.param("DARKNET_TOPIC", DARKNET_TOPIC, {"/darknet_ros/bounding_boxes"});
  nh.param("PCL_TOPIC", PCL_TOPIC, {"/camera/depth/color/points"});
  nh.param("TARGET_FRAME", TARGET_FRAME, {"map"});
  nh.param("VISUAL", VISUAL, {true});
  nh.param("PCL_VISUAL", PCL_VISUAL, {true});

  target_string.push_back("bottle");
  target_string.push_back("cup");

  std::deque<float> depthvector;
  depth_buffer.insert({"bottle", depthvector});
  depth_buffer.insert({"cup", depthvector});

  // Initialize subscribers to darknet detection and pointcloud
  //message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "/hsrb/head_rgbd_sensor/depth_registered/rectified_points", 1);
  //message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> sub_box(nh, "/darknet_ros/bounding_boxes", 1);
  //message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> sub_box(nh, "/retina_ros/bounding_boxes", 1);
  //message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "/lidar_points", 1);
  //message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "/pointcloud_transformer/output_pcl2", 1);
  //message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "/points2", 1);
  message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> sub_box(nh, DARKNET_TOPIC, 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, PCL_TOPIC, 1);

  // Initialize transform listener
  tf::TransformListener listener(ros::Duration(10));
  lst = &listener;
  tf2_ros::Buffer tf_buffer(ros::Duration(100));
  pbuffer = &tf_buffer;

  static tf2_ros::StaticTransformBroadcaster static_broadcaster;
  broad_caster = &static_broadcaster;

  // Synchronize darknet detection with pointcloud
  typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), sub_cloud, sub_box);
  sync.registerCallback(boost::bind(&cloud_cb, _1, _2));

  // Create a ROS publisher for the output point cloud
  pub_bottle = nh.advertise<sensor_msgs::PointCloud2> ("pcl_bottle", 1);
  pub_cup = nh.advertise<sensor_msgs::PointCloud2> ("pcl_cup", 1);

  // Create a ROS publisher for the output point cloud centroid markers
  pub_bottle_centroid = nh.advertise<visualization_msgs::Marker> ("bottle_centroids", 1);
  pub_cup_centroid = nh.advertise<visualization_msgs::Marker> ("cup_centroids", 1);

  //Create a ROS publisher for the detected  points
  pub_bottle_point= nh.advertise<geometry_msgs::PointStamped> ("/bottle_center", 1);
  pub_cup_point= nh.advertise<geometry_msgs::PointStamped> ("/cup_center", 1);

  // Create a ROS publisher for the detected poses
  pub_bottle_poses = nh.advertise<geometry_msgs::PoseArray> ("/bottle_poses", 1);
  pub_cup_poses = nh.advertise<geometry_msgs::PoseArray> ("/cup_poses", 1);

  // Create a ROS publisher for bottle position
  //pub_bottle_list = nh.advertise<pointcloud_processing_msgs::handle_position> ("detected_bottle", 1);
  pub_ObjectInfos = nh.advertise<pointcloud_processing_msgs::ObjectInfoArray> ("ObjectInfos", 1);

  ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && !received_first_message) {
        ROS_WARN_THROTTLE(2, "Waiting for image, depth and detections messages...");
        ros::spinOnce();
    }





  // Spin
  ros::spin ();
}
