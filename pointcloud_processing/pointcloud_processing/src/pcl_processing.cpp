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

// Set fixed reference frame
std::string fixed_frame = "map";

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

  ROS_INFO("cloud callback");
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
  bottle_poses.header.frame_id = cup_poses.header.frame_id = fixed_frame;

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
  int width = input_cloud->width;
  int height = input_cloud->height;
  //int width = cloud->width;
  //int height = cloud->height;
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
    ROS_INFO("object_name: %s, xmin: %d, xmax: %d, ymin: %d, ymax: %d, probability: %.2lf", object_name.c_str(), xmin, xmax, ymin, ymax, probability );
    if(probability<0.75)
        continue;

    // -------------------ROI extraction------------------------------

    // Initialize container for inliers of the ROI
    pcl::PointIndices::Ptr inliers_roi (new pcl::PointIndices());

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
    ROS_INFO("here-1.1");
    extract_roi.filter (*cloud_filtered_roi);
    //extract_roi.filter (*cloud);
    ROS_INFO("here-1.2");
    
    //ROS_INFO("here-1.2");
    // ----------------------VoxelGrid----------------------------------
    // Perform the downsampling
    pcl::VoxelGrid<pcl::PointXYZRGB> sor_voxelgrid;
    sor_voxelgrid.setInputCloud (cloud_filtered_roi);
    sor_voxelgrid.setLeafSize (0.01, 0.01, 0.01); //size of the grid
    sor_voxelgrid.filter (*cloud_filtered_voxelgrid);


   // ---------------------StatisticalOutlierRemoval--------------------
   // Create the 
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_noise;
   // Remove noise
   //sor_noise.setInputCloud (cloud_filtered_roi);
   sor_noise.setInputCloud (cloud_filtered_voxelgrid);
   sor_noise.setMeanK (50);
   sor_noise.setStddevMulThresh (1.0);
   sor_noise.filter (*cloud_filtered_sor);

   pcl::IndicesPtr indices_xyz(new std::vector <int>);
   pcl::PassThrough<pcl::PointXYZRGB> pass;
   pass.setInputCloud(cloud_filtered_sor);
   pass.setFilterFieldName("z");
   pass.setFilterLimits(0.0,3.0);
   pass.filter(*indices_xyz);

   pcl::ExtractIndices<pcl::PointXYZRGB> extract_roi2;
    // Extract the inliers  of the ROI
    extract_roi2.setInputCloud (cloud_filtered_sor);
    extract_roi2.setIndices (indices_xyz);
    extract_roi2.setNegative (false);
    extract_roi2.filter (*cloud_filtered_sor);
  

  ROS_INFO("here-2");
   //remove NaN points from the cloud
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr nanfiltered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
   std::vector<int> rindices;
   pcl::removeNaNFromPointCloud(*cloud_filtered_sor,*nanfiltered_cloud, rindices);

   float avg_depth =0;
   float nearest=0.0;
   float depth_maxcount=0.0;
   int depth_count =nanfiltered_cloud->points.size();
   //depthvector.resize(depth_count);
   for(size_t i=0;i<nanfiltered_cloud->points.size();i++)
   {
       //std::cout<<"inside clouds"<<nanfiltered_cloud->points[i].z<<std::endl;
       float cur_depth = nanfiltered_cloud->points[i].z;
       avg_depth+=nanfiltered_cloud->points[i].z;
       nearest =roundf(cur_depth*1000) /1000.0;
   }

   //If depth value is not reliable, don't use this value to update target object information
   if(depth_count>0)
       avg_depth =avg_depth/depth_count;
   else
       continue;
    
   if(Is_target(object_name) && !std::isnan(avg_depth) )
   {
       auto it_depthvector = depth_buffer.find(object_name);
       if(it_depthvector !=depth_buffer.end() )
           if(it_depthvector->second.size()>QUEUE_SIZE)
           {
               it_depthvector->second.push_back(avg_depth);
               it_depthvector->second.pop_front();
           }
           else
               it_depthvector->second.push_back(avg_depth);
   }

   if(Is_target(object_name) && avg_depth ==0.0)
   {
       auto it_depthvector = depth_buffer.find(object_name);
        if(it_depthvector !=depth_buffer.end() )
        {
           if(it_depthvector->second.size()>0)
           {
               float avg_buffer=0.0;
               int   avg_count=0;
               avg_depth =0.0;
               for(size_t j(0);j<it_depthvector->second.size();j++)
               {
                   if(std::isnan(it_depthvector->second[j]) && (it_depthvector->second[j]!=0.0))
                   {
                       avg_buffer+=it_depthvector->second[j];
                       avg_count++;
                   }
               }
               if(avg_count>0)
                   avg_buffer = avg_buffer/avg_count;

               avg_depth= avg_buffer ;
               //avg_depth =(it_depthvector->second[it_depthvector->second.size()-10]);
               ROS_INFO("%s depath is nan, so, put into deque value: %.3f",object_name.c_str(), avg_buffer);

           }
            else
            {
               avg_depth = 0.0;
               ROS_INFO("%s depath is nan, so, put into Zero----last casee: %.3f",object_name.c_str(), avg_depth);
            }
        }
   }
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

  
  lst->waitForTransform(input_cloud->header.frame_id, fixed_frame, ros::Time(0), ros::Duration(2.0));
  try{lst->transformPoint(fixed_frame,centroid_rel, centroid_abs);}
  //ROS_INFO("succedd");}
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

  pointcloud_processing_msgs::ObjectInfo cur_obj;
  cur_obj.x = xmin;
  cur_obj.y = ymin;
  cur_obj.width = xmax-xmin;
  cur_obj.height = ymax-ymin;
  cur_obj.label = object_name;
  cur_obj.last_time = ros::Time::now();
  cur_obj.average_depth=avg_depth;
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
    *cloud_bottle += *cloud_filtered_sor;
    centroid_bottle_list.points.push_back(centroid_rel.point);
    bottle_poses.poses.push_back(object_pose.pose);
    is_bottle = true;
  }
  else if (object_name == "cup"){
     //pub_cup_point.publish(centroid_rel);
      //pub_cup_point.publish(centroid_rel);
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
  // Create a container for the result data.
  //sensor_msgs::PointCloud2 output_bottle;
  //sensor_msgs::PointCloud2 output_cup;

  // Convert pcl::PointCloud to sensor_msgs::PointCloud2
  //pcl::toROSMsg(*cloud_bottle,output_bottle);
  //pcl::toROSMsg(*cloud_cup,output_cup);

  // Set output frame as the input frame
  //output_bottle.header.frame_id           = input_cloud->header.frame_id;
  //output_cup.header.frame_id      = input_cloud->header.frame_id;

  // Publish the data.
  //pub_bottle.publish (output_bottle);
  //pub_cup.publish (output_cup);

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

  target_string.push_back("bottle");
  target_string.push_back("cup");

  std::deque<float> depthvector;
  depth_buffer.insert({"bottle", depthvector});
  depth_buffer.insert({"cup", depthvector});

  // Initialize subscribers to darknet detection and pointcloud
  //message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "/hsrb/head_rgbd_sensor/depth_registered/rectified_points", 1);
  message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> sub_box(nh, "/darknet_ros/bounding_boxes", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud(nh, "/points2", 1);

  // Initialize transform listener
  tf::TransformListener listener(ros::Duration(10));
  lst = &listener;
  tf2_ros::Buffer tf_buffer(ros::Duration(50));
  pbuffer = &tf_buffer;

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
