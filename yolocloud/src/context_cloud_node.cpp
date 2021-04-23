#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

#include <sensor_msgs/Image.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>

#include <villa_yolocloud/ContextCloud.h>
#include <villa_yolocloud/DetectedObject.h>
#include <villa_yolocloud/GetShelfObjects.h>
#include <villa_yolocloud/GetEntities.h>
#include <villa_yolocloud/GetObjects.h>
#include <villa_yolocloud/BboxInfo.h>
#include <villa_yolocloud/ObjectInfo.h>
#include <villa_yolocloud/ObjectInfoArray.h>
#include <XmlRpcValue.h>
#include <yaml-cpp/yaml.h>

#define VISUALIZE true

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        darknet_ros_msgs::BoundingBoxes> SyncPolicy;


class Context_Manager {
private:
    ContextCloud context_manager;
    tf::TransformListener listener;
    std::unordered_map<int, int> entity_id_to_point;
    //knowledge_rep::LongTermMemoryConduit ltmc; // Knowledge base
    //std::map<std::string, int> current_list;


    ros::Publisher viz_pub;
    ros::Publisher Object_info_pub;
    ros::Publisher Object_info_FOV_pub;

public:
    bool received_first_message = false;
        //: ltmc(knowledge_rep::get_default_ltmc()) {
    Context_Manager(ros::NodeHandle node){
        viz_pub = node.advertise<visualization_msgs::MarkerArray>("yoloobjects/markers", 1, true);
        Object_info_pub = node.advertise<villa_yolocloud::ObjectInfoArray>("objects_info", 1, true);
        Object_info_FOV_pub = node.advertise<villa_yolocloud::ObjectInfoArray>("objects_info_FOV", 1, true);
        // NOTE: We assume there's only one ContextCloud, so this will blow away anything that is sensed
        //ltmc.get_concept("sensed").remove_instances();
        //ltmc.get_concept("scanned").remove_references();
        //loadparameters(node);
    }

    ~Context_Manager() {
        //ltmc.get_concept("sensed").remove_instances();
        //ltmc.get_concept("scanned").remove_references();
    }

    void data_callback(const sensor_msgs::Image::ConstPtr &rgb_image,
                       const sensor_msgs::Image::ConstPtr &depth_image,
                       //const tmc_yolo2_ros::Detections::ConstPtr &yolo_detections) {
                       const darknet_ros_msgs::BoundingBoxes::ConstPtr &yolo_detections) {
        received_first_message = true;
        //if (yolo_detections->detections.empty()) {
        if (yolo_detections->bounding_boxes.empty()) {
            return;
        }

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(rgb_image, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        tf::StampedTransform transform;
        try {
            //listener.waitForTransform("map", "head_rgbd_sensor_link", rgb_image->header.stamp, ros::Duration(0.1));
            //listener.lookupTransform("map", "head_rgbd_sensor_link", rgb_image->header.stamp, transform);
            listener.waitForTransform("map", "rgb_camera_link", rgb_image->header.stamp, ros::Duration(0.1));
            listener.lookupTransform("map", "rgb_camera_link", rgb_image->header.stamp, transform);
        } catch (tf::TransformException &ex){
            ROS_ERROR("%s",ex.what());
            return;
        }

        Eigen::Affine3d camToMap;
        tf::transformTFToEigen(transform, camToMap);

        cv::Mat depthI(depth_image->height, depth_image->width, CV_32FC1);
        memcpy(depthI.data, depth_image->data.data(), depth_image->data.size());


        //Current_list update
        context_manager.current_list.clear();
        for (const darknet_ros_msgs::BoundingBox&detection : yolo_detections->bounding_boxes) {

            if(detection.probability<0.5)
                continue;

            auto string_it = context_manager.current_list.find(detection.Class);
            if (string_it == context_manager.current_list.end())
               context_manager.current_list.insert({detection.Class, 1});
            else
                string_it->second+=1;
        }

        ROS_INFO("--------current detection-----------------");
        auto str_it = context_manager.current_list.begin();
        for(str_it=context_manager.current_list.begin(); str_it!=context_manager.current_list.end();str_it++)
            ROS_INFO("current dection class : %s, number of objects : %d", str_it->first.c_str(), str_it->second);


        //add_object
        for (const darknet_ros_msgs::BoundingBox&detection : yolo_detections->bounding_boxes) {
            ImageBoundingBox bbox;
            bbox.x = detection.xmin;
            bbox.y = detection.ymin;
            bbox.width = detection.xmax-detection.xmin;
            bbox.height = detection.ymax-detection.ymin;
            bbox.label = detection.Class;
            float confidence = detection.probability;

            if(confidence<0.5)
                continue;

            ROS_INFO("add object for %s", bbox.label.c_str());
	    if(bbox.label.c_str()!="sofa")
		    int idx = context_manager.addObject(bbox, cv_ptr->image, depthI, camToMap.cast<float>());
        }

        

            //std::vector<string>::iterator it = std::find(bbox.label);
            //if(it==current_list.end())
            
            //std::map<string, int>::iterator it = std::find(cur)
            //auto string_it = context_manager.current_list.find(bbox.label);
            //if (string_it == context_manager.current_list.end())
               //context_manager.current_list.insert({bbox.label, 1});
            //else

            //onlytargetset
            //for(size_t j=0; j<context_manager.TargetSet.size();j++)
                //if(context_manager.TargetSet[j].compare(bbox.label)==0)
                    //int idx = context_manager.addObject(bbox, cv_ptr->image, depthI, camToMap.cast<float>());

            //update object context
            
            //ROS_INFO("HERE--------object idx : %d ", idx);

            // If object was added to yolocloud, add to knowledge base
            //if (idx >= 0) {
                //add_to_ltmc(idx);
            //}
        

        publishdata();
        visualize();
#ifdef VISUALIZE
        visualize();
#endif
    }


    //void add_to_ltmc(int cloud_idx) {
        //std::string label = context_manager.labels.at(context_manager.objects->points[cloud_idx].label);
        //auto concept = ltmc.get_concept(label);
        //auto entity = concept.create_instance();
        //auto sensed = ltmc.get_concept("sensed");
        //entity.make_instance_of(sensed);
        //entity_id_to_point.insert({entity.entity_id, cloud_idx});
    //}

    bool get_objects(villa_yolocloud::GetObjects::Request &req, villa_yolocloud::GetObjects::Response &res) {
        
        std::vector<geometry_msgs::Point> map_locations;
        std::vector<std::string> labellist;
        std::vector<float> timelist;
        int numobjects=context_manager.getObjectsize();
        ROS_INFO("total size = %d", numobjects);
        std::map<std::string, std::vector<uint32_t> >::iterator mapiter
                                            =context_manager.labels_to_idset.begin(); 
        for(mapiter; mapiter != context_manager.labels_to_idset.end(); mapiter++)
        {
            //labellist.push_back(mapiter->first);
            for(size_t j=0; j< (int)((mapiter->second).size());j++)
            {
                auto idx = context_manager.id_to_position.find((mapiter->second)[j]);
                auto idtime=context_manager.id_to_time.find((mapiter->second)[j]);

                pcl::PointXYZL temppoint= idx->second;
                geometry_msgs::Point p_;
                p_.x=temppoint.x;
                p_.y=temppoint.y;
                p_.z=temppoint.z;
                map_locations.push_back(p_);
                labellist.push_back(mapiter->first);
                timelist.push_back(static_cast<ros::Time>(idtime->second).toSec());
            }
        }

        res.objectlist = labellist;
        res.map_locations = map_locations;
        res.timelist = timelist;
    }
        
    bool get_entities(villa_yolocloud::GetEntities::Request &req, villa_yolocloud::GetEntities::Response &res) {
        //std::vector<geometry_msgs::Point> map_locations;
        //for (int eid : req.entity_ids) {
            //auto points = context_manager.objects->points;
            //geometry_msgs::Point p;
             //Requested an ID that's not in the cloud. Return NANs
            //if (entity_id_to_point.count(eid) == 0) {
                //knowledge_rep::Entity entity = {eid, ltmc};
                //entity.remove_attribute("sensed");
                //p.x = NAN;
                //p.y = NAN;
                //p.z = NAN;
            //} else {
                //pcl::PointXYZL point = context_manager.objects->points[entity_id_to_point.at(eid)];
                //p.x = point.x;
                //p.y = point.y;
                //p.z = point.z;
            //}

            //map_locations.push_back(p);
        //}

        //res.map_locations = map_locations;

        return true;
    }


    bool get_shelf_objects(villa_yolocloud::GetShelfObjects::Request &req, villa_yolocloud::GetShelfObjects::Response &res) {
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = context_manager.sliceCloud(req.x_min, req.x_max, req.y_min, req.y_max, req.z_min, req.z_max);

        for (const auto &point : cloud->points) {
            villa_yolocloud::DetectedObject obj;
            obj.x = point.x;
            obj.y = point.y;
            obj.z = point.z;
            obj.label = context_manager.labels.at(point.label);
            res.objects.push_back(obj);
        }

        return true;
    }
    void publishdata(){
    
        villa_yolocloud::ObjectInfoArray InfoArray;
        villa_yolocloud::ObjectInfoArray InfoArray_FOV;

        std::map<std::string, std::vector<uint32_t> >::iterator mapiter
                                            =context_manager.labels_to_idset.begin(); 
        for(mapiter; mapiter != context_manager.labels_to_idset.end(); mapiter++)
        {
            //labellist.push_back(mapiter->first);
            ros::Time cur_time = ros::Time::now();
            
            for(size_t j=0; j< (int)((mapiter->second).size());j++)
            {
                auto idx = context_manager.id_to_position.find((mapiter->second)[j]);
                auto idtime=context_manager.id_to_time.find((mapiter->second)[j]);
                auto idbbox=context_manager.id_to_bbox.find((mapiter->second)[j]);
                auto iddepth=context_manager.id_to_depth.find((mapiter->second)[j]);

                pcl::PointXYZL temppoint= idx->second;
                geometry_msgs::Point p_;
                p_.x=temppoint.x;
                p_.y=temppoint.y;
                p_.z=temppoint.z;

                //generate ros messages
                villa_yolocloud::ObjectInfo objinfo;
                objinfo.label =mapiter->first;
                objinfo.id =idx->first;
                objinfo.point =p_;
                objinfo.bbox.x = (idbbox->second).x;
                objinfo.bbox.y = (idbbox->second).y;
                objinfo.bbox.width = (idbbox->second).width;
                objinfo.bbox.height = (idbbox->second).height;
                objinfo.header.stamp= idtime->second;
                objinfo.depth = iddepth->second;

                InfoArray.objectinfos.push_back(objinfo);

                if(abs((cur_time-objinfo.header.stamp).toSec())<3.0)
                    InfoArray_FOV.objectinfos.push_back(objinfo);

                //if (std::find(current_list.begin(), current_list.end(), bbox.label) != current_list.end())
                    //InfoArray_FOV.push_back(objinfo)

                
            }
        }

        //publish_objectinfo_msgs
        Object_info_pub.publish(InfoArray);
        Object_info_FOV_pub.publish(InfoArray_FOV);
        
    }


    void visualize() {
        visualization_msgs::MarkerArray yolo_marker_array;
        std::vector<uint32_t> idset;
        int count=0;
        std::map<std::string, std::vector<uint32_t> >::iterator mapiter
                                =context_manager.labels_to_idset.begin(); 
        for(mapiter; mapiter != context_manager.labels_to_idset.end(); mapiter++){

            idset= mapiter->second;
            for(size_t i=0;i<idset.size(); i++){
                auto indiv_id = context_manager.id_to_position.find(idset[i]);
                auto idtime=context_manager.id_to_time.find(idset[i]);
                auto object_point = indiv_id->second;

                //ros::Time cur_time = ros::Time::now();
                //auto time_duration = cur_time - idtime->second;
                //if(time_duration.toSec() > 15.0) 
                 //   continue;

                //for (const auto &object : context_manager.objects->points) {
                visualization_msgs::Marker marker;
                marker.header.frame_id = "map";
                //marker.ns = context_manager.labels.at(object.label);
                marker.ns = count;
                marker.id = 0;
                marker.type = 1;
                marker.action = 0;
                marker.pose.position.x = object_point.x;
                marker.pose.position.y = object_point.y;
                marker.pose.position.z = object_point.z;
                marker.pose.orientation.x = 0.;
                marker.pose.orientation.y = 0.;
                marker.pose.orientation.z = 0.;
                marker.pose.orientation.w = 1.;
                marker.scale.x = 0.25;
                marker.scale.y = 0.25;
                marker.scale.z = 0.25;
                marker.color.r = 1.;
                marker.color.b = 0.;
                marker.color.g = 0.;
                marker.color.a = 1.;
                marker.lifetime = ros::Duration(15.0);

                visualization_msgs::Marker text_marker;
                text_marker.header.frame_id = "map";
                text_marker.ns = count++;
                text_marker.id = 1;
                text_marker.type = 9;
                //text_marker.text = context_manager.labels.at(object.label);
                text_marker.text = mapiter->first;
                text_marker.action = 0;
                text_marker.pose.position.x = object_point.x;
                text_marker.pose.position.y = object_point.y;
                text_marker.pose.position.z = object_point.z + 0.10;
                text_marker.pose.orientation.x = 0.;
                text_marker.pose.orientation.y = 0.;
                text_marker.pose.orientation.z = 0.;
                text_marker.pose.orientation.w = 1.;
                text_marker.scale.x = 0.10;
                text_marker.scale.y = 0.10;
                text_marker.scale.z = 0.10;
                text_marker.color.r = 1.;
                text_marker.color.b = 0.;
                text_marker.color.g = 0.;
                text_marker.color.a = 1.;
                text_marker.lifetime = ros::Duration(15.0);

                yolo_marker_array.markers.push_back(marker);
                yolo_marker_array.markers.push_back(text_marker);
            }
        }        
        viz_pub.publish(yolo_marker_array);
    }

    void loadparameters(ros::NodeHandle n_)
    {
        XmlRpc::XmlRpcValue inputlist;
        n_.getParam("Interest_of_Objects", inputlist);
        ROS_INFO("load parameters");

        for(size_t i=0;i<inputlist.size();i++)
        {
            ROS_INFO ("\"target- = \"%s", static_cast<std::string>(inputlist[i]).c_str());
            context_manager.TargetSet.push_back(inputlist[i]);
        }

    }


};

int main (int argc, char **argv) {
    ROS_INFO("Initializing context node...");
    ros::init(argc, argv, "contextcloud_node");
    ros::NodeHandle n;

    Context_Manager yc_node(n);

    //message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/hsrb/head_rgbd_sensor/rgb/image_raw", 10);
    //message_filters::Subscriber<sensor_msgs::Image> depth_sub(n, "/hsrb/head_rgbd_sensor/depth_registered/image", 10);
    message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/rgb/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(n, "/depth/image_raw", 10);

    //message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/camera/rgb/image_color", 10);
    //message_filters::Subscriber<sensor_msgs::Image> depth_sub(n, "/camera/depth/image", 10);
    //message_filters::Subscriber<tmc_yolo2_ros::Detections> yolo_sub(n, "/yolo2_node/detections", 10);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> yolo_sub(n, "/darknet_ros/bounding_boxes", 10);
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(50), image_sub, depth_sub, yolo_sub);
    sync.registerCallback(boost::bind(&Context_Manager::data_callback, &yc_node, _1, _2, _3));


    ROS_INFO("Started. Waiting for inputs.");
    while (ros::ok() && !yc_node.received_first_message) {
        ROS_WARN_THROTTLE(2, "Waiting for image, depth and detections messages...");
        ros::spinOnce();
    }

    ros::ServiceServer getShelfObjects = n.advertiseService("getShelfObjects", &Context_Manager::get_shelf_objects,
                                                            &yc_node);
    ros::ServiceServer getEntities = n.advertiseService("getEntities", &Context_Manager::get_entities, &yc_node);

    ros::ServiceServer getObjects= n.advertiseService("getObjects", &Context_Manager::get_objects, &yc_node);

    ros::spin();
    return 0;
}
