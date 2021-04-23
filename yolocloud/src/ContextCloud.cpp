#include "villa_yolocloud/ContextCloud.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>

using namespace cv;
using namespace std;

const float NEIGHBOR_THRESH = 0.1;
const float MAXIMUM_DISTANCE= 1000.0;
const float INDIV_DISTANCE= 1000.0;
const float SURVIVAL_TIME = 60.0;



int ContextCloud::getObjectsize()
{
    int objectsize=0;
    std::map<std::string, std::vector<uint32_t> >::iterator mapiter
                                            =labels_to_idset.begin(); 
    for(mapiter; mapiter != labels_to_idset.end(); mapiter++)
    {
        //ROS_INFO("I am in this step: %s", mapiter->first.c_str());
        //ROS_INFO("the number of this object: %d", mapiter->second.size());
        objectsize+=(mapiter->second).size();

    }

    ROS_INFO("object_size %d", objectsize);

    return objectsize;

}

/*
 * Adds object
 * Returns index in point cloud or -1 if not added
 */

int ContextCloud::addObject(const ImageBoundingBox &bbox,
        const Mat &rgb_image,
        const Mat &depth_image,
        const Eigen::Affine3f &camToMap) {
    if (bbox.y + bbox.height > rgb_image.rows
        || bbox.x + bbox.width > rgb_image.cols) {
        //std::cout << bbox.x << std::endl;
        //std::cout << bbox.y << std::endl;
        //std::cout << bbox.width << std::endl;
        //std::cout << bbox.height << std::endl;
        ROS_INFO("No valid box for %s", bbox.label.c_str());
        //throw std::runtime_error("Invalid bbox");
        return -1;
    }
    if(bbox.label=="sofa")
	    return -1;

    // define bounding rectangle
    Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);
    //Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);


    Mat mask; // segmentation result (4 possible values)
    Mat bgModel, fgModel; // the models (internally used)


    ROS_INFO("before-grabcut");
    // GrabCut segmentation
    grabCut(rgb_image,    // input image
                mask,   // segmentation result
                rect,// rectangle containing foreground 
                bgModel, fgModel, // models
                1,        // number of iterations
                GC_INIT_WITH_RECT); // use rectangle


    ROS_INFO("after-grabcut");

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
    std::vector<float> depthvector;
    float depth = 0.;
    int count = 0;
    float nearest=0.;
    for (size_t y = bbox.y; y < bbox.y + bbox.height; y++) {
        for (size_t x = bbox.x; x < bbox.x + bbox.width; x++) {
            uint8_t mask_val = mask.at<uint8_t>(y, x);
            float cur_depth = depth_image.at<float>(y, x);
            if (cur_depth != 0 && !isnan(cur_depth) && (mask_val == GC_PR_FGD || mask_val == GC_FGD)) {
                depth += cur_depth;
                count++;
                //roundoff by 2 decimal points of deapth data
                nearest =roundf(cur_depth*100) /100.0;
                depthvector.push_back(nearest);
            }
        }
    }
    depth /= count;

    ROS_INFO("after-grabcut-depth");
    //float commonest = std::most_common(depthvector.begin(), depthvector.end());

    /*
    int co=0;
    int nmax=0;
    //fid the most common value in depth vector
    if(depthvector.size()>0)
    {
        float mostvalue=depthvector[0];
        for(int i=0;i<depthvector.size();i++)
        {
            co = (int) std::count(depthvector.begin(), depthvector.end(), depthvector[i]);
            if(co > nmax)
            {       nmax = co;
                mostvalue = depthvector[i];
            }
        }

    //ROS_INFO("avg : %.3lf, most_common: %.3lf ", depth, mostvalue);
    depth = mostvalue;
    }
    */


    // No valid depth, so return
    if (count == 0) {
        ROS_INFO("no valid depth for%s ", bbox.label.c_str());
        return -1;
    }

    ROS_INFO("depth: %.2lf", depth);
   // Compute 3d point in the world
    float x_center = bbox.x + bbox.width/2.;
    float y_center = bbox.y + bbox.height/2.;
    float camera_x = (x_center - intrinsic_cx) * (1.0 / intrinsic_sx) * depth;
    float camera_y = (y_center - intrinsic_cy) * (1.0 / intrinsic_sy) * depth;
    float camera_z = depth;
    Eigen::Vector3f world = camToMap * Eigen::Vector3f(camera_x, camera_y, camera_z);
    ROS_INFO("get 3d points");

    pcl::PointXYZL point;
    point.x = world(0);
    point.y = world(1);
    point.z = world(2);
    //point.label = id;

    // If the label does not yet exist, add it
    // Else if the labe does exist, check the position
    //float secs =ros::Time::now().toSec();
    ros::Time secs =ros::Time::now();
    auto it = labels_to_idset.find(bbox.label);
    uint32_t id=0;
    ROS_INFO("step2 : add index for %s ", bbox.label.c_str());
    std::vector<uint32_t> idset;
    if (it == labels_to_idset.end()) {
        //id = labels.size();
        ROS_INFO("add index for %s ", bbox.label.c_str());
        
        id = getObjectsize();
        point.label = id;
        idset.push_back(id);
        labels_to_idset.insert({bbox.label, idset});
        id_to_position.insert({id, point});
        id_to_time.insert({id, secs});
        id_to_bbox.insert({id, bbox});
        id_to_depth.insert({id, camera_z});
        labels.push_back(bbox.label);
        objects->push_back(point);

    } else {
        //check position
        //
        ROS_INFO("step3 : add index for %s ", bbox.label.c_str());
        auto ex_idset= it->second;
        float closest_distance = MAXIMUM_DISTANCE;
        int closest_idx=0;
        //ROS_INFO("check ex_idset size : %d ", ex_idset.size());

        for(size_t i=0;i<ex_idset.size(); i++) 
        {
            //auto indiv_id = id_to_position.find(ex_idset[i]);
            //auto object_point = indiv_id->second;
            
            auto indiv_id = id_to_bbox.find(ex_idset[i]);
            auto object_bbox= indiv_id->second;
            auto indiv_id_depth = id_to_depth.find(ex_idset[i]);
            auto object_depth= indiv_id_depth->second;

            float distance_=0.0;
            distance_+=pow((x_center-(float)(object_bbox.x+object_bbox.width/2)),2);
            distance_+=pow((y_center-(float)(object_bbox.y+object_bbox.height/2)),2);
            distance_+=pow(50*(object_depth-camera_z),2);
            distance_=sqrt(distance_);

            if(distance_ < closest_distance)
            {
                //ROS_INFO("bbox.x: %f, bbox.y : %f , object_bbox.x : %f, object_bbox.y:%f, dist : %.3lf"
                                //,bbox.x, bbox.y, object_bbox.x, object_bbox.y, distance_);
                closest_distance=distance_;
                closest_idx=ex_idset[i];
            }
        }

        //If the distance is bigger than threshold, add new info
        //ROS_INFO("check distance for %s ", bbox.label.c_str());
        //ROS_INFO("object: %s, closest distance : %.3lf", bbox.label.c_str(),closest_distance);
        if (closest_distance < INDIV_DISTANCE )
        {
            //ROS_INFO("position updated for existing object");
            point.label=(id_to_position[closest_idx]).label;
            //ROS_INFO("position updated for %d", point.label);
            id_to_position[closest_idx]=point;
            //ROS_INFO("size of id_to_position:  %d", id_to_position.size());
            id_to_time[closest_idx]= secs;
            id_to_bbox[closest_idx]= bbox;
            id_to_depth[closest_idx]= camera_z;
            //return -1;
        }
        else
        {
           id = getObjectsize();
           point.label = id;
           it->second.push_back(id);
           //ROS_INFO("an %d object with another position added", point.label);
           id_to_position.insert({id, point});
           id_to_time.insert({id, secs});
           id_to_bbox.insert({id, bbox});
           id_to_depth.insert({id, camera_z});
           ROS_INFO("size of id_to_position:  %d", id_to_position.size());
           objects->push_back(point);

        }
    }

/*
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


*/

    std::vector<int> erase_index;
    ros::Time current_time = ros::Time::now();

    ROS_INFO("----current label & id to position---" );
    std::map<std::string, std::vector<uint32_t> >::iterator mapiter
                                            =labels_to_idset.begin(); 
        for(mapiter; mapiter != labels_to_idset.end(); mapiter++)
        {
            //labellist.push_back(mapiter->first);
            for(size_t j=0; j< (int)((mapiter->second).size());j++)
            {
                auto idx = id_to_position.find((mapiter->second)[j]);
                auto idtime=id_to_time.find((mapiter->second)[j]);
                auto iddepth=id_to_depth.find((mapiter->second)[j]);

                ROS_INFO("label: %s, id:%d, position x: %.3lf, y: %.3lf, z: %.3lf,d: %.3lf",
                mapiter->first.c_str(), (mapiter->second)[j], (idx->second).x, (idx->second).y, (idx->second).z, (iddepth->second));

                float dur = (current_time-idtime->second).toSec();
                //ROS_INFO("current duration : %.3lf",dur );
                if((current_time-idtime->second).toSec()>SURVIVAL_TIME)
                {
                    id_to_position.erase(idx);
                    id_to_time.erase(idtime);
                    id_to_depth.erase(iddepth);
                    erase_index.push_back(idx->first);
                    ROS_INFO("Index %d, class %s is erased!!! --out of time", idx->first, mapiter->first.c_str() );
                }
            }
        }

        //
        mapiter=labels_to_idset.begin(); 
        for(mapiter; mapiter != labels_to_idset.end(); mapiter++)
        {
            for(size_t i=0; i<erase_index.size();i++)
            {
                std::vector<uint32_t>::iterator pos = std::find((mapiter->second).begin(), 
                                                        (mapiter->second).end(),erase_index[i]);
                if(pos != mapiter->second.end())
                {

                    ROS_INFO("Index %d is erased from ex_idset!!! --out of time", static_cast<uint32_t>(*pos));
                    (mapiter->second).erase(pos);
                }
            
            }
        
        }

    //ROS_INFO("----current id to position---" );
    //std::map<uint32_t, pcl::PointXYZL >::iterator idmap_iter = id_to_position.begin();
    //for (idmap_iter;idmap_iter!=id_to_position.end();idmap_iter++)
    //{   

        //ROS_INFO("id_to_position key : %d, x: %.3lf, y: %3lf, z: %3lf,",
                //idmap_iter->first, (idmap_iter->second).x, (idmap_iter->second).y, (idmap_iter->second).z);
    //}


    //ROS_INFO("objects->size()::::::::%d", objects->size());
    //objects->push_back(point);
    return objects->size() - 1;
    //return getObjectsize();
}

pcl::PointCloud<pcl::PointXYZL>::Ptr ContextCloud::sliceCloud(float x_min, float x_max,
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

