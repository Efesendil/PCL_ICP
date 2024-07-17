#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/memory.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <eigen3/Eigen/Dense>

class IO{

    public:

        auto read_pcd(std::string const& pcd_path){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);    
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_path, *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        }
        std::cout << "Loaded "
                    << cloud->width * cloud->height
                    << " data points from test_pcd.pcd with the following fields: "
                    << std::endl;
        return(cloud);
        /*for (const auto& point: *cloud){

            std::cout << "    " << point.x

                    << " "    << point.y

                    << " "    << point.z 
                    
                    << " "    << point.intensity << std::endl;
        }*/
        }

    void write_pcd(std::string const& pcd_path, pcl::PointCloud<pcl::PointXYZ> cloud){
        pcl::io::savePCDFileASCII (pcd_path, cloud);
        std::cout << "File written at : " << pcd_path << "\n";
    }

    void printUsage(const char* progName){

            std::cout << "\n\nUsage: "<<progName<<" [options]\n\n"

            << "Options:\n"
            << "-------------------------------------------\n"

            << "-h           this help\n"

            << "-s           Simple visualisation example\n"

            << "-w           Pointcloud writing mode\n"
           
            << "-rgb         RGB colour visualisation example\n"

            << "-c           Custom colour visualisation example\n"

            << "-v           Viewports example\n"

            << "-n           Normals visualisation example\n"

            << "-i           Interaction Customization example\n"

            << "\n\n";
    }

    pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2){
        
        viewer->setBackgroundColor (0, 0, 0);

        viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
        viewer->addPointCloud<pcl::PointXYZ> (cloud2, "sample cloud2");

        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud2");

        viewer->addCoordinateSystem (1.0);

        viewer->initCameraParameters ();

        return (viewer);

    }

    
    pcl::visualization::PCLVisualizer::Ptr viewportsVis (pcl::visualization::PCLVisualizer::Ptr viewer, 
        pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud2,
        pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud1_rgb, pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud2_rgb){

    viewer->initCameraParameters ();

    int v1(0);

    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);

    viewer->setBackgroundColor (0, 0, 0, v1);

    viewer->addText("Radius: 0.1", 10, 10, "v1 text", v1);

    viewer->addPointCloud<pcl::PointXYZ> (_cloud1, "sample cloud 1", v1);
    viewer->addPointCloud<pcl::PointXYZ> (_cloud2, "sample cloud 2", v1);


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud1_rgb->points.resize(_cloud1_rgb->points.size());
    cloud2_rgb->points.resize(_cloud2_rgb->points.size());
    for (size_t i = 0; i < _cloud1_rgb->points.size(); ++i){

        pcl::PointXYZRGB point1;
        point1.x = _cloud1_rgb->points[i].x;
        point1.y = _cloud1_rgb->points[i].y;
        point1.z = _cloud1_rgb->points[i].z;

        cloud1_rgb->points[i] = point1;
        }

    for (size_t i = 0; i < _cloud2_rgb->points.size(); ++i){

        pcl::PointXYZRGB point2;
        point2.x = _cloud2_rgb->points[i].x;
        point2.y = _cloud2_rgb->points[i].y;
        point2.z = _cloud2_rgb->points[i].z;

        cloud2_rgb->points[i] = point2;
        }
    
    int v2(0);

    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    viewer->setBackgroundColor (0.3, 0.3, 0.3, v2);

    viewer->addText("Radius: 0.1", 10, 10, "v2 text", v2);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color1(cloud1_rgb, 0, 255, 0); //GREEN
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color2(cloud2_rgb, 255, 0, 0); //RED

    viewer->addPointCloud<pcl::PointXYZRGB> (cloud1_rgb, single_color1, "sample cloud rgb 1", v2);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud2_rgb, single_color2, "sample cloud rgb 2", v2);

    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud rgb 1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud rgb 2");

    viewer->addCoordinateSystem (1.0);

    return (viewer);

}   

};

