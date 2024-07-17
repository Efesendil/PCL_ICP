#include <iostream>
#include <chrono>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <eigen3/Eigen/Dense>

class Filter{

    public:

        pcl::PointCloud<pcl::PointXYZ>::Ptr RadiusOutlier(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

            auto time_start = std::chrono::system_clock::now();

            std::cerr << "\n::RadiusOutlierRemoval\n";

            std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;

            pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
            outrem.setInputCloud(cloud);
            outrem.setRadiusSearch(1.0);
            outrem.setMinNeighborsInRadius(12);
            outrem.setKeepOrganized(false);
            // apply filter
            outrem.filter (*cloud_filtered);
            
            std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;

            auto time_end = std::chrono::system_clock::now();
            auto diff = time_end - time_start;
            std::cout << "Radius Outlier Filtering took " << diff.count()/(1e9) << " seconds.\n";

            return cloud_filtered;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr VoxelGrid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
            
            auto time_start = std::chrono::system_clock::now();
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

            std::cerr << "\n::VoxelDownSampling\n";

            std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height << " data points (" << pcl::getFieldsList (*cloud) << ")." << std::endl;

            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud (cloud);
            sor.setLeafSize (0.1f, 0.1f, 0.1f);
            sor.filter (*cloud_filtered);

            std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")." << std::endl;

            auto time_end = std::chrono::system_clock::now();
            auto diff = time_end - time_start;
            std::cerr << "Voxel Grid Filtering took " << diff.count()/(1e9) << " seconds.\n";

            return cloud_filtered;
        }


};

class Registration{
    /*

        1. from a set of points, identify interest points (i.e., keypoints) that best represent the scene in both datasets;

        2. at each keypoint, compute a feature descriptor;

        3. from the set of feature descriptors together with their XYZ positions in the two datasets, estimate a set of correspondences, based on the similarities between features and positions;

        4. given that the data is assumed to be noisy, not all correspondences are valid, so reject those bad correspondences that contribute negatively to the registration process;

        5. from the remaining set of good correspondences, estimate a motion transformation.

    */
    public:
        pcl::PointCloud<pcl::PointXYZ>::Ptr ComputeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){

            auto time_start = std::chrono::system_clock::now();

            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
            norm_est.setSearchMethod (tree);
            norm_est.setKSearch (30);
            norm_est.setInputCloud (cloud);
            norm_est.compute (*cloud_normals);
            pcl::copyPointCloud (*cloud, *cloud_normals);

            auto time_end = std::chrono::system_clock::now();
            auto diff = time_end - time_start;
            std::cerr << "Computing normals took " << diff.count()/(1e9) << " seconds.\n";

            return cloud;

        }

        Eigen::Matrix4f icp_pose(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target){

            auto time_start = std::chrono::system_clock::now();

            pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;
            //icp.setTransformationEpsilon (1e-6);
            icp.setMaxCorrespondenceDistance (0.01);  
            icp.setMaximumIterations(5);
            icp.setInputSource(source);
            icp.setInputTarget(target);
            pcl::PointCloud<pcl::PointXYZ> Final;
            icp.align(Final);

            std::cout << "ICP has " << (icp.hasConverged()?"converged":"not converged") << ", score: " <<
            icp.getFitnessScore() << std::endl;

            Eigen::Matrix4f Ti = (icp.getFinalTransformation());

            auto time_end = std::chrono::system_clock::now();
            auto diff = time_end - time_start;
            std::cerr << "ICP took " << diff.count()/(1e9) << " seconds.\n";

            return Ti.inverse();
        }
        
};