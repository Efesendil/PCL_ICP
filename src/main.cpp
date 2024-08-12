#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h> // for pcl::console::parse_argument
#include "pipeline.h"

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

 std::vector<std::string> get_pcd_files(const std::string& pcd_folder_path)
{
    std::vector<std::string> pcd_files;

    if (fs::exists(pcd_folder_path) && fs::is_directory(pcd_folder_path))
    {
        for (const auto& entry : fs::directory_iterator(pcd_folder_path))
        {
            auto extension = entry.path().extension();
            if (extension == ".pcd")
            {
                pcd_files.push_back(entry.path().string());
            }
        }
    }

    std::sort(pcd_files.begin(), pcd_files.end());

    return pcd_files;
}

int main(int argc, char **argv) {
    // Check the number of parameters
    // if (argc < 3) {
    //     std::cerr << "Usage: " << argv[0] << " source_cloud.pcd target_cloud.pcd" << std::endl;
    //     return -1;
    // }

    // Load point clouds
    PointCloudT::Ptr source_cloud(new PointCloudT);
    PointCloudT::Ptr target_cloud(new PointCloudT);
    
    // pcl::io::loadPCDFile(argv[1], *source_cloud);
    // pcl::io::loadPCDFile(argv[2], *target_cloud);

    std::vector<std::string> raw_pcds = get_pcd_files("/home/efesendil/Data/mms_pcd_samples/");
    std::vector<Eigen::Matrix4f> poses; // Assume poses are populated with 4x4 transformation matrices
    std::vector<Eigen::Matrix4f> poses_traj;

    poses_traj.push_back(Eigen::Matrix4f::Identity());

    PointCloudT::Ptr global_map(new PointCloudT);


    for(int i = 1; i < 50 /*raw_pcds.size()*/; i++){


        pcl::io::loadPCDFile(raw_pcds[i-1], *target_cloud);
        pcl::io::loadPCDFile(raw_pcds[i], *source_cloud);

        if (i == 1)
        {
            *global_map = *target_cloud;
        }

        PointCloudT::Ptr source_cpy(new PointCloudT(*source_cloud));
        PointCloudT::Ptr target_cpy(new PointCloudT(*target_cloud));

        // Preprocess point clouds
        preprocessPointCloud(source_cloud);
        preprocessPointCloud(target_cloud);

        //visualizePointClouds(source_cloud, target_cloud, 1);

        // Keypoint estimation
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
        estimateKeypoints(source_cloud, source_keypoints);
        estimateKeypoints(target_cloud, target_keypoints);

        //visualizePointClouds(source_keypoints, source_cloud, 8);
        //visualizePointClouds(target_keypoints, target_cloud, 8);

        // Descriptor computation
        pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
        DescriptorCloudT::Ptr source_descriptors(new DescriptorCloudT);
        DescriptorCloudT::Ptr target_descriptors(new DescriptorCloudT);

        computeDescriptors(source_cloud, source_keypoints, source_normals, source_descriptors);
        computeDescriptors(target_cloud, target_keypoints, target_normals, target_descriptors);

        // Correspondence estimation
        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
        estimateCorrespondences(source_descriptors, target_descriptors, correspondences);
        //visualizeCorrespondences(source_cloud, target_cloud, correspondences);

        // Correspondence rejection using ransac
        pcl::CorrespondencesPtr inliers(new pcl::Correspondences);
        rejectCorrespondencesRANSAC(source_keypoints, target_keypoints, correspondences, inliers);

        // Transformation estimation
        Eigen::Matrix4f transformation_rough;
        estimateTransformation(source_keypoints, target_keypoints, inliers, transformation_rough);

        // Apply transformation to the source keypoint
        Eigen::Matrix4f transformation_keypoint;
        pcl::PointCloud<PointT>::Ptr result_keypoint (new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*source_keypoints, *result_keypoint, transformation_rough);

        // Registration refinement
        Eigen::Matrix4f transformation_fine;
        pcl::PointCloud<PointT>::Ptr result (new pcl::PointCloud<PointT>());
        refineRegistration(result_keypoint, target_cloud, transformation_fine);

        // Apply transformation to the original cloud 
        Eigen::Matrix4f transformation_original = transformation_fine * transformation_rough;
        pcl::PointCloud<PointT>::Ptr result_original (new pcl::PointCloud<PointT>());
        //pcl::transformPointCloud(*source_cloud, *result, transformation_original);
        pcl::transformPointCloud(*source_cpy, *result_original, transformation_original);
        pcl::transformPointCloud(*result_original, *result_original, poses_traj.back());

        *global_map += *result_original;

        //pcl::io::savePCDFileBinary("../pcds/result_refine_icp.pcd", *result_original);

        // if(i % 10 == 0){
        //     visualizeCorrespondences(source_cloud, target_cloud, correspondences);
        //     visualizePointClouds(result_original, target_cpy, 1);
        // }

        Eigen::Matrix4f pose_global = poses_traj.back() *  transformation_original;     
        poses_traj.push_back(pose_global);
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& pose : poses_traj) {
            //cout << pose << endl;
            pcl::PointXYZ point;
            Eigen::Vector3f pos = pose.block<3, 1>(0, 3);
            point.x = pos(0);
            point.y = pos(1);
            point.z = pos(2);
            cloud->points.push_back(point);
        }

        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;

        // Save poses to a PLY file
        std::string output_ply_file = "../pcds/traj_odometry.ply";
        pcl::PLYWriter writer;
        writer.write<pcl::PointXYZ>(output_ply_file, *cloud, true);

        // Save global map to a PCD file
        pcl::io::savePCDFileBinary("../pcds/global_map.pcd", *global_map);


    // PointCloudT::Ptr source_cpy(new PointCloudT(*source_cloud));
    // PointCloudT::Ptr target_cpy(new PointCloudT(*target_cloud));

    // // Preprocess point clouds
    // preprocessPointCloud(source_cloud);
    // preprocessPointCloud(target_cloud);

    // visualizePointClouds(source_cloud, target_cloud, 1);

    // // Keypoint estimation
    // pcl::PointCloud<pcl::PointXYZ>::Ptr source_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr target_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    // estimateKeypoints(source_cloud, source_keypoints);
    // estimateKeypoints(target_cloud, target_keypoints);

    // visualizePointClouds(source_keypoints, source_cloud, 8);
    // visualizePointClouds(target_keypoints, target_cloud, 8);

    // // Descriptor computation
    // pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
    // pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
    // DescriptorCloudT::Ptr source_descriptors(new DescriptorCloudT);
    // DescriptorCloudT::Ptr target_descriptors(new DescriptorCloudT);

    // // computeDescriptors(source_cloud, source_keypoints, source_normals, source_descriptors);
    // // computeDescriptors(target_cloud, target_keypoints, target_normals, target_descriptors);

    // // computeNormals(source_cloud, source_normals);
    // // computeNormals(target_cloud, target_normals);

    // // multi_scale_keypoint_and_descriptor(source_cloud, source_normals, source_descriptors, source_keypoints);
    // // multi_scale_keypoint_and_descriptor(target_cloud, target_normals, target_descriptors, target_keypoints);

    // computeDescriptors(source_cloud, source_keypoints, source_normals, source_descriptors);
    // computeDescriptors(target_cloud, target_keypoints, target_normals, target_descriptors);

    // // pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    // // estimateCorrespondences(source_descriptors, target_descriptors, correspondences);
    // // visualizeCorrespondences(source_cloud, target_cloud, correspondences);

    // // // Transformation estimation
    // // Eigen::Matrix4f transformation_rough;
    // // estimateTransformation(source_keypoints, target_keypoints, correspondences, transformation_rough);

    // // // Apply the transformation to the source cloud
    // // pcl::PointCloud<PointT>::Ptr transformed_source_cloud(new pcl::PointCloud<PointT>());
    // // pcl::transformPointCloud(*source_cloud, *transformed_source_cloud, transformation_rough);

    // // visualizePointClouds(transformed_source_cloud, target_cloud, 1);

    // // pcl::io::savePCDFileBinary("../pcds/result_rough_alignment.pcd", *transformed_source_cloud);

    // // Correspondence estimation
    // pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    // estimateCorrespondences(source_descriptors, target_descriptors, correspondences);
    // visualizeCorrespondences(source_cloud, target_cloud, correspondences);

    // // Correspondence rejection using multiple methods
    // pcl::CorrespondencesPtr inliers(new pcl::Correspondences);
    // rejectCorrespondencesRANSAC(source_keypoints, target_keypoints, correspondences, inliers);
    // //rejectCorrespondencesDistance(source_descriptors, target_descriptors, correspondences, inliers);
    // //rejectCorrespondencesSurfaceNormal(source_keypoints, target_keypoints, correspondences, inliers);
    
    // // Transformation estimation
    // Eigen::Matrix4f transformation_rough;
    // estimateTransformation(source_keypoints, target_keypoints, inliers, transformation_rough);

    // // Apply transformation to the source keypoint
    // Eigen::Matrix4f transformation_keypoint;
    // pcl::PointCloud<PointT>::Ptr result_keypoint (new pcl::PointCloud<PointT>());
    // pcl::transformPointCloud(*source_keypoints, *result_keypoint, transformation_rough);

    // // Registration refinement
    // Eigen::Matrix4f transformation_fine;
    // pcl::PointCloud<PointT>::Ptr result (new pcl::PointCloud<PointT>());
    // refineRegistration(result_keypoint, target_cloud, transformation_fine);

    // // Apply transformation to the original cloud 
    // Eigen::Matrix4f transformation_original = transformation_fine * transformation_rough;
    // pcl::PointCloud<PointT>::Ptr result_original (new pcl::PointCloud<PointT>());
    // //pcl::transformPointCloud(*source_cloud, *result, transformation_original);
    // pcl::transformPointCloud(*source_cpy, *result_original, transformation_original);

    // pcl::io::savePCDFileBinary("../pcds/result_refine_icp.pcd", *result_original);

    // visualizePointClouds(result_original, target_cloud, 1);

    // // Apply transformation to the original cloud 
    // // Registration refinement
    // Eigen::Matrix4f transformation_test;
    // pcl::PointCloud<PointT>::Ptr result_test (new pcl::PointCloud<PointT>());
    // refineRegistration(source_cloud, target_cloud, transformation_test);
    // pcl::transformPointCloud(*source_cloud, *result_test, transformation_test);
    // //pcl::transformPointCloud(source_cpy, *result_test, transformation_test);

    // visualizePointClouds(result_test, target_cloud, 1);

    // std::cout << "End" << std::endl;
    return 0;
}
