#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h> // for pcl::console::parse_argument
#include "pipeline.cpp"

int main(int argc, char **argv) {
    // Check the number of parameters
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " source_cloud.pcd target_cloud.pcd" << std::endl;
        return -1;
    }

    // Load point clouds
    PointCloudT::Ptr source_cloud(new PointCloudT);
    PointCloudT::Ptr target_cloud(new PointCloudT);
    
    pcl::io::loadPCDFile(argv[1], *source_cloud);
    pcl::io::loadPCDFile(argv[2], *target_cloud);

    // Preprocess point clouds
    preprocessPointCloud(source_cloud);
    preprocessPointCloud(target_cloud);

    visualizePointClouds(source_cloud, target_cloud, 1);

    // Keypoint estimation
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    estimateKeypoints(source_cloud, source_keypoints);
    estimateKeypoints(target_cloud, target_keypoints);

    visualizePointClouds(source_keypoints, source_cloud, 8);
    visualizePointClouds(target_keypoints, target_cloud, 8);

    // Descriptor computation
    pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr source_keypoint_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_keypoint_normals(new pcl::PointCloud<pcl::Normal>);
    DescriptorCloudT::Ptr source_descriptors(new DescriptorCloudT);
    DescriptorCloudT::Ptr target_descriptors(new DescriptorCloudT);
    // computeDescriptors(source_cloud, source_keypoints, source_normals, source_descriptors);
    // computeDescriptors(target_cloud, target_keypoints, target_normals, target_descriptors);
    computeDescriptors(source_cloud, source_keypoints, source_normals, source_descriptors);
    computeDescriptors(target_cloud, target_keypoints, target_normals, target_descriptors);

    // Correspondence estimation
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    estimateCorrespondences(source_descriptors, target_descriptors, correspondences);

    visualizeCorrespondences(source_cloud, target_cloud, correspondences);

    // Correspondence rejection using multiple methods
    pcl::CorrespondencesPtr inliers(new pcl::Correspondences);
    rejectCorrespondencesRANSAC(source_keypoints, target_keypoints, correspondences, inliers);
    //rejectCorrespondencesDistance(source_descriptors, target_descriptors, correspondences, inliers);
    //rejectCorrespondencesSurfaceNormal(source_keypoints, target_keypoints, correspondences, inliers);

    // Transformation estimation
    Eigen::Matrix4f transformation_rough;
    estimateTransformation(source_keypoints, target_keypoints, inliers, transformation_rough);

    // Apply the transformation to the source cloud
    pcl::PointCloud<PointT>::Ptr transformed_source_cloud(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*source_cloud, *transformed_source_cloud, transformation_rough);

    visualizePointClouds(transformed_source_cloud, target_cloud, 1);

    //pcl::io::savePCDFileBinary("/home/efesendil/Outputs/output.pcd", *transformed_source_cloud);

    // Registration refinement
    Eigen::Matrix4f transformation_fine;
    pcl::PointCloud<PointT>::Ptr result (new pcl::PointCloud<PointT>());
    refineRegistration(transformed_source_cloud, target_cloud, transformation_fine);
    pcl::transformPointCloud(*transformed_source_cloud, *result, transformation_fine);

    // //pcl::io::savePCDFileBinary("/home/efesendil/Outputs/output.pcd", *source_cloud);

    visualizePointClouds(result, target_cloud, 1);

    return 0;
}
