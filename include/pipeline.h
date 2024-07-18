#ifndef PIPELINE_H
#define PIPELINE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/shot.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/multiscale_feature_persistence.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h> // Include Generalized ICP
#include <pcl/registration/transformation_estimation_svd.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/print.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/io/ply_io.h>
#include <cmath> // for std::sqrt
#include <chrono>
#include <thread>
#include <vector>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using DescriptorT = pcl::FPFHSignature33;  // Change to FPFH
using DescriptorCloudT = pcl::PointCloud<DescriptorT>;  // Change to FPFH

// using PointT = pcl::PointXYZ;
// using PointCloudT = pcl::PointCloud<PointT>;
// using DescriptorT = pcl::SHOT352;
// using DescriptorCloudT = pcl::PointCloud<DescriptorT>;

void visualizePointClouds(typename pcl::PointCloud<PointT>::Ptr source_cloud,
                           typename pcl::PointCloud<PointT>::Ptr target_cloud,
                           int point_size);
void visualizeCorrespondences(PointCloudT::Ptr source_cloud,
                              PointCloudT::Ptr target_cloud,
                              pcl::CorrespondencesPtr correspondences);
void preprocessPointCloud(PointCloudT::Ptr cloud);
void estimateKeypoints(PointCloudT::Ptr cloud, pcl::PointCloud<PointT>::Ptr keypoints);
void computeDescriptors(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints, pcl::PointCloud<pcl::Normal>::Ptr normals, DescriptorCloudT::Ptr descriptors);
void computeNormals(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals);
void estimateCorrespondences(DescriptorCloudT::Ptr source_descriptors,
                             DescriptorCloudT::Ptr target_descriptors, pcl::CorrespondencesPtr correspondences);
void multi_scale_keypoint_and_descriptor(PointCloudT::Ptr cloud, 
                                        pcl::PointCloud<pcl::Normal>::Ptr normals, 
                                        DescriptorCloudT::Ptr descriptors, 
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints);
void rejectCorrespondencesRANSAC(pcl::PointCloud<PointT>::Ptr source_keypoints,
                                 pcl::PointCloud<PointT>::Ptr target_keypoints, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers);
void rejectCorrespondencesDistance(DescriptorCloudT::Ptr source_descriptors,
                                   DescriptorCloudT::Ptr target_descriptors, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers);
void rejectCorrespondencesSurfaceNormal(pcl::PointCloud<PointT>::Ptr source_keypoints,
                                        pcl::PointCloud<PointT>::Ptr target_keypoints, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers);
void estimateTransformation(PointCloudT::Ptr source_keypoints, PointCloudT::Ptr target_keypoints,pcl::CorrespondencesPtr correspondences, Eigen::Matrix4f &transformation);
void refineRegistration(PointCloudT::Ptr source_cloud, PointCloudT::Ptr target_cloud, Eigen::Matrix4f &transformation);

#endif // PIPELINE_H