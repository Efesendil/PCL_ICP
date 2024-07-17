#include "pipeline.h"

void visualizePointClouds(typename pcl::PointCloud<PointT>::Ptr source_cloud,
                           typename pcl::PointCloud<PointT>::Ptr target_cloud,
                           int point_size) {
    // Create PCL visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Add source keypoints with custom red color
    pcl::visualization::PointCloudColorHandlerCustom<PointT> source_color(source_cloud, 255, 0, 0);
    viewer->addPointCloud<PointT>(source_cloud, source_color, "source cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "source cloud");

    // Add target cloud with custom green color
    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_color(target_cloud, 0, 255, 0);
    viewer->addPointCloud<PointT>(target_cloud, target_color, "target cloud");

    // Display the visualizer until 'q' key is pressed
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void visualizeCorrespondences(PointCloudT::Ptr source_cloud,
                              PointCloudT::Ptr target_cloud,
                              pcl::CorrespondencesPtr correspondences) {
    // Create PCL visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Add source and target point clouds
    pcl::visualization::PointCloudColorHandlerCustom<PointT> source_color(source_cloud, 255, 0, 0);
    viewer->addPointCloud<PointT>(source_cloud, source_color, "source cloud");

    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_color(target_cloud, 0, 255, 0);
    viewer->addPointCloud<PointT>(target_cloud, target_color, "target cloud");

    // Add correspondences as lines between matched points
    for (size_t i = 0; i < correspondences->size(); ++i) {
        PointT pt_src = source_cloud->at(correspondences->at(i).index_query);
        PointT pt_tgt = target_cloud->at(correspondences->at(i).index_match);

        std::stringstream ss_line;
        ss_line << "correspondence_line_" << i;
        viewer->addLine<PointT, PointT>(pt_src, pt_tgt, 255, 255, 255, ss_line.str());
    }

    // Display the visualizer until 'q' key is pressed
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void preprocessPointCloud(PointCloudT::Ptr cloud) {
    auto start = std::chrono::high_resolution_clock::now();

    // Downsampling
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.1f, 0.1f, 0.1f);
    vg.filter(*cloud);

    //Noise filtering
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(5);
    sor.setStddevMulThresh(2.0);
    sor.filter(*cloud);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Preprocessing took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("PointCloud after preprocessing: %zu points\n", cloud->size());
}

void estimateKeypoints(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::ISSKeypoint3D<PointT, pcl::PointXYZ> iss;
    iss.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    iss.setSearchMethod(tree);
    iss.setSalientRadius(3 * 0.1f); // 6 times the leaf size used in voxel grid
    iss.setNonMaxRadius(2 * 0.1f); // 4 times the leaf size used in voxel grid
    iss.setThreshold21(0.975);
    iss.setThreshold32(0.975);
    iss.setMinNeighbors(5);
    iss.setNumberOfThreads(4);
    iss.compute(*keypoints);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Keypoint estimation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Detected %zu keypoints\n", keypoints->size());
}

// void estimateKeypoints(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints) {
//     auto start = std::chrono::high_resolution_clock::now();

//     pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI> harris;
//     pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_temp(new pcl::PointCloud<pcl::PointXYZI>);

//     harris.setInputCloud(cloud);
//     harris.setNonMaxSupression(true);
//     harris.setRadius(0.15); // Adjust radius for your dataset
//     harris.setThreshold(1e-6); // Adjust threshold for your dataset
//     harris.compute(*keypoints_temp);

//     // Convert keypoints from pcl::PointXYZI to pcl::PointXYZ
//     pcl::copyPointCloud(*keypoints_temp, *keypoints);

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     pcl::console::print_info("Keypoint estimation took %.4f seconds.\n", elapsed.count());
//     pcl::console::print_info("Detected %zu keypoints\n", keypoints->size());
// }

void computeDescriptors(PointCloudT::Ptr cloud,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints,
                        pcl::PointCloud<pcl::Normal>::Ptr normals,
                        DescriptorCloudT::Ptr descriptors) {
    auto start = std::chrono::high_resolution_clock::now();

    // Compute normals for keypoints
    pcl::PointCloud<pcl::Normal>::Ptr key_point_normals;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.1);  // Adjust based on point density
    ne.compute(*normals);

    // cloud_normals->points.size () should have the same size as the input cloud->points.size ()
    std::cout << "cloud_normals->points.size (): " << cloud->points.size () << std::endl;
    // cloud_normals->points.size () should have the same size as the input cloud->points.size ()
    std::cout << "cloud_normals->points.size (): " << normals->points.size () << std::endl;

    // Compute FPFH descriptors
    pcl::FPFHEstimation<PointT, pcl::Normal, DescriptorT> fpfh;
    fpfh.setSearchSurface(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setRadiusSearch(0.2);
    fpfh.setInputCloud(keypoints);  // Set keypoints as input cloud
    fpfh.compute(*descriptors);

    // Check if descriptors are computed
    if (descriptors->empty()) {
        pcl::console::print_error("Descriptor computation failed or resulted in an empty cloud!\n");
        return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Descriptor computation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Computed %zu descriptors\n", descriptors->size());
}

void estimateCorrespondences(DescriptorCloudT::Ptr source_descriptors, DescriptorCloudT::Ptr target_descriptors, pcl::CorrespondencesPtr correspondences) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::CorrespondenceEstimation<DescriptorT, DescriptorT> ce;
    ce.setInputSource(source_descriptors);
    ce.setInputTarget(target_descriptors);
    ce.determineReciprocalCorrespondences(*correspondences);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Correspondence estimation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Estimated %zu correspondences\n", correspondences->size());
}

void rejectCorrespondencesRANSAC(PointCloudT::Ptr source_keypoints, PointCloudT::Ptr target_keypoints, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ransac;
    ransac.setInputSource(source_keypoints);
    ransac.setInputTarget(target_keypoints);
    ransac.setInlierThreshold(2.5);
    ransac.setMaximumIterations(1000);
    ransac.setRefineModel(false);
    ransac.setInputCorrespondences(correspondences);
    ransac.getCorrespondences(*inliers);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("RANSAC correspondence rejection took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Inliers after RANSAC: %zu\n", inliers->size());
}

// float computeDescriptorDistance(const DescriptorT &desc1, const DescriptorT &desc2) {
//     float squared_distance = 0.0f;
//     for (int i = 0; i < DescriptorT::descriptorSize(); ++i) {
//         float diff = desc1.histogram[i] - desc2.histogram[i];
//         squared_distance += diff * diff;
//     }
//     return std::sqrt(squared_distance);
// }

// void rejectCorrespondencesDistance(DescriptorCloudT::Ptr source_descriptors, DescriptorCloudT::Ptr target_descriptors, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers) {
//     auto start = std::chrono::high_resolution_clock::now();

//     float max_distance = 10.0; // Adjust this threshold based on your application

//     for (auto &corr : *correspondences) {
//         // Calculate descriptor distance
//         float descriptor_distance = computeDescriptorDistance(source_descriptors->at(corr.index_query), target_descriptors->at(corr.index_match));

//         // Reject correspondence if descriptor distance is above threshold
//         if (descriptor_distance < max_distance) {
//             inliers->push_back(corr);
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     pcl::console::print_info("Distance-based correspondence rejection took %.4f seconds.\n", elapsed.count());
//     pcl::console::print_info("Inliers after distance-based rejection: %zu\n", inliers->size());
// }

void rejectCorrespondencesSurfaceNormal(PointCloudT::Ptr source_keypoints, PointCloudT::Ptr target_keypoints, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::CorrespondenceRejectorSurfaceNormal rej_normals;
    rej_normals.setInputSource<PointT>(source_keypoints);
    rej_normals.setInputTarget<PointT>(target_keypoints);
    rej_normals.setInputCorrespondences(correspondences);
    rej_normals.setThreshold(0.5); // Adjust this threshold as per your data characteristics
    rej_normals.getCorrespondences(*inliers);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Surface normal-based correspondence rejection took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Inliers after surface normal-based rejection: %zu\n", inliers->size());
}

void estimateTransformation(PointCloudT::Ptr source_keypoints, PointCloudT::Ptr target_keypoints, pcl::CorrespondencesPtr inliers, Eigen::Matrix4f &transformation) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::TransformationEstimationSVD<PointT, PointT> trans_est;
    trans_est.estimateRigidTransformation(*source_keypoints, *target_keypoints, *inliers, transformation);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Transformation estimation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Initial transformation:\n");
    std::cout << transformation << std::endl;
}

void refineRegistration(PointCloudT::Ptr source_cloud, PointCloudT::Ptr target_cloud, Eigen::Matrix4f &transformation) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    icp.setMaximumIterations(50);
    //icp.setTransformationEpsilon(1e-6);
    icp.setMaxCorrespondenceDistance(10.0); // Adjust this value according to your dataset
    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    icp.align(*aligned);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("ICP refinement took %.4f seconds.\n", elapsed.count());
    if (icp.hasConverged()) {
        pcl::console::print_info("ICP converged.\n");
        pcl::console::print_info("Final transformation:\n");
        std::cout << icp.getFinalTransformation() << std::endl;
        std::cout << "Fitness score : " << icp.getFitnessScore() << std::endl;
        transformation = icp.getFinalTransformation();
    } else {
        pcl::console::print_warn("ICP did not converge.\n");
        transformation = Eigen::Matrix4f::Identity();
    }
}

/*
void preprocessPointCloud(PointCloudT::Ptr cloud) {
    auto start = std::chrono::high_resolution_clock::now();

    // Noise filtering
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(5);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud);

    // Downsampling
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.1f, 0.1f, 0.1f);
    vg.filter(*cloud);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Preprocessing took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("PointCloud after preprocessing: %zu points\n", cloud->size());
}

void estimateKeypoints(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::ISSKeypoint3D<PointT, pcl::PointXYZ> iss;
    iss.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    iss.setSearchMethod(tree);
    iss.setSalientRadius(6 * 0.1f); // 6 times the leaf size used in voxel grid
    iss.setNonMaxRadius(4 * 0.1f); // 4 times the leaf size used in voxel grid
    iss.setThreshold21(1);
    iss.setThreshold32(1);
    iss.setMinNeighbors(5);
    iss.setNumberOfThreads(4);
    iss.compute(*keypoints);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Keypoint estimation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Detected %zu keypoints\n", keypoints->size());
}

void computeDescriptors(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints, pcl::PointCloud<pcl::Normal>::Ptr normals, DescriptorCloudT::Ptr descriptors) {
    auto start = std::chrono::high_resolution_clock::now();

    // Compute normals
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.3);
    ne.compute(*normals);

    // Compute descriptors
    pcl::SHOTEstimation<PointT, pcl::Normal, DescriptorT> shot;
    shot.setInputCloud(keypoints);  // Set keypoints as input cloud
    shot.setInputNormals(normals);
    shot.setSearchSurface(cloud);
    shot.setRadiusSearch(0.3);
    shot.compute(*descriptors);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Descriptor computation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Computed %zu descriptors\n", descriptors->size());
}

void estimateCorrespondences(DescriptorCloudT::Ptr source_descriptors, DescriptorCloudT::Ptr target_descriptors, pcl::CorrespondencesPtr correspondences) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::CorrespondenceEstimation<DescriptorT, DescriptorT> ce;
    ce.setInputSource(source_descriptors);
    ce.setInputTarget(target_descriptors);
    ce.determineReciprocalCorrespondences(*correspondences);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Correspondence estimation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Estimated %zu correspondences\n", correspondences->size());
}

void rejectCorrespondencesRANSAC(PointCloudT::Ptr source_keypoints, PointCloudT::Ptr target_keypoints, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ransac;
    ransac.setInputSource(source_keypoints);
    ransac.setInputTarget(target_keypoints);
    ransac.setInlierThreshold(0.05);
    ransac.setMaximumIterations(1000);
    ransac.setInputCorrespondences(correspondences);
    ransac.getCorrespondences(*inliers);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("RANSAC correspondence rejection took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Inliers after RANSAC: %zu\n", inliers->size());
}

float computeDescriptorDistance(const pcl::SHOT352 &desc1, const pcl::SHOT352 &desc2) {
    float squared_distance = 0.0f;
    for (int i = 0; i < 352; ++i) {
        float diff = desc1.descriptor[i] - desc2.descriptor[i];
        squared_distance += diff * diff;
    }
    return std::sqrt(squared_distance);
}

void rejectCorrespondencesDistance(DescriptorCloudT::Ptr source_descriptors, DescriptorCloudT::Ptr target_descriptors, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers) {
    auto start = std::chrono::high_resolution_clock::now();

    float max_distance = 0.2; // Adjust this threshold based on your application

    for (auto &corr : *correspondences) {
        // Calculate descriptor distance
        float descriptor_distance = computeDescriptorDistance(source_descriptors->at(corr.index_query), target_descriptors->at(corr.index_match));

        // Reject correspondence if descriptor distance is above threshold
        if (descriptor_distance < max_distance) {
            inliers->push_back(corr);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Distance-based correspondence rejection took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Inliers after distance-based rejection: %zu\n", inliers->size());
}

void rejectCorrespondencesSurfaceNormal(PointCloudT::Ptr source_keypoints, PointCloudT::Ptr target_keypoints, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr inliers) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::CorrespondenceRejectorSurfaceNormal rej_normals;
    rej_normals.setInputSource<PointT>(source_keypoints);
    rej_normals.setInputTarget<PointT>(target_keypoints);
    rej_normals.setInputCorrespondences(correspondences);
    rej_normals.setThreshold(0.75); // Adjust this threshold as per your data characteristics
    rej_normals.getCorrespondences(*inliers);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Surface normal-based correspondence rejection took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Inliers after surface normal-based rejection: %zu\n", inliers->size());
}

void estimateTransformation(PointCloudT::Ptr source_cloud, PointCloudT::Ptr target_cloud, pcl::CorrespondencesPtr inliers, Eigen::Matrix4f &transformation) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::registration::TransformationEstimationSVD<PointT, PointT> trans_est;
    trans_est.estimateRigidTransformation(*source_cloud, *target_cloud, *inliers, transformation);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("Transformation estimation took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Initial transformation:\n");
    std::cout << transformation << std::endl;
}

void refineRegistration(PointCloudT::Ptr source_cloud, PointCloudT::Ptr target_cloud, Eigen::Matrix4f &transformation) {
    auto start = std::chrono::high_resolution_clock::now();

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(0.1); // Adjust this value according to your dataset
    icp.align(*source_cloud, transformation);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    pcl::console::print_info("ICP refinement took %.4f seconds.\n", elapsed.count());
    pcl::console::print_info("Final transformation:\n");
    std::cout << icp.getFinalTransformation() << std::endl;

    auto transformation_matrix = icp.getFinalTransformation();
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>& transformed_cloud = *transformed_cloud_ptr;
    pcl::transformPointCloud(*source_cloud, transformed_cloud, transformation_matrix);
}
*/

