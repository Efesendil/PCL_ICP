#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <chrono>
#include <pcl/memory.h>  // for pcl::make_shared
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>

// Define the standalone MyPointRepresentation class
class MyPointRepresentation : public pcl::PointRepresentation<pcl::PointNormal> {
    using pcl::PointRepresentation<pcl::PointNormal>::nr_dimensions_;
public:
    MyPointRepresentation() {
        // Define the number of dimensions
        nr_dimensions_ = 4;
    }

    // Override the copyToFloatArray method to define our feature vector
    virtual void copyToFloatArray(const pcl::PointNormal &p, float *out) const {
        // < x, y, z, curvature >
        out[0] = p.x;
        out[1] = p.y;
        out[2] = p.z;
        out[3] = p.curvature;
    }
};

class IO_ICP {
public:
    pcl::visualization::PCLVisualizer *p;
    // its left and right viewports
    int vp_1, vp_2;
    const Eigen::Vector4f leaf_size = {0.1f, 0.1f, 0.1f, 0.1f};
    const int KdNeighbours = 30;
    // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
    float alpha[4] = {5.0, 5.0, 5.0, 1.0};
    bool downsample = true;

    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    typedef pcl::PointNormal PointNormalT;
    typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

    struct PCD {
        PointCloud::Ptr cloud;
        std::string f_name;
        PCD() : cloud(new PointCloud) {}
    };

    struct PCDComparator {
        bool operator()(const PCD &p1, const PCD &p2) {
            return (p1.f_name < p2.f_name);
        }
    };

    // Comparator for sorting PCD objects by filename
    static bool comparePCD(const PCD& pcd1, const PCD& pcd2) {
        return pcd1.f_name < pcd2.f_name;
    }

    void sortData(std::vector<PCD, Eigen::aligned_allocator<PCD>>& data) {
        std::sort(data.begin(), data.end(), comparePCD);
    }

    // Type alias for MyPointRepresentation
    using PointRepresentationT = MyPointRepresentation;

    void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source) {
        p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
        p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
        p->removePointCloud("vp1_target");
        p->removePointCloud("vp1_source");
        pcl::visualization::PointCloudColorHandlerCustom<PointT> tgt_h(cloud_target, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<PointT> src_h(cloud_source, 255, 0, 0);
        p->addPointCloud(cloud_target, tgt_h, "vp1_target", vp_1);
        p->addPointCloud(cloud_source, src_h, "vp1_source", vp_1);
        PCL_INFO("Press q to begin the registration.\n");
        p->spin();
    }

    void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source) {
        p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration example");
        p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
        p->removePointCloud("source");
        p->removePointCloud("target");
        pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler(cloud_target, "curvature");
        if (!tgt_color_handler.isCapable())
            PCL_WARN("Cannot create curvature color handler!\n");
        pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> src_color_handler(cloud_source, "curvature");
        if (!src_color_handler.isCapable())
            PCL_WARN("Cannot create curvature color handler!\n");
        p->addPointCloud(cloud_target, tgt_color_handler, "target", vp_2);
        p->addPointCloud(cloud_source, src_color_handler, "source", vp_2);
        p->spinOnce();
    }

    void loadData(const std::string &directory, std::vector<PCD, Eigen::aligned_allocator<PCD>> &models) {
        std::string extension(".pcd");

        // Suppose the first argument is the actual test model
        for (const auto &entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == extension) {
                std::string fname = entry.path().string();
                // Load the cloud and save it into the global list of models
                PCD m;
                m.f_name = fname;
                m.cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

                if (pcl::io::loadPCDFile(fname, *m.cloud) == -1) {
                    PCL_ERROR("Couldn't read file %s \n", fname.c_str());
                    continue;
                }
                // Remove NAN points from the cloud
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*m.cloud, *m.cloud, indices);
                models.push_back(m);
            }
        }
    }

    std::vector<std::pair<PointCloudWithNormals::Ptr, PointCloudWithNormals::Ptr>> preProcess(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt) {
        PointRepresentationT point_representation;

        // Downsample for consistency and speed
        // \note enable this for large datasets
        PointCloud::Ptr src(new PointCloud);
        PointCloud::Ptr tgt(new PointCloud);

        pcl::VoxelGrid<PointT> grid;
        if (downsample) {
            grid.setLeafSize(leaf_size.x(), leaf_size.y(), leaf_size.z());

            grid.setInputCloud(cloud_src);
            grid.filter(*src);

            grid.setInputCloud(cloud_tgt);
            grid.filter(*tgt);
        } else {
            src = cloud_src;
            tgt = cloud_tgt;
        }
        // Compute surface normals and curvature
        PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
        PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

        pcl::NormalEstimation<PointT, PointNormalT> norm_est;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

        norm_est.setSearchMethod(tree);
        norm_est.setKSearch(KdNeighbours);

        norm_est.setInputCloud(src);
        norm_est.compute(*points_with_normals_src);
        pcl::copyPointCloud(*src, *points_with_normals_src);

        norm_est.setInputCloud(tgt);
        norm_est.compute(*points_with_normals_tgt);
        pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

        // Instantiate our custom point representation (defined above) ...
        point_representation.setRescaleValues(alpha);

        std::vector<std::pair<PointCloudWithNormals::Ptr, PointCloudWithNormals::Ptr>> returnValues;
        returnValues.emplace_back(points_with_normals_src, points_with_normals_tgt);

        return returnValues;
    }

    void icp_align(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt,
                   const PointCloudWithNormals::Ptr points_with_normals_src, const PointCloudWithNormals::Ptr points_with_normals_tgt,
                   PointCloud::Ptr output, Eigen::Matrix4f &final_transform) {

        PointRepresentationT point_representation;

        // Align
        pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
        *output = *cloud_tgt;
        reg.setTransformationEpsilon(1e-6);
        // Set the maximum distance between two correspondences (src<->tgt) to 10cm
        // Note: adjust this based on the size of your datasets
        reg.setMaxCorrespondenceDistance(0.01);
        reg.setMaximumIterations(5);
        // Set the point representation
        reg.setPointRepresentation(pcl::make_shared<const PointRepresentationT>(point_representation));
        reg.setInputSource(points_with_normals_src);
        reg.setInputTarget(points_with_normals_tgt);

        // Run the same optimization in a loop and visualize the results
        Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
        PointCloudWithNormals::Ptr reg_result(new PointCloudWithNormals);
        *reg_result = *points_with_normals_src; // Initialize reg_result with points_with_normals_src
        reg.setInputSource(points_with_normals_src);
        reg.align(*reg_result);
        // accumulate transformation between each Iteration
        Ti = reg.getFinalTransformation() * Ti;
        // for (int i = 0; i < 1; ++i) {
        //     PCL_INFO("Iteration Nr. %d.\n", i);
        //     // save cloud for visualization purpose
        //     *points_with_normals_src = *reg_result;
        //     // Estimate
        //     reg.setInputSource(points_with_normals_src);
        //     reg.align(*reg_result);
        //     // accumulate transformation between each Iteration
        //     Ti = reg.getFinalTransformation() * Ti;
        //     // if the difference between this transformation and the previous one
        //     // is smaller than the threshold, refine the process by reducing
        //     // the maximal correspondence distance
        //     if (std::abs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
        //         reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);
        //     prev = reg.getLastIncrementalTransformation();
        //     // visualize current state
        //     //showCloudsRight(points_with_normals_tgt, points_with_normals_src);
        // }
        // Get the transformation from target to source
        targetToSource = Ti.inverse();

        // Transform target back in source frame
        pcl::transformPointCloud(*cloud_src, *output, targetToSource);
        // p->removePointCloud("source");
        // p->removePointCloud("target");
        // pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tgt_h(output, 0, 255, 0);
        // pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_src_h(cloud_src, 255, 0, 0);
        // p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
        // p->addPointCloud(cloud_src, cloud_src_h, "source", vp_2);
        // PCL_INFO("Press q to continue the registration.\n");
        // p->spin();
        // p->removePointCloud("source");
        // p->removePointCloud("target");
        // add the source to the transformed target
        *output += *cloud_src;
        final_transform = targetToSource;
    }
};
