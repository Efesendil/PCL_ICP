#include "include/pcd_read.cpp"
#include "include/pcd_filter.cpp"
#include <thread>
#include <pcl/console/parse.h>


int main (int argc, char ** argv)

{
    IO io;
    Filter filter;

    if (pcl::console::find_argument (argc, argv, "-h") >= 0){
    io.printUsage (argv[0]);
    }
    else if(pcl::console::find_argument (argc, argv, "-s") >= 0){
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        auto cloud1 = io.read_pcd(argv[2]);
        auto cloud2 = io.read_pcd(argv[3]);
        auto cloud1_filtered = filter.VoxelGrid(cloud1);
        auto cloud2_filtered = filter.VoxelGrid(cloud2);
        auto cloud1_outlier_rem = filter.RadiusOutlier(cloud1_filtered);
        auto cloud2_outlier_rem = filter.RadiusOutlier(cloud2_filtered);

        Registration reg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_normals = reg.ComputeNormals(cloud1_outlier_rem);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_normals = reg.ComputeNormals(cloud2_outlier_rem);

        auto final_ptr = cloud2_normals;

        Eigen::Matrix4f final = reg.icp_pose(cloud1_normals, cloud2_normals);
        pcl::transformPointCloud (*cloud1_normals, *final_ptr, final);
        *final_ptr += *cloud1_normals;

        io.simpleVis(viewer, cloud2_normals, final_ptr);
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
        }
    }

    else if(pcl::console::find_argument (argc, argv, "-v") >= 0){
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 = io.read_pcd(argv[2]);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 = io.read_pcd(argv[3]);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_rgb(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_rgb(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*cloud1, *cloud1_rgb);
        pcl::copyPointCloud(*cloud2, *cloud2_rgb);

        auto cloud1_rgb_filtered = filter.VoxelGrid(cloud1_rgb);
        auto cloud2_rgb_filtered = filter.VoxelGrid(cloud2_rgb);
        auto cloud1_rgb_outlier_rem = filter.RadiusOutlier(cloud1_rgb_filtered);
        auto cloud2_rgb_outlier_rem = filter.RadiusOutlier(cloud2_rgb_filtered);

        Registration reg;
        auto cloud1_normals = reg.ComputeNormals(cloud1_rgb_outlier_rem);
        auto cloud2_normals = reg.ComputeNormals(cloud2_rgb_outlier_rem);

        auto output = cloud2_normals;
        
        Eigen::Matrix4f Ti = reg.icp_pose(cloud1_normals, cloud2_normals);
        pcl::transformPointCloud(*cloud1_normals, *output, Ti);
        *output += *cloud1_normals;

        io.viewportsVis(viewer, cloud1, cloud2, output, cloud1_normals);
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
        }
    }
    else if(pcl::console::find_argument (argc, argv, "-w") >= 0){
        auto cloud = io.read_pcd(argv[2]);
        io.write_pcd(argv[3], *cloud);
    }
    else {
        std::cout << "\nOptions are: \n\n";
        std::cout << "./pcd_io_test <arg> <reading_cloud>\n";
        std::cout << "./pcd_io_test <arg -v or -s> <reading_cloud1> <reading_cloud2>\n";
        std::cout << "./pcd_io_test <arg> <reading_cloud> <writing_cloud>\n";
    }
      
}