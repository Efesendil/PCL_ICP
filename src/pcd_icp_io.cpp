#include "include/pcd_icp.cpp"

int main(int argc, char** argv){

    IO_ICP icp;
    
    std::vector<IO_ICP::PCD, Eigen::aligned_allocator<IO_ICP::PCD>> data;
    icp.loadData(argv[1], data);
    icp.sortData(data);

    if(data.empty()){
        PCL_ERROR ("Syntax is: %s <source.pcd> <target.pcd> [*]\n", argv[0]);
        PCL_ERROR ("[*] - multiple files can be added. The registration results of (i, i+1) will be registered against (i+2), etc\n");
        return (-1);
    }

    // Create a PCLVisualizer object
    pcl::visualization::PCLVisualizer *p;
    // its left and right viewports
    int vp_1, vp_2;

    pcl::PointCloud<pcl::PointXYZ>::Ptr pc (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr result (new pcl::PointCloud<pcl::PointXYZ>);
    // Load data or create point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform, t;

    for (std::size_t i = 1; i < data.size (); ++i){
        source = data[i-1].cloud;
        target = data[i].cloud;
        // Add visualization data
        //icp.showCloudsLeft(source, target);

        	pcl::PointCloud<pcl::PointXYZ>::Ptr temp (new pcl::PointCloud<pcl::PointXYZ>);
        PCL_INFO ("Aligning %s (%zu) with %s (%zu).\n", data[i-1].f_name.c_str (), static_cast<std::size_t>(source->size ()), data[i].f_name.c_str (), static_cast<std::size_t>(target->size ()));
        auto preprocessed_data = icp.preProcess(source, target);
        // Access the elements of the preprocessed data
        pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_src (new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr points_with_normals_tgt (new pcl::PointCloud<pcl::PointNormal>);

        points_with_normals_src = preprocessed_data[0].first;
        points_with_normals_tgt = preprocessed_data[0].second;

        icp.icp_align(source, target, points_with_normals_src, points_with_normals_tgt, temp, pairTransform);
        std::cout << pairTransform << std::endl;
        //update the global transform
        GlobalTransform *= pairTransform;

        pcl::transformPointCloud (*temp, *result, GlobalTransform);      

        //save aligned pair, transformed into the first cloud's frame
        std::stringstream ss;
        ss << i << ".pcd";
        if(i==data.size()-1){
            pcl::io::savePCDFile ("../pcds/" + ss.str(), *result, true);
        }
    }

}