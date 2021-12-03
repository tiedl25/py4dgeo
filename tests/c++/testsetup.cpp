#include "testsetup.hpp"

#include <fstream>
#include <sstream>
#include <vector>

using namespace py4dgeo;

std::shared_ptr<EigenPointCloud>
benchcloud_from_file(const std::string& filename)
{
  std::vector<float> data;
  std::ifstream stream(filename);
  std::string line;
  std::size_t points{ 0 };

  while (std::getline(stream, line)) {
    std::istringstream s(line);
    float x;
    for (int i = 0; i < 3; ++i) {
      s >> x;
      data.push_back(x);
    }
    ++points;
  }

  // Interpret the given data as an Eigen matrix
  auto cloud = std::make_shared<EigenPointCloud>(points, 3);
  std::copy(data.data(), data.data() + points * 3, cloud->data());
  return cloud;
}

std::shared_ptr<EigenPointCloud>
slice_cloud(EigenPointCloudConstRef cloud, int sampling_factor)
{
  auto sliced =
    std::make_shared<EigenPointCloud>(cloud.rows() / sampling_factor, 3);
  for (IndexType i = 0; i < cloud.rows() / sampling_factor; ++i)
    (*sliced)(i, Eigen::all) = cloud(i * sampling_factor, Eigen::all);
  return sliced;
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
ahk_benchcloud()
{
  auto cloud = benchcloud_from_file(DATAPATH(ahk_2017_small.xyz));
  return std::make_pair(cloud, slice_cloud(*cloud, 1000));
}

std::pair<std::shared_ptr<EigenPointCloud>, std::shared_ptr<EigenPointCloud>>
testcloud()
{
  auto cloud = benchcloud_from_file(DATAPATH(plane_horizontal_t1.xyz));
  return std::make_pair(cloud, cloud);
}
