// Packaged 2026-10-24: native observation covariance, file/stdin streaming.
// Image-only observation adapter for the unchanged project 11-state EKF.
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>
#include "autoaim_research/coordinates.hpp"
#include "autoaim_research/pose_solver.hpp"
#include "autoaim_research/research_tracker.hpp"
#include "tools/math_tools.hpp"
std::string aim_diagnostic_context;
Eigen::Matrix3d aim_position_covariance_world = Eigen::Matrix3d::Identity();
using namespace autoaim_research;
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("INPUT_TSV OUTPUT_JSONL");std::ifstream input_file;std::ofstream output_file;if(std::string(argv[1])!="-")input_file.open(argv[1]);if(std::string(argv[2])!="-")output_file.open(argv[2]);std::istream& in=std::string(argv[1])=="-"?std::cin:input_file;std::ostream& out=std::string(argv[2])=="-"?std::cout:output_file;if(!in||!out)throw std::runtime_error("file open");out<<std::setprecision(17);
 std::map<std::string,TongjiResearchTracker> trackers;std::string line;
 cv::Mat K=(cv::Mat_<double>(3,3)<<1303.675283386667,0,720,0,1303.675283386667,540,0,0,1);ArmorGeometry geometry;geometry.small_width_m=.1337707;geometry.height_m=.0538873;TongjiPoseSolver solver(K,cv::Mat::zeros(1,5,CV_64F),geometry);
 while(std::getline(in,line)){
  std::vector<std::string> v;std::istringstream stream(line);std::string x;while(std::getline(stream,x,'\t'))v.push_back(x);if(v.size()<20)throw std::runtime_error("TSV header width");
  std::string session=v[0],mode=v[1],method=v[4],key=session+"/"+method;auto frame=std::stoull(v[2]),time=std::stoull(v[3]);int n=std::stoi(v[5]);if(v.size()!=20+n*11)throw std::runtime_error("TSV observation width");size_t i=6;auto val=[&](){return std::stod(v.at(i++));};ExposurePose ex;
  ex.camera_position_world_m={val(),val(),val()};i=13;
  // Quaternion constructor arguments must be sequenced explicitly (C++ evaluation order).
  auto quat=[&](size_t p){return Eigen::Quaterniond(std::stod(v[p]),std::stod(v[p+1]),std::stod(v[p+2]),std::stod(v[p+3]));};ex.world_from_camera_link=quat(9);
  ex.gimbal_position_world_m={val(),val(),val()};ex.world_from_gimbal=quat(16);i=20;
  const Eigen::Matrix3d optical_cov = (Eigen::Vector3d(0.003*0.003,0.003*0.003,0.020*0.020)).asDiagonal();
  const Eigen::Matrix3d world_optical = worldFromOptical(ex);
  aim_position_covariance_world = world_optical * optical_cov * world_optical.transpose();

  if(!trackers.count(key)){TrackerConfig config;config.max_observation_gap_s=.5;trackers.emplace(key,TongjiResearchTracker(config));}
  std::list<auto_aim::Armor> armors;int invalid=0;
  for(int j=0;j<n;j++){
   std::array<cv::Point2f,4> c;for(auto&pt:c){pt.x=val();pt.y=val();}Eigen::Vector3d xyz{val(),val(),val()};
   std::vector<cv::Point2f> points{c[1],c[2],c[3],c[0]};auto_aim::Armor armor(0,3,1.0F,cv::Rect(),points);armor.priority=auto_aim::ArmorPriority::third;armor.duplicated=false;
   auto pose=solver.solve(armor,ex);if(!pose.valid){invalid++;continue;}armor.xyz_in_world=opticalPointToResearch(ex,xyz);armor.xyz_in_gimbal=ex.world_from_gimbal.normalized().toRotationMatrix().transpose()*armor.xyz_in_world;armor.ypd_in_world=tools::xyz2ypd(armor.xyz_in_world);armors.push_back(armor);
  }
  ::aim_diagnostic_context=session+"\t"+method+"\t"+std::to_string(frame);auto result=trackers.at(key).update(armors,time);out<<"{\"session\":\""<<session<<"\",\"mode\":\""<<mode<<"\",\"method\":\""<<method<<"\",\"frame_seq\":"<<frame<<",\"timestamp_ns\":"<<time<<",\"state\":\""<<result.state<<"\",\"observation_count\":"<<n<<",\"invalid_pose\":"<<invalid<<",\"association_ids\":[";
  for(size_t k=0;k<result.association_ids.size();k++){if(k)out<<',';out<<result.association_ids[k];}out<<"],\"state_vector\":";
  if(result.has_estimate){out<<'[';for(int k=0;k<11;k++){if(k)out<<',';out<<result.state_vector[k];}out<<']';}else out<<"null";out<<"}\n";out.flush();
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
