// Whole native geometry wrapper. OpenCV supplies IPPE/SQPnP starts.
#include "glyph_trf.cpp"
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
extern "C" int glyph_pose(int n,const double*points,const double*prob,const double*K,const double*templ,const double*quat,double*out){
 if(n<8||n>64)return 0;
 std::vector<int> ids;int glyphs=0;
 for(int i=0;i<n;i++)if(prob[i]>=.1&&std::isfinite(points[2*i])&&std::isfinite(points[2*i+1])){ids.push_back(i);glyphs+=i>=4;}
 if(ids.size()<8||glyphs<4)return 0;
 Eigen::Quaterniond cq(quat[0],quat[1],quat[2],quat[3]);cq.normalize();
 Eigen::Matrix3d link;link<<0,-1,0,0,0,-1,1,0,0;
 Eigen::Matrix<double,3,3,Eigen::RowMajor> wc=link*cq.toRotationMatrix().transpose();
 Eigen::Matrix3d mount=Eigen::AngleAxisd(-15*M_PI/180.,Eigen::Vector3d::UnitX()).toRotationMatrix();
 std::vector<cv::Point3d> all3,valid3;std::vector<cv::Point2d> first2,valid2;
 std::vector<double> mounted,obs,weights;
 for(int i=0;i<n;i++){
  all3.emplace_back(templ[3*i],templ[3*i+1],templ[3*i+2]);
  if(i<4)first2.emplace_back(points[2*i],points[2*i+1]);
 }
 for(int i:ids){
  valid3.push_back(all3[i]);valid2.emplace_back(points[2*i],points[2*i+1]);
  Eigen::Vector3d v=mount*Eigen::Map<const Eigen::Vector3d>(templ+3*i);
  for(int j=0;j<3;j++)mounted.push_back(v[j]);
  obs.push_back(points[2*i]);obs.push_back(points[2*i+1]);
  weights.push_back(std::sqrt(std::max(.1,std::min(1.,prob[i]))));
 }
 cv::Mat km(3,3,CV_64F,const_cast<double*>(K)),dist=cv::Mat::zeros(5,1,CV_64F);
 std::vector<cv::Mat> rvec,tvec;
 try{std::vector<cv::Point3d> first3(all3.begin(),all3.begin()+4);cv::solvePnPGeneric(first3,first2,km,dist,rvec,tvec,false,cv::SOLVEPNP_IPPE);}catch(const cv::Exception&){}
 try{cv::Mat rv,tv;if(cv::solvePnP(valid3,valid2,km,dist,rv,tv,false,cv::SOLVEPNP_SQPNP)){rvec.push_back(rv);tvec.push_back(tv);}}catch(const cv::Exception&){}
 double kk[4]={K[0],K[4],K[2],K[5]},best=std::numeric_limits<double>::infinity();bool found=false;
 for(size_t k=0;k<rvec.size();k++){
  const double*t=tvec[k].ptr<double>();if(!std::isfinite(t[0])||!std::isfinite(t[1])||!std::isfinite(t[2])||t[2]<=.05)continue;
  cv::Mat rc;cv::Rodrigues(rvec[k],rc);
  Eigen::Matrix3d rw=wc.transpose()*Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(rc.ptr<double>());
  double initial[4]={t[0],t[1],t[2],std::atan2(rw(1,0),rw(0,0))},result[7];
  if(glyph_trf(ids.size(),mounted.data(),wc.data(),obs.data(),weights.data(),kk,initial,result))continue;
  if(!std::isfinite(result[4])||result[4]>=best||result[2]<=.05)continue;
  Eigen::Vector3d tv(result[0],result[1],result[2]);
  Eigen::Matrix<double,3,3,Eigen::RowMajor> rot=wc*Eigen::AngleAxisd(result[3],Eigen::Vector3d::UnitZ()).toRotationMatrix()*mount;
  std::vector<double> projected(2*n);bool finite=true;
  for(int i=0;i<n;i++){
   Eigen::Vector3d p=rot*Eigen::Map<const Eigen::Vector3d>(templ+3*i)+tv;
   projected[2*i]=K[0]*p[0]/p[2]+K[2];projected[2*i+1]=K[4]*p[1]/p[2]+K[5];
   finite=finite&&std::isfinite(projected[2*i])&&std::isfinite(projected[2*i+1]);
  }
  if(!finite)continue;
  cv::Mat cvrot(3,3,CV_64F,rot.data()),rv;cv::Rodrigues(cvrot,rv);
  for(int j=0;j<3;j++){out[j]=result[j];out[3+j]=rv.at<double>(j);}
  std::copy(projected.begin(),projected.end(),out+6);
  double residual=0;
  for(int i:ids)for(int a=0;a<2;a++){double d=projected[2*i+a]-points[2*i+a];residual+=std::max(.1,std::min(1.,prob[i]))*d*d;}
  out[6+2*n]=std::sqrt(residual/(2*ids.size()));out[7+2*n]=result[4];out[8+2*n]=ids.size();best=result[4];found=true;
 }
 return found?1:0;
}
