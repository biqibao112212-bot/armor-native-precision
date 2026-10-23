#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
extern "C" int glyph_crop_sizes(int count,int n,const double* corners,const double* K,const double* templ,int* sizes){
 if(count<0||n<4||n>64)return -1;
 cv::Mat km(3,3,CV_64F,const_cast<double*>(K)),dist=cv::Mat::zeros(5,1,CV_64F);
 std::vector<cv::Point3d> obj;for(int j=0;j<n;j++)obj.emplace_back(templ[3*j],templ[3*j+1],templ[3*j+2]);
 std::vector<cv::Point3d> first(obj.begin(),obj.begin()+4);
 for(int i=0;i<count;i++){
  const double* cc=corners+8*i;std::vector<cv::Point2d> image;
  double minx=1e300,miny=1e300,maxx=-1e300,maxy=-1e300;
  for(int j=0;j<4;j++){image.emplace_back(cc[2*j],cc[2*j+1]);minx=std::min(minx,cc[2*j]);maxx=std::max(maxx,cc[2*j]);miny=std::min(miny,cc[2*j+1]);maxy=std::max(maxy,cc[2*j+1]);}
  double extent=std::max(maxx-minx,maxy-miny);
  try{
   std::vector<cv::Mat> rv,tv;cv::Mat errors;
   cv::solvePnPGeneric(first,image,km,dist,rv,tv,false,cv::SOLVEPNP_IPPE,cv::noArray(),cv::noArray(),errors);
   int best=-1;double besterr=std::numeric_limits<double>::infinity();
   for(size_t j=0;j<tv.size();j++){
    const double* t=tv[j].ptr<double>();double e=errors.depth()==CV_64F?errors.at<double>(j):errors.at<float>(j);
    if(std::isfinite(t[0])&&std::isfinite(t[1])&&std::isfinite(t[2])&&t[2]>0&&e<besterr){best=int(j);besterr=e;}
   }
   if(best>=0){
    std::vector<cv::Point2d> projected;cv::projectPoints(obj,rv[best],tv[best],km,dist,projected);
    minx=miny=1e300;maxx=maxy=-1e300;bool finite=true;
    for(auto& p:projected){finite=finite&&std::isfinite(p.x)&&std::isfinite(p.y);minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);}
    if(finite)extent=std::max(extent,std::max(maxx-minx,maxy-miny));
   }
  }catch(const cv::Exception&){}
  sizes[i]=-1;for(int s:{128,192,256,384,512})if(s>=extent+48){sizes[i]=s;break;}
 }
 return 0;
}
