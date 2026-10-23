#pragma once
#include <array>
#include <vector>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <cmath>
inline std::array<double,3> configured_target_position(const std::string& json){
 auto key=json.find("\"target_position_world_m\"");
 if(key==std::string::npos)throw std::runtime_error("configured target position missing");
 auto begin=json.find('[',key),end=json.find(']',begin);
 if(begin==std::string::npos||end==std::string::npos)throw std::runtime_error("configured target position array missing");
 std::string s=json.substr(begin+1,end-begin-1);for(char& c:s)if(c==',')c=' ';std::istringstream in(s);std::array<double,3> p{};
 for(double& x:p)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid configured position");
 std::string extra;if(in>>extra)throw std::runtime_error("configured position must have three coordinates");return p;
}
inline int unique_target_at_position(const std::vector<std::array<double,3>>& positions,const std::array<double,3>& requested,double tolerance=.02){
 int found=-1;for(unsigned i=0;i<positions.size();i++){double d=0;for(int k=0;k<3;k++)d+=std::pow(positions[i][k]-requested[k],2);
 if(d<=tolerance*tolerance){if(found>=0)throw std::runtime_error("ambiguous configured target identity");found=i;}}
 return found;
}
