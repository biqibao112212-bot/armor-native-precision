#include "../scripts/capture_target_identity.hpp"
#include <cassert>
#include <iostream>
int main(){
 auto p=configured_target_position(R"({"target_position_world_m": [4.43200456, 5.8, 1.16366952]})");
 std::vector<std::array<double,3>> points={{5.40055656,5.79997349,.23629075},p};
 assert(unique_target_at_position(points,p)==1);
 std::swap(points[0],points[1]);assert(unique_target_at_position(points,p)==0);
 assert(unique_target_at_position(points,{0,0,0})==-1);
 points.push_back(p);bool rejected=false;try{unique_target_at_position(points,p);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 for(auto bad:{R"({})",R"({"target_position_world_m":[1,2]})",R"({"target_position_world_m":[1,2,3,4]})"}){rejected=false;try{configured_target_position(bad);}catch(const std::runtime_error&){rejected=true;}assert(rejected);}
 std::cout<<"configured target identity: passed\n";
}
