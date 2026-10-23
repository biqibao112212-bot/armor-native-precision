// Number-3 offline acquisition. Public release SDK only; no private IPC implementation.
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "capture_target_identity.hpp"
using namespace daedalus::sim::sdk::v1;
template<class T> void ok(const T&r){if(!r)throw std::runtime_error(r.status.message);if(r.value->status!=SceneControlStatus::Ok)throw std::runtime_error(r.value->message);}
template<class T> void arr(std::ostream&o,const T*p,int n){o<<'[';for(int i=0;i<n;i++){if(i)o<<',';o<<p[i];}o<<']';}
int main(int argc,char**argv){try{
 if(argc!=10)throw std::runtime_error("IPC SESSION CONFIG N MODE DIRECTION SPEED SPAN SPIN");
 SceneControlOptions opts;opts.session_id=argv[2];opts.timeout=std::chrono::milliseconds(5000);
 ContestClientOptions co;co.ipc_directory=argv[1];co.scene_control=opts;co.frame_timeout=std::chrono::milliseconds(4000);ContestClient sim(co);auto connected=sim.connect();if(!connected)throw std::runtime_error(connected.message);
 SceneControlClient control(opts);ok(control.setScene(SceneMode::ShootingRange));RangeTargetMotion m;m.target=3;ok(control.setRangeTargetMotion(m));m.target=1;ok(control.setRangeTargetMotion(m));m.target=3;
 std::ifstream file(argv[3]);std::string cfg((std::istreambuf_iterator<char>(file)),{});auto applied=control.setCaptureScenario(cfg);ok(applied);uint64_t ack=applied.value->applied_frame_seq;
 TalosMetadataMapping mapping;auto opened=mapping.open(std::string(argv[1])+"/talos_ipc_meta");if(!opened)throw std::runtime_error(opened.message);auto rr=mapping.reader();if(!rr)throw std::runtime_error(rr.status.message);auto camera=rr.value->readCameraInfo();if(!camera)throw std::runtime_error(camera.status.message);
 const auto requested_position=configured_target_position(cfg);
 uint64_t bound_id=0,binding_seq=0;std::array<double,3> bound_position{};
 auto binding_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(8);uint64_t binding_cursor=0;
 while(!bound_id&&std::chrono::steady_clock::now()<binding_deadline){
  auto fr=sim.nextFrame(binding_cursor);if(!fr)continue;auto& f=*fr.value;auto& h=f.image.header;binding_cursor=h.source_sequence;if(binding_cursor<ack)continue;
  auto tr=rr.value->readGroundTruthForFrame(binding_cursor);if(!tr)continue;auto& e=*tr.value;
  if(e.producer_epoch!=h.producer_epoch||e.ground_truth.timestamp_ns!=h.capture_timestamp_ns||e.exposure_state.timestamp_ns!=h.capture_timestamp_ns)throw std::runtime_error("target binding exposure mismatch");
  std::vector<std::array<double,3>> positions;std::vector<uint64_t> ids;
  for(unsigned i=0;i<e.ground_truth.target_count;i++){auto& x=e.ground_truth.targets[i];if(x.armor_label==3&&x.armor_count==4){positions.push_back({x.position[0],x.position[1],x.position[2]});ids.push_back(x.target_id);}}
  int index=unique_target_at_position(positions,requested_position);if(index>=0){bound_id=ids[index];bound_position=positions[index];binding_seq=binding_cursor;}
 }
 if(!bound_id)throw std::runtime_error("configured target not found before starting motion");
 std::string mode=argv[5];m.mode=mode=="linear"?RangeMotionMode::Linear:mode=="spin"?RangeMotionMode::Spin:mode=="combined"?RangeMotionMode::LinearAndSpin:RangeMotionMode::Stationary;m.direction_deg=std::stof(argv[6]);m.linear_speed_mps=std::stof(argv[7]);m.linear_span_m=std::stof(argv[8]);m.spin_deg_s=std::stof(argv[9]);auto moving=control.setRangeTargetMotion(m);ok(moving);ack=std::max(ack,moving.value->applied_frame_seq);
 uint64_t seq=binding_cursor,first=0,last=0,target_id=bound_id;int saved=0,misses=0,count=std::stoi(argv[4]);auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(40);
 while(saved<count&&std::chrono::steady_clock::now()<deadline){auto fr=sim.nextFrame(seq);if(!fr){misses++;continue;}auto&f=*fr.value;auto&h=f.image.header;seq=h.source_sequence;if(seq<ack)continue;
 auto tr=rr.value->readGroundTruthForFrame(seq);if(!tr){misses++;continue;}auto&e=*tr.value;auto&ex=e.exposure_state;
 if(e.producer_epoch!=h.producer_epoch||e.ground_truth.timestamp_ns!=h.capture_timestamp_ns||ex.timestamp_ns!=h.capture_timestamp_ns||f.gimbal.frame_seq!=seq||f.gimbal.timestamp_ns!=h.capture_timestamp_ns)throw std::runtime_error("exact identity mismatch");
 const GroundTruthTarget*t=nullptr;for(unsigned i=0;i<e.ground_truth.target_count;i++){auto& x=e.ground_truth.targets[i];if(x.target_id==target_id&&x.armor_label==3&&x.armor_count==4){t=&x;break;}}if(!t)throw std::runtime_error("bound target disappeared");
 if(!first)first=h.capture_timestamp_ns;if(h.capture_timestamp_ns-first<300000000ULL||h.capture_timestamp_ns-last<100000000ULL)continue;
 if(h.width!=1440||h.height!=1080||f.image.payload.size()!=6220800)throw std::runtime_error("not native RGBA");
 std::ostringstream o;o<<std::setprecision(17)<<"{\"session\":\""<<argv[2]<<"\",\"producer_epoch\":"<<h.producer_epoch<<",\"frame_seq\":"<<seq<<",\"timestamp_ns\":"<<h.capture_timestamp_ns<<",\"ack_seq\":"<<ack<<",\"target_id\":"<<target_id<<",\"camera_position_world\":";arr(o,ex.camera_position_world,3);o<<",\"camera_quaternion_wxyz\":";arr(o,ex.camera_quaternion_world_wxyz,4);o<<",\"target_position_world\":";arr(o,t->position,3);o<<",\"target_quaternion_wxyz\":";arr(o,t->world_quaternion_wxyz,4);o<<",\"velocity_world\":";arr(o,t->velocity,3);o<<",\"omega\":"<<t->vyaw<<",\"intrinsics\":["<<camera.value->fx<<','<<camera.value->fy<<','<<camera.value->cx<<','<<camera.value->cy<<"],\"armors\":[";
 for(int j=0;j<4;j++){if(j)o<<',';auto&a=t->armors[j];o<<"{\"slot\":"<<int(a.relative_slot)<<",\"position\":";arr(o,a.relative_position,3);o<<",\"normal\":";arr(o,a.outward_normal,3);o<<'}';}o<<"],\"binding_frame_seq\":"<<binding_seq<<",\"bound_initial_position_world\":";arr(o,bound_position.data(),3);o<<"}";auto s=o.str();uint32_t n=s.size();std::cout.write(reinterpret_cast<char*>(&n),4);std::cout.write(s.data(),s.size());std::cout.write(reinterpret_cast<const char*>(f.image.payload.data()),f.image.payload.size());std::cout.flush();if(!std::cout)throw std::runtime_error("pipe failed");last=h.capture_timestamp_ns;saved++;
 }std::cerr<<"saved="<<saved<<" exact_metadata_misses="<<misses<<'\n';return saved==count?0:2;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
