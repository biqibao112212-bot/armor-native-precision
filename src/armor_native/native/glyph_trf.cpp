// Trust-region reflective specialization for four camera-translation/yaw parameters.
// Algorithm follows SciPy optimize._lsq.trf/common (SciPy BSD-3-Clause).
// Preserve lower z bound, soft_l1 f_scale=.5, 40 evaluations, 1e-7 tolerances.
// Analytic projection Jacobian replaces finite differences.
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>
using V=Eigen::Vector4d;using M=Eigen::Matrix4d;
using A=Eigen::MatrixXd;
static double bound_stride(const V&x,const V&p){return p[2]<0?(.05-x[2])/p[2]:std::numeric_limits<double>::infinity();}
static double quadratic(const M&H,const V&g,const V&s){return .5*s.dot(H*s)+g.dot(s);}
static double min1(double a,double b,double lo,double hi){
 double best=lo,val=a*lo*lo+b*lo;
 if(a*hi*hi+b*hi<val){best=hi;val=a*hi*hi+b*hi;}
 if(a!=0){double t=-.5*b/a;if(t>lo&&t<hi&&a*t*t+b*t<val)best=t;}return best;
}
static V trust_step(const V&s,const M&vm,const V&uf,int m,double delta,double&alpha){
 V suf=s.cwiseProduct(uf);bool rank=s[3]>std::numeric_limits<double>::epsilon()*m*s[0];
 if(rank){V p=-vm*(uf.cwiseQuotient(s));if(p.norm()<=delta){alpha=0;return p;}}
 auto phi=[&](double a){V den=s.array().square().matrix()+V::Constant(a);V t=suf.cwiseQuotient(den);double norm=t.norm();
 return std::pair<double,double>(norm-delta,-(suf.array().square()/den.array().cube()).sum()/norm);};
 double upper=suf.norm()/delta,lower=0;
 if(rank){auto q=phi(0);lower=-q.first/q.second;}
 if(!rank&&alpha==0)alpha=std::max(.001*upper,std::sqrt(lower*upper));
 for(int i=0;i<10;i++){
  if(alpha<lower||alpha>upper)alpha=std::max(.001*upper,std::sqrt(lower*upper));
  auto q=phi(alpha);if(q.first<0)upper=alpha;
  double ratio=q.first/q.second;lower=std::max(lower,alpha-ratio);alpha-=(q.first+delta)*ratio/delta;
  if(std::abs(q.first)<.01*delta)break;
 }
 V p=-vm*suf.cwiseQuotient(s.array().square().matrix()+V::Constant(alpha));p*=delta/p.norm();return p;
}
extern "C" int glyph_trf(int n,const double* points,const double* camera,const double* observed,const double* weights,const double* K,const double* initial,double* output){
 Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> W(camera);int m=2*n;
 auto evaluate=[&](const V&x,A*J,Eigen::VectorXd*f){
  if(J)J->resize(m,4);if(f)f->resize(m);
  double cost=0,c=std::cos(x[3]),s=std::sin(x[3]);
  for(int i=0;i<n;i++){
   Eigen::Vector3d v(c*points[3*i]-s*points[3*i+1],s*points[3*i]+c*points[3*i+1],points[3*i+2]);
   Eigen::Vector3d q=W*v+x.head<3>(),dy=W*Eigen::Vector3d(-v.y(),v.x(),0);
   for(int axis=0;axis<2;axis++){
    double r=weights[i]*(K[axis]*q[axis]/q.z()+K[axis+2]-observed[2*i+axis]);
    double root=std::sqrt(1+4*r*r);cost+=.25*(root-1);
    if(J){
     Eigen::RowVector3d dp=Eigen::RowVector3d::Zero();dp[axis]=K[axis]/q.z();dp[2]=-K[axis]*q[axis]/(q.z()*q.z());
     V j;j.head<3>()=weights[i]*dp.transpose();j[3]=weights[i]*dp.dot(dy);
     double js=std::sqrt(std::max(1/(root*root*root),std::numeric_limits<double>::epsilon()));
     J->row(2*i+axis)=js*j.transpose();(*f)[2*i+axis]=r/(root*js);
    }
   }
  }return cost;
 };
 V x=Eigen::Map<const V>(initial);if(x[2]<=.05)x[2]=.05+1e-10;
 A J;Eigen::VectorXd f;double cost=evaluate(x,&J,&f);V g=J.transpose()*f;
 V v=V::Ones();if(g[2]>0)v[2]=x[2]-.05;
 double delta=x.cwiseQuotient(v.cwiseSqrt()).norm();if(delta==0)delta=1;
 double alpha=0;int nfev=1,status=0;
 while(true){
  v.setOnes();V dv=V::Zero();if(g[2]>0){v[2]=x[2]-.05;dv[2]=1;}
  double gn=g.cwiseProduct(v).lpNorm<Eigen::Infinity>();if(gn<1e-7)status=1;
  if(status||nfev>=40)break;
  V d=v.cwiseSqrt(),diag=g.cwiseProduct(dv),gh=d.cwiseProduct(g);A ja(m+4,4);
  ja.topRows(m)=J*d.asDiagonal();ja.bottomRows(4)=diag.cwiseSqrt().asDiagonal();
  Eigen::JacobiSVD<A> svd(ja,Eigen::ComputeThinU|Eigen::ComputeFullV);V s=svd.singularValues();
  V uf=svd.matrixU().topRows(m).transpose()*f;M H=ja.transpose()*ja;
  double theta=std::max(.995,1-gn),actual=-1,next=cost;V xn=x;
  while(actual<=0&&nfev<40){
   V ph=trust_step(s,svd.matrixV(),uf,m,delta,alpha),p=d.cwiseProduct(ph),sh=ph,step=p;
   double predicted;
   if(x[2]+p[2]>=.05){predicted=-quadratic(H,gh,ph);}
   else{
    double ps=bound_stride(x,p);V rh=ph;rh[2]*=-1;V r=d.cwiseProduct(rh);p*=ps;ph*=ps;V xb=x+p;
    double aa=rh.squaredNorm(),bb=ph.dot(rh),cc=ph.squaredNorm()-delta*delta;
    double totr=(-bb+std::sqrt(std::max(0.,bb*bb-aa*cc)))/aa,tob=bound_stride(xb,r),rs=std::min(tob,totr);
    double low=rs>0?(1-theta)*ps/rs:0,high=rs>0?(rs==tob?theta*tob:totr):-1;
    double rv=std::numeric_limits<double>::infinity();
    if(low<=high){double a=.5*rh.dot(H*rh),b=rh.dot(H*ph)+gh.dot(rh);rs=min1(a,b,low,high);rh=ph+rs*rh;r=d.cwiseProduct(rh);rv=quadratic(H,gh,rh);}
    p*=theta;ph*=theta;double pv=quadratic(H,gh,ph);
    V ah=-gh,ag=d.cwiseProduct(ah);totr=delta/ah.norm();tob=bound_stride(x,ag);
    double astride=tob<totr?theta*tob:totr;astride=min1(.5*ah.dot(H*ah),gh.dot(ah),0,astride);ah*=astride;ag*=astride;
    double av=quadratic(H,gh,ah);
    if(pv<rv&&pv<av){step=p;sh=ph;predicted=-pv;}
    else if(rv<pv&&rv<av){step=r;sh=rh;predicted=-rv;}
    else{step=ag;sh=ah;predicted=-av;}
   }
   xn=x+step;if(xn[2]<=.05)xn[2]=std::nextafter(.05,std::numeric_limits<double>::infinity());
   next=evaluate(xn,nullptr,nullptr);nfev++;double shn=sh.norm();
   if(!std::isfinite(next)){delta=.25*shn;continue;}
   actual=cost-next;double ratio=predicted>0?actual/predicted:((predicted==0&&actual==0)?1:0),dn=delta;
   if(ratio<.25)dn=.25*shn;else if(ratio>.75&&shn>.95*delta)dn=2*delta;
   bool ft=actual<1e-7*cost&&ratio>.25,xt=step.norm()<1e-7*(1e-7+x.norm());
   if(ft||xt){status=ft?(xt?4:2):3;break;}
   alpha*=delta/dn;delta=dn;
  }
  if(actual>0){x=xn;cost=evaluate(x,&J,&f);g=J.transpose()*f;}
  if(!x.allFinite()||!std::isfinite(cost))return -1;
 }
 Eigen::Map<V> result(output);result=x;output[4]=cost;output[5]=nfev;output[6]=status;return 0;
}

extern "C" int glyph_trf_many(int n,int starts,const double* points,const double* camera,const double* observed,const double* weights,const double*K,const double*initial,double*output){
 for(int i=0;i<starts;i++){int status=glyph_trf(n,points,camera,observed,weights,K,initial+4*i,output+7*i);if(status)return status;}return 0;
}
// Native integer image reads, exactly the two-lamp-cyan-v2/v3 sampling rule.
extern "C" int glyph_lamp(const unsigned char*image,int height,int width,const double*c,const double*config,double*out){
 auto length=[&](int a,int b){return std::hypot(c[2*b]-c[2*a],c[2*b+1]-c[2*a+1]);};
 double lengths[2]={length(0,1),length(3,2)};
 double sep=std::hypot((c[0]+c[2]-c[4]-c[6])*.5,(c[1]+c[3]-c[5]-c[7])*.5);
 double scores[2],medians[2];
 for(int side=0;side<2;side++){
  double len=lengths[side];if(len<1){scores[side]=medians[side]=0;continue;}
  int a=side==0?0:3,b=side==0?1:2;
  double dx=c[2*b]-c[2*a],dy=c[2*b+1]-c[2*a+1],nx=-dy/len,ny=dx/len;
  int radius=std::max(1,std::min({6,int(std::ceil(len*.06)),int(std::max(1.,sep*.15))}));
  int count=std::max(9,std::min(81,int(len))),support=0;double ch[81];
  for(int i=0;i<count;i++){
   double t=i==count-1?.85:.15+i*((.85-.15)/(count-1));bool any=false;double maxch=-1e9;
   for(int d=-radius;d<=radius;d++){
    int x=int(std::nearbyint(c[2*a]+t*dx+d*nx-config[6])),y=int(std::nearbyint(c[2*a+1]+t*dy+d*ny-config[6]));
    bool inside=x>=0&&x<width&&y>=0&&y<height;
    x=std::max(0,std::min(width-1,x));y=std::max(0,std::min(height-1,y));
    const unsigned char*p=image+3*(y*width+x);double red=int(p[0])-std::max(p[1],p[2]),blue=int(p[2])-std::max(p[0],p[1]);
    if(config[7]&&p[2]>=p[1]-config[5])blue=int(p[2])-p[0];
    double chroma=std::max(red,blue);maxch=std::max(maxch,inside?chroma:0.);
    any=any||(inside&&chroma>=config[0]&&std::max({p[0],p[1],p[2]})>=config[1]);
   }support+=any;ch[i]=maxch;
  }
  scores[side]=double(support)/count;std::sort(ch,ch+count);medians[side]=count%2?ch[count/2]:.5*(ch[count/2-1]+ch[count/2]);
 }
 bool positive=true,negative=true;
 for(int i=0;i<4;i++){
  int a=i,b=(i+1)%4,d=(i+2)%4;double cross=(c[2*b]-c[2*a])*(c[2*d+1]-c[2*b+1])-(c[2*b+1]-c[2*a+1])*(c[2*d]-c[2*b]);
  positive=positive&&cross>0;negative=negative&&cross<0;
 }
 bool resolved=std::min(lengths[0],lengths[1])>=config[3]&&sep>=config[4]&&(positive||negative);
 bool accepted=resolved&&std::min(scores[0],scores[1])>=config[2];
 out[0]=accepted;out[1]=accepted?0:(!resolved?1:2);
 out[2]=scores[0];out[3]=scores[1];out[4]=medians[0];out[5]=medians[1];out[6]=lengths[0];out[7]=lengths[1];out[8]=sep;return 0;
}
