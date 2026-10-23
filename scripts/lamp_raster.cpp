#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
extern "C" void raster(const double* v, const int32_t* triangles, const uint8_t* material, int count, int width, int height, double pixel_center, uint8_t* image, double* depth) {
    std::fill(image,image+width*height,0);
    std::fill(depth,depth+width*height,std::numeric_limits<double>::infinity());
    for(int t=0;t<count;++t) {
        const double *a=v+3*triangles[t*3],*b=v+3*triangles[t*3+1],*c=v+3*triangles[t*3+2];
        if(std::min({a[2],b[2],c[2]})<=0)continue;
        double det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
        if(std::abs(det)<1e-10)continue;
        int x0=std::max(0,(int)std::floor(std::min({a[0],b[0],c[0]})));
        int x1=std::min(width-1,(int)std::ceil(std::max({a[0],b[0],c[0]})));
        int y0=std::max(0,(int)std::floor(std::min({a[1],b[1],c[1]})));
        int y1=std::min(height-1,(int)std::ceil(std::max({a[1],b[1],c[1]})));
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x) {
            double px=x+pixel_center,py=y+pixel_center;
            double u=((b[1]-c[1])*(px-c[0])+(c[0]-b[0])*(py-c[1]))/det;
            double w=((c[1]-a[1])*(px-c[0])+(a[0]-c[0])*(py-c[1]))/det;
            double q=1-u-w;
            if(u < -1e-8 || w < -1e-8 || q < -1e-8)continue;
            double z=1/(u/a[2]+w/b[2]+q/c[2]);int idx=y*width+x;
            if(z<depth[idx]){depth[idx]=z;image[idx]=material[t];}
        }
    }
}
