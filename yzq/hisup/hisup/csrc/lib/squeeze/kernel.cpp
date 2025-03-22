#include "squeeze.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/** 3/2 pi*/
#define M_3_2_PI 4.71238898038
/** 2 pi */
#define M_2__PI 6.28318530718


class Point {
public:
    Point(){}
    Point(float x, float y, float ang) :
        x_(x), y_(y), ang_(ang), used_(false) {}
    inline float x() const { return this->x_; }
    inline float y() const { return this->y_; }
    inline float ang() const { return this->ang_;}
    inline bool used() const { return this->used_;}
    inline float &x() { return this->x_;}
    inline float &y() { return this->y_;}
    inline float &ang() { return this->ang_;}
    inline bool &used() { return this->used_;}
private:
    float x_,y_, ang_;
    bool used_;
};

class Rectangle
{
public:
    float x1, y1, x2, y2;
    float width;
    float l_min, l_max, w_min, w_max;
    float x,y;
    float theta;
    float dx,dy;
    float prec;
    float p;
};

class PoLsMap {
public:
    struct PoLs {
        PoLs() {inds.clear(); }
        ~PoLs() {inds.clear(); }
        inline void push_back(int id) {inds.push_back(id); }
        inline size_t size() { return inds.size(); }
        std::vector<int> inds;
    };

    PoLsMap(int height, int width, const float *arr_x, const float *arr_y, const float *arr_ang, int cnt)
    : height_(height), width_(width)
    {
        pols_ = new PoLs[height*width];
        for(int i = 0; i < cnt; ++i)
        {
            int x_int = int(arr_x[i]);
            int y_int = int(arr_y[i]);
            points_.push_back(Point(arr_x[i],arr_y[i],arr_ang[i]));
            pols_[y_int*width_ + x_int].push_back(i);
        }
    }
    const PoLs& operator () (int x, int y) const
    {
        return pols_[y*width_ + x];
    }
    PoLs& operator() (int x, int y)
    {
        return pols_[y*width_ + x];
    }

    const Point& operator () (int x, int y, int id) const
    {
        return points_[pols_[y*width_+x].inds[id]];
    }
    Point& operator () (int x, int y, int id)
    {
        return points_[pols_[y*width_+x].inds[id]];
    }

    int height_, width_;
    PoLs* pols_;
    std::vector<Point> points_;

};
//通过计算角度差并判断是否小于等于精度值 prec
bool isaligned(float phi, float theta, float prec)
{
    theta -= phi;
    if (theta < 0.0) theta = - theta;
    if (theta > M_3_2_PI)
    {
        theta -= M_2__PI;
        if (theta < 0.0) theta = -theta;
    }

    return theta <=prec;
}
//计算两个角度之间的差值，并将其限制在 -π 到 π 的范围内。
float angle_diff(float a, float b)
{
    a -= b;
    while( a <= -M_PI ) a += M_2__PI;
    while( a >   M_PI ) a -= M_2__PI;
    if( a < 0.0 ) a = -a;
    return a;
}
//从给定的起始点 (x, y) 开始，根据角度 ang 和精度 prec，生长出一个区域。
//使用 isaligned 函数判断点是否符合角度条件，并将符合条件的点加入到区域中。
//返回生长后的区域角度 reg_ang、区域点集 reg、区域内部点集 reg_int 和置信度 confidence。
void region_grow(int x, int y, float ang,
                float &reg_ang,
                std::vector<Point>& reg,
                std::vector<Point>& reg_int,
                std::vector<float>& confidence,
                PoLsMap& map,
                float prec)
{
    reg.clear();
    reg_int.clear();
    confidence.clear();
    if (map(x,y).size() == 0)
        return;
    reg_ang = ang;
    float sumdx = 0;
    float sumdy = 0;
    int cnt = 0;
    for(int id = 0; id < map(x,y).size(); ++id) {
        Point& pt = map(x,y,id);
        if (pt.used())
            continue;
        if (isaligned(reg_ang,pt.ang(),prec))
        {
            pt.used() = true;
            reg.push_back(pt);

            sumdx += cos(pt.ang());
            sumdy += sin(pt.ang());
            reg_ang = atan2(sumdy, sumdx);
            ++cnt;
        }
    }

    if (reg.size() == 0)
        return;
//    std::vector<Point> reg_int;

    reg_int.push_back(Point(int(x),int(y),reg_ang));
    confidence.push_back((float)cnt/(float)map(x,y).size());


    //在区域生长过程中，角度的计算可能会受到局部噪声的影响，导致锯齿化现象。可以通过对角度进行平滑处理来减少这种影响。
    float smooth_factor = 0.5; //增加平滑因子
    for(int i = 0; i < reg_int.size(); ++i)
    {
        for(int xx = (int)reg_int[i].x()-1; xx<= (int)reg_int[i].x()+1; ++xx)
        {
            for(int yy = (int)reg_int[i].y()-1; yy<= (int)reg_int[i].y()+1; ++yy)
            {
                if (xx<0 || yy<0 || xx >= map.width_ || yy >= map.height_ || map(xx,yy).size() <=1)
                    continue;
                bool flag = false;
                cnt = 0;
                for(int k = 0; k < map(xx,yy).size(); ++k) {
                    Point& pt = map(xx,yy,k);
                    if(isaligned(reg_ang, pt.ang(), prec)&&pt.used()==false)
                    {
                        reg.push_back(pt);
                        pt.used() = true;
                        flag = true;
                        sumdx += cos(pt.ang());
                        sumdy += sin(pt.ang());
                        //平滑角度更新
                         reg_ang = smooth_factor * atan2(sumdy, sumdx) + (1 - smooth_factor) * reg_ang;
                        //reg_ang = atan2(sumdy, sumdx);
                        ++cnt;
                    }
                }
                if (flag == false)
                    continue;
                // for(int k = 0; k < map(xx,yy).size(); ++k) {
                //     Point& pt = map(xx,yy,k);
                //     pt.used() = true;
                // }


                // 连通性检查
                bool is_connected = false;
                for (const auto& p : reg_int) {
                    if (abs(p.x() - xx) <= 1 && abs(p.y() - yy) <= 1) {
                        is_connected = true;
                        break;
                    }
                }
                if (!is_connected) {
                    continue;
                }

                reg_int.push_back(Point(xx,yy,reg_ang));
                confidence.push_back((float)cnt/(float)map(xx,yy).size());
            }
        }
    }
}

//将生长出的区域转换为矩形。
//计算矩形的中心点 (x, y)、方向角 theta、长度范围 (l_min, l_max) 和宽度范围 (w_min, w_max)。
bool region2rect(const std::vector<Point> &reg_int,
                 const std::vector<float> &confidence,
                 float reg_angle, float prec, float p, Rectangle &rect)
{
    float x,y,dx,dy,l,w,theta,weight,sum,l_min,l_max,w_min,w_max;
    x = y = sum = 0.0;
    for(int i=0; i<reg_int.size(); ++i)
    {
        weight = confidence[i];
        // weight = 1.0;
        x += reg_int[i].x()*weight;
        y += reg_int[i].y()*weight;
        sum += weight;
    }
    if (sum<1e-6)
        return false;
    x /= sum;
    y /= sum;

    // theta = get_theta(reg_int, confidence,reg_angle, prec, x,y);
    theta = reg_angle;

    dx = cos(theta);
    dy = sin(theta);
    l_min = l_max = w_max = w_min = 0.0;
    for(int i=0; i<reg_int.size(); ++i)
    {
        l = (reg_int[i].x() - x)*dx + (reg_int[i].y() - y)*dy;
        w = -(reg_int[i].x() - x)*dy + (reg_int[i].y() - y)*dx;
      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w > w_max ) w_max = w;
      if( w < w_min ) w_min = w;
    }
    // 平滑边界处理
    float smooth_margin = 0.5; // 平滑边界范围

    l_min -= smooth_margin;
    l_max += smooth_margin;
    w_min -= smooth_margin;
    w_max += smooth_margin;

    rect.x1 = x + l_min*dx;
    rect.y1 = y + l_min*dy;
    rect.x2 = x + l_max*dx;
    rect.y2 = y + l_max*dy;
    rect.l_min = l_min;
    rect.l_max = l_max;
    rect.w_min = w_min;
    rect.w_max = w_max;
    rect.width = w_max - w_min;
    rect.x = x;
    rect.y = y;
    rect.theta = theta;
    rect.dx = dx;
    rect.dy = dy;
    rect.prec = prec;
    rect.p = p;
    if (rect.width < 1.0)
        rect.width = 1.0;

    // return rect.width/(rect.l_max-rect.l_min)<0.3;
    return true;
}

// (int x, int y, float ang,
//                 float &reg_ang,
//                 std::vector<Point>& reg,
//                 std::vector<Point>& reg_int,
//                 std::vector<float>& confidence,
//                 PoLsMap& map,
//                 float prec)


//对矩形区域进行优化。
//通过旋转和投影，去除不符合矩形形状的点。
//更新点云映射，为后续的区域生长做准备。
void refine(std::vector<Point> &reg_int,
            Rectangle& rect, std::vector<float> &confidence, PoLsMap& map)
{
    
    float xc = rect.x;
    float yc = rect.y;
    float ang_rect = rect.theta;    
    float dx  = cos(ang_rect), dy = sin(ang_rect);
    
    float x1 = rect.l_min*dx - rect.w_min*dy;
    float y1 = rect.l_min*dy + rect.w_min*dx;
    float x2 = rect.l_max*dx - rect.w_min*dy;
    float y2 = rect.l_max*dy + rect.w_min*dx;
    float x3 = rect.l_max*dx - rect.w_max*dy;
    float y3 = rect.l_max*dy + rect.w_max*dx;
    float x4 = rect.l_min*dx - rect.w_max*dy;
    float y4 = rect.l_min*dy + rect.w_max*dx;

    std::vector<Point> reg_rot;
    for(int i = 0; i < reg_int.size(); ++i)
    {
        Point p;
        p.x() = dx*(reg_int[i].x()-xc) + dy*(reg_int[i].y()-yc);
        p.y() = -dy*(reg_int[i].x()-xc) + dx*(reg_int[i].y()-yc);
        reg_rot.push_back(p);
        // std::cout<<reg_int[i].x()<<" "<<reg_int[i].y()<<" "<<p.x()<<" "<<p.y()<<"\n";
    }
    int start_w = (int) floor(rect.w_min);
    int end_w = (int) ceil(rect.w_max);
    int start_l = (int) floor(rect.l_min);
    int end_l = (int) ceil(rect.l_max);
    std::vector< std::vector<int> > indexes(end_w-start_w);

    for(int i = 0; i < reg_rot.size(); ++i)
    {
        const Point &p = reg_rot[i];
        if (p.x()<start_l || p.x()>=end_l || p.y()<start_w || p.y()>=end_w){
            continue;
        }
        for(int k = start_w; k < end_w; ++k)
        {
            if(p.y()>=k && p.y()<k+1)
            {
                indexes[k-start_w].push_back(i);
                break;
            }
        }
    }


     // 优化区域
    for(int k = start_w; k < end_w; ++k)
    {
        float ratio = (float)(indexes[k-start_w].size()/(end_l-start_l));
        const std::vector<int>& local_ind = indexes[k-start_w];
        if(ratio<=0.1)// 如果比例小于 10%，认为是噪声列
            continue;
        for(int i = 0; i < local_ind.size();++i)
        {
            int id = local_ind[i];
            int x = reg_int[id].x();
            int y = reg_int[id].y();
            for(int n = 0; n < map(x,y).size();++n)
                map(x,y,n).used() = false;
        }
    }

    // 重新计算矩形边界
    float new_l_min = std::numeric_limits<float>::max();
    float new_l_max = -std::numeric_limits<float>::max();
    float new_w_min = std::numeric_limits<float>::max();
    float new_w_max = -std::numeric_limits<float>::max();
    for (const auto& p : reg_rot) {
        float l = p.x();
        float w = p.y();
        if (l < new_l_min) new_l_min = l;
        if (l > new_l_max) new_l_max = l;
        if (w < new_w_min) new_w_min = w;
        if (w > new_w_max) new_w_max = w;
    }

    rect.l_min = new_l_min;
    rect.l_max = new_l_max;
    rect.w_min = new_w_min;
    rect.w_max = new_w_max;
    rect.width = new_w_max - new_w_min;





//    for(int k = start_w; k < end_w; ++k)
//    {
//        float ratio = (float)(indexes[k-start_w].size()/(end_l-start_l));
//        const std::vector<int>& local_ind = indexes[k-start_w];
//        if(ratio<=0.1)
//            continue;
//        for(int i = 0; i < local_ind.size();++i)
//        {
//            int id = local_ind[i];
//            int x = reg_int[id].x();
//            int y = reg_int[id].y();
//            for(int n = 0; n < map(x,y).size();++n)
//                map(x,y,n).used() = false;
//        }
//    }



    // int xx = reg_int[0].x();
    // int yy = reg_int[0].y();
    // std::vector<Point> reg_new;
    // confidence.clear();
    // reg_int.clear();
    // region_grow(xx,yy,ang_rect,ang_rect,reg_new,reg_int,confidence,map,10.0/180.0*M_PI);
}
//输入参数包括图像高度 H、宽度 W、点云数据的坐标数组 (arr_x, arr_y)、角度数组 arr_ang 和点的数量 cnt;输出参数包括矩形数组 rectangles 和矩形数量 num。

//1.初始化点云映射 map。
//2.遍历每个点，调用 region_grow 函数生长区域。
//3.将生长出的区域转换为矩形，并存储到 vec_rects 中。
//4.对每个矩形调用 refine 函数进行优化。
//5.将最终的矩形信息存储到输出数组 rectangles 中。

void _region_grow(int H, int W,
                const float* arr_x, const float* arr_y, const float* arr_ang,int cnt,
                float* rectangles, int *num)
{
    float prec = 10.0/180.0*M_PI; //rad = degree/180*pi
    float p = prec/M_PI;

    PoLsMap map(H, W, arr_x, arr_y, arr_ang, cnt);

    std::vector<Rectangle> vec_rects;    
    for(int i = 0; i < cnt; ++i)
    {
        std::vector<Point> reg, reg_int;
        std::vector<float> confidence;
        float reg_ang = 0;
        region_grow(arr_x[i],arr_y[i],arr_ang[i],reg_ang,reg,reg_int,confidence,map,prec);
        if(reg_int.size() <= 5)
            continue;
        Rectangle rect;
        if(!region2rect(reg_int, confidence, reg_ang, prec, p, rect ))
            continue;
        vec_rects.push_back(rect);
    }
    *num = vec_rects.size();

    for(int i = 0; i < *num; ++i)
    {
        // x1,y1,x2,y2,ratio
        // ratio = width/length
        rectangles[5*i + 0] = vec_rects[i].x1;
        rectangles[5*i + 1] = vec_rects[i].y1;
        rectangles[5*i + 2] = vec_rects[i].x2;
        rectangles[5*i + 3] = vec_rects[i].y2;
        float length = sqrt((vec_rects[i].x1-vec_rects[i].x2)*(vec_rects[i].x1-vec_rects[i].x2) +
                            (vec_rects[i].y1-vec_rects[i].y2)*(vec_rects[i].y1-vec_rects[i].y2));
        rectangles[5*i + 4] = vec_rects[i].width;
    }
}