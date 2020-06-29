#include "common_funcs.h"
#include "logger_ini.h"

#include <chrono>
#include <future>

void show_img(const std::string& win_name, const cv::Mat& img, int delay) {
	cv::namedWindow(win_name, cv::WINDOW_NORMAL);
	cv::imshow(win_name, img);
	cv::waitKey(delay);
}

// morphologic filter
// morphType  0:ERODE  1:DILATE  2:OPEN  3:CLOSE  4:GRADIENT  5:TOPHAT
// 6:BLACKHAT
// kernalShape  0:RECT  1:CROSS  2:ELLIPSE
cv::Mat morph(const cv::Mat& img, int morphType, int kernalSize, int iteration,
  int kernalShape) {
  cv::Mat element =
    cv::getStructuringElement(kernalShape, cv::Size(kernalSize, kernalSize));
  cv::Mat morphImg;
  cv::morphologyEx(img, morphImg, morphType, element, cv::Point(-1, -1),
    iteration);
  return morphImg;
}

// *************************denoise*************************
cv::Mat denoise_sub(const bool& ignore_edge, const cv::Mat& labels, const cv::Mat& stats,
  const std::vector<int>& ignore_direction, const float& denoise_area, const cv::Size& size,
  const size_t& start, const size_t& end) {
  //auto start_t = std::chrono::steady_clock::now();
  cv::Mat re= cv::Mat::zeros(size, CV_8UC1);
  bool at_edge = false;

  for (size_t i = start; i < end; i++) {
    at_edge = false;
    if (ignore_edge) {
      at_edge = ignore_direction[0] * cv::sum((labels == i).row(0))[0] +
        ignore_direction[2] * cv::sum((labels == i).col(0))[0] +
        ignore_direction[1] * cv::sum((labels == i).row((labels == i).size().height - 1))[0] +
        ignore_direction[3] * cv::sum((labels == i).col((labels == i).size().width - 1))[0];
    }
    if (!at_edge && stats.at<int>(i, cv::CC_STAT_AREA) > denoise_area) {
      re+=(labels == i);  // (labels==i) return a binary image
    }
  }
  //auto end_t = std::chrono::steady_clock::now();
  //auto t = std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t);
  //double t_d = t.count();
  //LOG4CPLUS_WARN(logger, "denoise_sub用时" << t_d / 1000 << "毫秒");
  return re;
}

// 计算每个线程分配到的循环次数
std::vector<int> cal_loopnum_thread(const size_t& loop_total, const size_t& thread_num) {
  uint rest = loop_total % thread_num;
  std::vector<int> loopnum(rest, loop_total / thread_num+1);
  std::vector<int> loopnum_back(thread_num-rest, loop_total / thread_num);
  loopnum.insert(loopnum.end(), loopnum_back.begin(), loopnum_back.end());
  return loopnum;
}

// delete small areas in binary image
// ignore_edge: if true, delete components at edge
// mode: area value: 0, fixed value
//                   1, calculate area via max areas in components
//                   factor*mean(max_areas);
// max_num: 计算最大面积均值时取的面积最大的components的个数
// ignore_direction:{up,down,left,right},对应位置为1则删除该边界处的components
void denoise(const cv::Mat& binary, cv::Mat* denoise_img, const int area,
  const bool ignore_edge, const int mode, const float factor,
  const int max_num, const std::vector<int>& ignore_direction) {
  cv::Mat labels, stats, centroids;
  int component_num =
    cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
  *denoise_img = cv::Mat::zeros(binary.size().height, binary.size().width, CV_8UC1);
  float denoise_area = area;
  if (mode == 1) {
    cv::Mat areas = stats.col(cv::CC_STAT_AREA);
    cv::Mat areas_sort;
    cv::sort(areas, areas_sort, cv::SORT_EVERY_COLUMN + cv::SORT_DESCENDING);
    denoise_area = factor * (cv::mean(areas_sort.rowRange(1, max_num + 1))[0]);
  }
  size_t thread_num = 8;
  std::vector<std::future<cv::Mat> > future_sub_vec;
  
  auto loopnum=cal_loopnum_thread(size_t(component_num)-1, thread_num);

  size_t start = 1;
  size_t end = 1;
  for (size_t i = 0; i < thread_num; i++) {
    end = start + loopnum[i];
    future_sub_vec.emplace_back(std::async(std::launch::async, denoise_sub,
      std::ref(ignore_edge), std::ref(labels),std::ref(stats), std::ref(ignore_direction),
      std::ref(denoise_area), binary.size(), start, end));
    start+= loopnum[i];
  }
 
  //auto com_ind=denoise_sub(ignore_edge, labels, stats, ignore_direction,
  //              denoise_area, binary.size(), 0, component_num);

  for (size_t i = 0; i < thread_num; i++) {
    cv::Mat denoise_sub_img = future_sub_vec[i].get();
    std::string win_n = "denoise_img" + std::to_string(i);
    //show_img(win_n, denoise_sub_img, 0);
    cv::bitwise_or(*denoise_img, denoise_sub_img, *denoise_img);
    //show_img("denoise_sub_img", *denoise_img, 0);
  }
}
// *************************denoise end*************************

/*
result_img: 画板图像，内容将会被更改
outer_only: if true, detect outer contours only.
min_len, max_len length range of rect
min_wid, max_wid width range of rect
only the rect in range will return
rectInfo : // centerX, centerY, angle, height, width(原始rect)
         : centerX, centerY, angle(顺时针正), width短边, length长边.
*/
std::vector<std::vector<float>> boundingRectRotate(const cv::Mat& binary_img,
  cv::Mat& result_img, bool integrate, bool outer_only,
  uint16_t min_len, uint16_t max_len,
  uint16_t min_wid, uint16_t max_wid,
  int offset_x, int offset_y) {
  std::vector<std::vector<float>> resultInfo;
  // channels convert
  if (result_img.channels() == 1) {
    cv::cvtColor(result_img, result_img,
      cv::COLOR_GRAY2BGR);  // convert grayImg to 3 channels
  }
  else if (result_img.channels() == 3) {}
  else if (result_img.channels() == 4) {
    cv::cvtColor(result_img, result_img,
      cv::COLOR_BGRA2BGR);  // convert grayImg to 3 channels
  }
  else {
    std::cout << "Input result image(arg 2) has " << result_img.channels()
      << " channel(s), expect 1, 3 or 4. Draw result fail!" << std::endl;
  }
  int line_width = (result_img.size().height + result_img.size().width) / 1988 + 1;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  int mode = (outer_only) ? cv::RETR_EXTERNAL : cv::RETR_TREE;
  cv::findContours(binary_img, contours, hierarchy, mode,
    cv::CHAIN_APPROX_NONE, cv::Point(offset_x, offset_y));
  // cv::drawContours(rectImg, contours, -1, cv::Scalar(0, 255, 0), line_width,
  // 8, hierarchy, 1, cv::Point());
  std::vector<cv::Point> contour_integrate;
  if (contours.size() != 0) {
    for (int i = 0; i < contours.size(); i++) {
      if (integrate) {
        contour_integrate.insert(contour_integrate.end(), contours[i].begin(),
          contours[i].end());
      }
      else {
        cv::RotatedRect rRect = cv::minAreaRect(cv::Mat(contours[i]));
        float len = std::max(rRect.size.height, rRect.size.width);
        float wid = std::min(rRect.size.height, rRect.size.width);
        std::cout << "len = " << len << " wid = " << wid << std::endl;
        if (min_len < len && len < max_len && min_wid < wid && wid < max_wid) {
          cv::Point2f rect_points[4];
          rRect.points(rect_points);
          for (int j = 0; j < 4; j++) {
            line(result_img, rect_points[j], rect_points[(j + 1) % 4],
              cv::Scalar(0, 255, 0), line_width, 8);
          }
          float ang = (rRect.size.height <= rRect.size.width) ?
            rRect.angle : rRect.angle + 90;
          std::vector<float> rectInfo;
          rectInfo.push_back(rRect.center.x);
          rectInfo.push_back(rRect.center.y);
          rectInfo.push_back(ang);
          rectInfo.push_back(wid);
          rectInfo.push_back(len);
          resultInfo.push_back(std::move(rectInfo));
        }
      }
    }
    if (integrate) {
      cv::RotatedRect rRectIntegrate =
        cv::minAreaRect(cv::Mat(contour_integrate));
      float len_in = std::max(rRectIntegrate.size.height,
        rRectIntegrate.size.width);
      float wid_in = std::min(rRectIntegrate.size.height,
        rRectIntegrate.size.width);
      std::cout << "len_in = " << len_in << " wid_in = " << wid_in << std::endl;
      if (min_len < len_in && len_in < max_len &&
        min_wid < wid_in && wid_in < max_wid) {

        cv::Point2f rect_integrate_points[4];
        rRectIntegrate.points(rect_integrate_points);
        for (int j = 0; j < 4; j++) {
          line(result_img, rect_integrate_points[j],
            rect_integrate_points[(j + 1) % 4], cv::Scalar(0, 255, 0),
            line_width, 8);
        }
        float ang = (rRectIntegrate.size.height <=
          rRectIntegrate.size.width) ?
          rRectIntegrate.angle : rRectIntegrate.angle + 90;
        std::vector<float> rectIntegrateInfo;
        rectIntegrateInfo.push_back(rRectIntegrate.center.x);
        rectIntegrateInfo.push_back(rRectIntegrate.center.y);
        rectIntegrateInfo.push_back(ang);
        rectIntegrateInfo.push_back(wid_in);
        rectIntegrateInfo.push_back(len_in);
        resultInfo.push_back(std::move(rectIntegrateInfo));
      }
    }
  }
  std::cout << "Rect num = " << resultInfo.size() << std::endl;
  return resultInfo;
}