#pragma once
//#include "defination.h"

#include <opencv2/opencv.hpp>

#include <iostream>






void show_img(const std::string& win_name, const cv::Mat& img, int delay);

cv::Mat morph(const cv::Mat& img, int morphType, int kernalSize, int iteration, int kernalShape);

void denoise(const cv::Mat& binary, cv::Mat* denoise_img, const int area,
  const bool ignore_edge, const int mode, const float factor,
  const int max_num, const std::vector<int>& ignore_direction);

std::vector<std::vector<float>> boundingRectRotate(const cv::Mat& binary_img,
  cv::Mat& result_img, bool integrate, bool outer_only,
  uint16_t min_len, uint16_t max_len,
  uint16_t min_wid, uint16_t max_wid,
  int offset_x, int offset_y);



