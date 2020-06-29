#pragma once
#include <opencv2/opencv.hpp>

// 模板 点集和锚点索引列表
struct TmpltPoly {
	std::vector<cv::Point> tmplt_pts = {};
	std::vector<int> anchor_indices = {};
};

// 每个点的angle, distance一一对应
struct AngDist {
	std::vector<float> angles = {};
	std::vector<float> dists = {};
};

// 每个锚点的匹配结果
struct Match_Result {
	float match_score = 0;
	cv::Point match_anchor = cv::Point(0, 0);
	cv::Point shift = cv::Point(0, 0);
	float match_angle = 0;
};