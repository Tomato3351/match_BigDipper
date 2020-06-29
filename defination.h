#pragma once
#include <opencv2/opencv.hpp>

// ģ�� �㼯��ê�������б�
struct TmpltPoly {
	std::vector<cv::Point> tmplt_pts = {};
	std::vector<int> anchor_indices = {};
};

// ÿ�����angle, distanceһһ��Ӧ
struct AngDist {
	std::vector<float> angles = {};
	std::vector<float> dists = {};
};

// ÿ��ê���ƥ����
struct Match_Result {
	float match_score = 0;
	cv::Point match_anchor = cv::Point(0, 0);
	cv::Point shift = cv::Point(0, 0);
	float match_angle = 0;
};