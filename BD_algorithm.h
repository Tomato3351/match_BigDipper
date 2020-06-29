#pragma once
#include "defination.h"

#include <opencv2/opencv.hpp>


void draw_points(cv::Mat* points_image, const std::vector<cv::Point>& pts,
	const cv::Scalar& ptcolor = cv::Scalar(255, 255, 255));
void draw_polygon(cv::Mat* poly_image, const std::vector<cv::Point>& polygon_pts,
	int mode = 0, const cv::Scalar ptcolor = cv::Scalar(255, 255, 255));

void get_polypts(const cv::Mat& binary, std::vector<cv::Point>* template_points,
	cv::Mat* poly_image,
	const float& epsilon = 3.0, const bool& draw_poly=true);

void create_template(const cv::Mat& morph_img, cv::Mat* poly_img,
	std::string save_path = "template.xml");

void read_tmplt(const std::string& file_path, TmpltPoly* tmplt);

float EuclideanDist(const cv::Point2f& pt0, const cv::Point2f& pt1);
float Angle2P(const cv::Point2f& pt0, const cv::Point2f& pt1);

void cal_angle_dist(const std::vector<cv::Point>& pts,
	const int& index, AngDist* sorted);

std::vector<std::vector<int> >  find_match(const std::vector<float>& tmplt_vec,
	const std::vector<float>& vec, const float& error);

void angle2zero(const AngDist& sorted, AngDist* j2zero,
	const int& index);

void get_subvec_ind(const std::vector<float>& vec, const std::vector<int>& ind,
	std::vector<float>* sub_vec);

Match_Result match_one_anchor(const TmpltPoly& tem, const int& tem_anchor_ind,
	const std::vector<cv::Point>& pts);

void tmplt_transform(const TmpltPoly& tem, const Match_Result& match_re,
	cv::Mat* result_img, std::vector<std::vector<float>>* result_info);