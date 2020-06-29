#include "common_funcs.h"
#include "logger_ini.h"
#include "BD_algorithm.h"

#include <opencv2/opencv.hpp>
//#include <log4cplus/logger.h>
#include <log4cplus/log4cplus.h>
#include <log4cplus/fileappender.h>
#include <log4cplus/configurator.h>
//#include <log4cplus/loggingmacros.h>
//#include <log4cplus/helpers/loglog.h>
//#include <log4cplus/helpers/stringhelper.h>
#include <chrono>

#include <iomanip>
#include <algorithm>
#include <iostream>
#include <future>


bool image_onshow = 0;
uint show_time = 0;
float epsilon = 3.0;

int main()
{	//用Initializer类进行初始化
	log4cplus::Initializer initializer;
	//加载日志配置文件
	log4cplus::PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("./log.properties"));
	//设置日志级别
	//log4cplus::Logger::getRoot().setLogLevel(log4cplus::ALL_LOG_LEVEL);
	// 初始化logger
	//log4cplus::Logger logger = log4cplus::Logger::getRoot();

	//LOG4CPLUS_WARN(logger, "日志");
	//cv::Mat img = cv::imread("Images/baizhentang_imgs/51.jpg", cv::IMREAD_UNCHANGED);
	cv::Mat binary = cv::imread("Images/luquan/luquan2.png", cv::IMREAD_UNCHANGED);
	if(image_onshow){show_img("img", binary, show_time);}
	//cv::Mat img_d, Gaussian, gray, canny;
	//cv::pyrDown(img, img_d);
	//cv::GaussianBlur(img_d, Gaussian, cv::Size(5,5),0);
	//cv::cvtColor(Gaussian, gray, cv::COLOR_BGR2GRAY);
	//cv::Canny(gray, canny, 37, 80, 3, false);
	//if (image_onshow) {show_img("canny", canny, show_time);}
	cv::Mat morph_img=morph(binary, 3, 7, 1, 2);
	if (image_onshow) {show_img("morph_img", morph_img, show_time);}
	std::vector<cv::Point> pts, anchor_pts;
	cv::Mat poly_img;
	get_polypts(morph_img, &pts, &poly_img, 3);
	if (image_onshow) { show_img("poly_img", poly_img, show_time); }

	// 画待匹配图像关键点
	cv::Mat pts_img = cv::Mat::zeros(morph_img.size(), CV_8UC3);
	draw_points(&pts_img, pts);
	if (image_onshow) { show_img("pts_img", pts_img, show_time); }

	//cv::Mat denoise_img;
	//denoise(morph_img, &denoise_img, 1000, false, 0, 0.1, 4, { 1,1,1,1 });



	//create_template(morph_img, &tem_pts, "template.xml");
	//if (image_onshow) { show_img("tem_pts", tem_pts, show_time); }
	TmpltPoly tem;
	read_tmplt("Template.xml", &tem);
	// 画模板关键点
	cv::Mat tem_image = cv::Mat::zeros(morph_img.size(), CV_8UC3);
	draw_points(&tem_image, tem.tmplt_pts);
	if (image_onshow) { show_img("tem_image", tem_image, show_time); }


	int tem_anchor_ind = 5;
	cv::Point tem_anchor = tem.tmplt_pts[tem.anchor_indices[tem_anchor_ind]];
	draw_points(&tem_image, { tem.tmplt_pts[tem.anchor_indices[tem_anchor_ind]] },cv::Scalar(0,0,255));
	if (image_onshow) { show_img("tem_image", tem_image, show_time); }
	// 画模板rect
	cv::RotatedRect rRect_tmplt =
		cv::minAreaRect(tem.tmplt_pts);

	cv::Point2f rect_tmplt_points[4];
	rRect_tmplt.points(rect_tmplt_points);
	for (int j = 0; j < 4; j++) {
		line(tem_image, rect_tmplt_points[j],
			rect_tmplt_points[(j + 1) % 4], cv::Scalar(0, 255, 0),
			1, 8);
	}
	if (image_onshow) { show_img("tem_image", tem_image, show_time); }

	auto start = std::chrono::steady_clock::now();


// ************测试***********
	//std::vector<float> test_vec1 = { -170,-166,-130,-104,-40,6,7,8,9,168 };
	//std::vector<int> test_vec2 = { 0,3,4,6,8,9};


	////sorted.angles = test_vec1;
	////sorted.dists = test_vec2;
	//std::vector<float> sub_vec;
	//get_subvec_ind(test_vec1, test_vec2, &sub_vec);
	//for (auto ele : sub_vec) {
	//	std::cout << ele << std::endl;
	//}

	//angle2zero(sorted, &j2zero, 4);
	//std::cout << "j2zero = " << std::endl;
	//for (int i = 0; i < sorted.angles.size(); i++) {
	//	std::cout << j2zero.angles[i] << std::endl;
	//}
	//std::cout << "j2zero.dists = " << std::endl;
	//for (auto ele : j2zero.dists) {
	//	std::cout << ele << std::endl;
	//}
	//***********测试end********

	//auto re = match_one_anchor(tem, 2, pts);
	//LOG4CPLUS_INFO(logger, "match score is " << re.match_score << std::endl <<
	//	"match_anchor = " << re.match_anchor.x<<","<<re.match_anchor.y << std::endl<<
	//	"match_angle = " <<re.match_angle << std::endl;);

	std::vector<std::future<Match_Result> > match_thread_vec;
	for (int i = 0; i < tem.anchor_indices.size(); i++) {
		match_thread_vec.emplace_back(std::async(std::launch::async, match_one_anchor,
			std::ref(tem), i, std::ref(pts)));
	}
	Match_Result match_result;
	for (int i = 0; i < tem.anchor_indices.size(); i++) {
		Match_Result re = match_thread_vec[i].get();
		if (re.match_score > match_result.match_score) {
			match_result = re;
		}
	}




	std::cout << "match_score = " << match_result.match_score << std::endl;

	cv::Mat result_img;
	if (match_result.match_score > 0.3) {

		cv::cvtColor(morph_img, result_img, cv::COLOR_GRAY2BGR);
		std::vector<std::vector<float>> result_info;
		tmplt_transform(tem, match_result, &result_img, &result_info);
		if (image_onshow) { show_img("result_img", result_img, show_time); }

	}
	else {
		LOG4CPLUS_WARN(logger, "match score<0.3, is " << match_result.match_score << std::endl;);
	}






	auto end = std::chrono::steady_clock::now();
	auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double t_d = t.count();
	LOG4CPLUS_INFO(logger, "cal_angle_dist用时"<<t_d/1000<<"毫秒");





	log4cplus::deinitialize();
	return 0;
}

