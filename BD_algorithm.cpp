#include "BD_algorithm.h"

#include "logger_ini.h"


// 在给定图像上画点
void draw_points(cv::Mat* points_image, const std::vector<cv::Point>& pts,
	const cv::Scalar& ptcolor) {
	int line_width = ((*points_image).size().height +
		(*points_image).size().width) / 1800 + 1;
	for (int i = 0; i < pts.size(); i++) {
		cv::circle(*points_image, pts[i], line_width + 1, ptcolor, -1, 8);
	}
}

// 画一个polygon
// mode 0 : 只画点
//      1 : 画点和线
void draw_polygon(cv::Mat* poly_image, const std::vector<cv::Point>& polygon_pts,
	int mode, const cv::Scalar ptcolor) {
	int line_width = ((*poly_image).size().height + (*poly_image).size().width) / 1800 + 1;

	cv::circle(*poly_image, polygon_pts[0], line_width + 1, ptcolor, -1, 8);
	if (mode == 1) {
		cv::line(*poly_image, polygon_pts[polygon_pts.size() - 1], polygon_pts[0],
			cv::Scalar(255, 0, 255), line_width);
	}
	cv::Point start_pt, end_pt;
	for (int i = 0; i < polygon_pts.size() - 1; i++) {
		start_pt = polygon_pts[size_t(i)];
		end_pt = polygon_pts[size_t(i) + 1];
		cv::circle(*poly_image, end_pt, line_width + 1, ptcolor, -1, 8);
		if (mode == 1) {
			cv::line(*poly_image, start_pt, end_pt, cv::Scalar(255, 0, 255), line_width);
		}
	}
}

// 获取多边形逼近点集
// binary: input
// poly_points : 点集
// poly_points : 多边形图像
// epsilon : 多边形逼近参数，越小越接近实际轮廓
// draw_poly : 是否画出结果图像
void get_polypts(const cv::Mat& binary, std::vector<cv::Point>* template_points,
	cv::Mat* poly_image, const float& epsilon, const bool& draw_poly) {
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary, contours, hierarchy, cv::RETR_TREE,
		cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
	*poly_image = cv::Mat::zeros(binary.size(), CV_8UC3);
	int line_width = (binary.size().height + binary.size().width) / 1800 + 1;
	if (draw_poly) {
		cv::drawContours(*poly_image, contours, -1, cv::Scalar(0, 255, 0),
			line_width, 8, hierarchy);
	}
	std::vector<cv::Point> polygon_pts;
	cv::Point start_pt, end_pt;

	for (auto cnt : contours) {
		cv::approxPolyDP(cnt, polygon_pts, epsilon, true);
		if (draw_poly) {
			draw_polygon(poly_image, polygon_pts, 1, cv::Scalar(0, 0, 255) );
		}
		(*template_points).insert((*template_points).end(),
			polygon_pts.begin(), polygon_pts.end());
	}
}

// 创建模板
// morph_img 输入滤波后的二值图像
// poly_img 关键点图像
void create_template(const cv::Mat& morph_img, cv::Mat* poly_img,
	std::string save_path) {
	*poly_img = cv::Mat::zeros(morph_img.size(), CV_8UC3);
	std::vector<cv::Point> pts, anchor_pts;
	get_polypts(morph_img, &pts, poly_img, 3, false);
	get_polypts(morph_img, &anchor_pts, poly_img, 99, false);
	draw_polygon(poly_img, pts, 0);
	draw_polygon(poly_img, anchor_pts, 0, cv::Scalar(0, 0, 255));

	//std::cout << "anchor_pts = " << anchor_pts << std::endl;
	//std::cout << "pts = " << pts << std::endl;
	std::vector<int> indices;
	for (int i = 0; i < anchor_pts.size(); i++) {
		auto iter = std::find(pts.begin(), pts.end(), anchor_pts[i]);
		if (iter != pts.end()) {
			auto index = std::distance(pts.begin(), iter);
			indices.push_back(index);
		}
	}
	cv::FileStorage fs;
	fs.open("Template.xml", cv::FileStorage::WRITE);
	fs << "points" << pts << "anchor_indices"<< indices;
	LOG4CPLUS_INFO(logger,
		"Create polygon template sucess! See data in->Template.xml" << std::endl;);
	fs.release();
}

// 读取模板
void read_tmplt(const std::string& file_path, TmpltPoly* tmplt) {
	cv::FileStorage fs;
	fs.open(file_path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		//std::cout << "can't open file " << file_path << std::endl;
		LOG4CPLUS_INFO(logger, "can't open tmpltPoly file " <<
			file_path.data() << std::endl;);
	}
	else {
		fs["points"] >> tmplt->tmplt_pts;
		fs["anchor_indices"] >> tmplt->anchor_indices;
	}
	fs.release();
}

// Euclidean distance between two points in an image
float EuclideanDist(const cv::Point2f& pt0, const cv::Point2f& pt1) {
	return std::sqrt(std::pow((pt0.x - pt1.x), 2) + std::pow((pt0.y - pt1.y), 2));
}
// angle in image
float Angle2P(const cv::Point2f& pt0, const cv::Point2f& pt1) {
	float ang_rad = atan2(pt1.y - pt0.y, pt1.x - pt0.x);
	float angle = ang_rad * 180.0 / CV_PI;
	return angle;
}

// 计算各点到锚点的角度和距离
// pts : 关键点集
// anchor_index : 锚点索引
// return : sorted = {ang_sorted, dist_sorted} 按角度升序
void cal_angle_dist(const std::vector<cv::Point>& pts,
	const int& anchor_index, AngDist* sorted) {
	cv::Point anchor = pts[anchor_index];
	std::vector<float> angle_vec, dist_vec;
	for (int i = 0; i < pts.size(); i++) {
		angle_vec.push_back(Angle2P(anchor, pts[size_t(i)]));
		dist_vec.push_back(EuclideanDist(anchor, pts[size_t(i)]));
	}
	cv::Mat sort_ind;
	cv::sortIdx(cv::Mat(angle_vec), sort_ind,
		cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
	std::vector<float> angle_sorted, dist_sorted;
	for (int i = 0; i < pts.size(); i++) {
		angle_sorted.push_back(angle_vec[sort_ind.at<int>(i, 0)]);
		dist_sorted.push_back(dist_vec[sort_ind.at<int>(i, 0)]);
		//std::cout << dist_vec[sort_ind.at<int>(i, 0)] << std::endl;
	}
	sorted->angles = angle_sorted;
	sorted->dists = dist_sorted;
}

std::vector<std::vector<int> > find_match(const std::vector<float>& tmplt_vec,
	const std::vector<float>& vec, const float& error) {
	std::vector<int> match_ind_tmplt, match_ind;
	cv::Mat err_mat;
	double max, min;
	cv::Point min_loc, max_loc;
	for (int i = 0; i < tmplt_vec.size(); i++) {
		err_mat = cv::abs(cv::Mat(vec) - tmplt_vec[i]);
		//std::cout << "cv::Mat(vec)" << cv::Mat(vec) << std::endl;
		//std::cout << "tmplt_vec[i]" << tmplt_vec[i] << std::endl;
		//std::cout << "err_mat=\n" << err_mat << std::endl;
		cv::minMaxLoc(err_mat, &min, &max, &min_loc, &max_loc);
		//std::cout << min << std::endl;
		//std::cout << min_loc << std::endl;
		if (min < error) {
			match_ind_tmplt.push_back(i);
			match_ind.push_back(min_loc.y);
		}
	}
	return { match_ind_tmplt, match_ind };
}

// vector拼接。angles在索引位置的值为0（整个vec减此值），
// 索引位置前面的值加360，放到vec后边
// dists只做拼接不做加减，保证与angles一一对应。
void angle2zero(const AngDist& sorted,
	AngDist* j2zero, const int& index) {
	std::vector<float> angle_pre, angle_back, dist_pre, dist_back;
	std::vector<float> angle_vec= sorted.angles;
	float j_angle = angle_vec[index];
	for (int i = 0; i < angle_vec.size(); i++) {
		angle_vec[i] -= j_angle;
	}
	angle_pre.assign(angle_vec.begin(), angle_vec.begin() + index);
	angle_back.assign(angle_vec.begin() + index, angle_vec.end());
	dist_pre.assign(sorted.dists.begin(), sorted.dists.begin() + index);
	dist_back.assign(sorted.dists.begin() + index, sorted.dists.end());

	for (int i = 0; i < angle_pre.size();i++) {
		angle_pre[i] += 360;
	}
	j2zero->angles = {};
	j2zero->angles.insert(j2zero->angles.end(), angle_back.begin(), angle_back.end());
	j2zero->angles.insert(j2zero->angles.end(), angle_pre.begin(), angle_pre.end());
	j2zero->dists = {};
	j2zero->dists.insert(j2zero->dists.end(), dist_back.begin(), dist_back.end());
	j2zero->dists.insert(j2zero->dists.end(), dist_pre.begin(), dist_pre.end());
}

// 在vec中找对应索引（ind），生成新向量sub_vec
void get_subvec_ind(const std::vector<float>& vec, const std::vector<int>& ind, 
	std::vector<float>* sub_vec) {
	sub_vec->clear();
	for (int i = 0; i < ind.size(); i++) {
		sub_vec->push_back(vec[ind[i]]);
	}
}

// 对于模板tem的某个锚点，计算pts的最佳匹配，返回匹配分数，以及匹配锚点、旋转角度。
Match_Result match_one_anchor(const TmpltPoly& tem, const int& tem_anchor_ind,
	const std::vector<cv::Point>& pts) {
	AngDist ad_tmplt, ad_pts, tmplt_02zero, j2zero;
	cal_angle_dist(tem.tmplt_pts, tem.anchor_indices[tem_anchor_ind], &ad_tmplt);
	std::vector<int> matchind_a_tpl, matchind_a, matchind_d_tpl, matchind_d;
	std::vector<float> dist_tpl_sub, dist_sub;
	angle2zero(ad_tmplt, &tmplt_02zero, 0);
	int match_i=0, match_j=0, max_match_num = 0;
	float match_angle = 0.0, j_angle = 0.0;
	// 以pts中第i个点为锚点
	for (int i = 0; i < pts.size(); i++) {
		cal_angle_dist(pts, i, &ad_pts);
		// 以第j个点到锚点的角度为0度
		for (int j = 0; j < tmplt_02zero.angles.size(); j++) {
			angle2zero(ad_pts, &j2zero, j);
			auto match_re = find_match(tmplt_02zero.angles, j2zero.angles, 0.4);
			matchind_a_tpl = match_re[0];
			matchind_a = match_re[1];
			if (matchind_a.size() > j2zero.angles.size() * 0.25) {
				get_subvec_ind(tmplt_02zero.dists, matchind_a_tpl, &dist_tpl_sub);
				get_subvec_ind(j2zero.dists, matchind_a, &dist_sub);

				auto match_dist_re = find_match(dist_tpl_sub, dist_sub, 4);
				int match_num = match_dist_re[0].size();
				if (match_num > max_match_num) {
					max_match_num = match_num;
					match_i = i;
					match_j = j;
					j_angle = ad_pts.angles[j];
				}
			}
		}
	}
	//float match_score = float(max_match_num) / float(tem.tmplt_pts.size());
	//std::cout << "match_score= " << match_score << std::endl;
	Match_Result re;
	re.match_score = float(max_match_num) / float(tem.tmplt_pts.size());
	re.match_anchor= pts[match_i];
	cv::Point tem_anchor = tem.tmplt_pts[tem.anchor_indices[tem_anchor_ind]];
	re.shift = re.match_anchor - tem_anchor;
	re.match_angle = ad_tmplt.angles[0] - j_angle;
	return re;
}

// result_img: 结果图，3通道
void tmplt_transform(const TmpltPoly& tem, const Match_Result& match_re,
	cv::Mat* result_img, std::vector<std::vector<float>>* result_info) {

	cv::Point anchor_pt = match_re.match_anchor;
	float match_angle = match_re.match_angle;

	// 模板点集投射到待测图像，构造新的点集
	auto pts_new = tem.tmplt_pts;
	for (int i = 0; i < pts_new.size(); i++) {
		pts_new[i] += match_re.shift;
	}
	// 求旋转矩阵
	cv::Mat M = cv::getRotationMatrix2D(anchor_pt, match_angle, 1);
	cv::Mat M_homo;
	cv::Mat zzo = (cv::Mat_<double>(1, 3) << 0, 0, 1);
	cv::vconcat(M, zzo, M_homo);
	// pt_mat:点集vector转mat,r_pts_new:旋转后的点集
	cv::Mat pt_mat(3, 1, CV_64FC1), r_pt_mat;
	std::vector<cv::Point2f> r_pts_new;
	std::vector<cv::Point> r_pts_new_2i;
	float r_pt_x, r_pt_y;
	for (int i = 0; i < pts_new.size(); i++) {
		pt_mat.at<double>(0, 0) = pts_new[i].x;
		pt_mat.at<double>(1, 0) = pts_new[i].y;
		pt_mat.at<double>(2, 0) = 1.0;
		//cv::hconcat(pts_new_mat, pt_mat, pts_new_mat);
		r_pt_mat = M_homo * pt_mat;
		r_pt_x = r_pt_mat.at<double>(0, 0);
		r_pt_y = r_pt_mat.at<double>(1, 0);
		r_pts_new.push_back(cv::Point2f(r_pt_x, r_pt_y));
		r_pts_new_2i.push_back(cv::Point(r_pt_x, r_pt_y));
	}
	// 画结果图
	draw_points(result_img, r_pts_new_2i, cv::Scalar(0, 255, 0));
	// 最小外接矩
	cv::RotatedRect rRect =	cv::minAreaRect(r_pts_new_2i);
	// 画矩形
	cv::Point2f rect_points[4];
	rRect.points(rect_points);
	for (int j = 0; j < 4; j++) {
		line(*result_img, rect_points[j],
			rect_points[(j + 1) % 4], cv::Scalar(0, 255, 0),
			1, 8);
	}
	float len = std::max(rRect.size.height, rRect.size.width);
	float wid = std::min(rRect.size.height, rRect.size.width);
	float ang = (rRect.size.height <= rRect.size.width) ?
		rRect.angle : rRect.angle + 90;
	std::vector<float> rect_info = {};
	rect_info.push_back(rRect.center.x);
	rect_info.push_back(rRect.center.y);
	rect_info.push_back(ang);
	rect_info.push_back(wid);
	rect_info.push_back(len);
	rect_info.push_back(match_re.match_score);
	*result_info = {};
	(*result_info).push_back(std::move(rect_info));
}