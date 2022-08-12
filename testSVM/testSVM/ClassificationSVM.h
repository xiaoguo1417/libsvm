#pragma once
#include "svm.h"
#include <vector>
#include <list>
#include <iostream>

//https://www.cnblogs.com/-ldzwzj-1991/p/5897199.html
class ClassificationSVM
{
public:
	ClassificationSVM();
	~ClassificationSVM();
	void train(const std::string& featureFileName, const std::string& modelFileName);
	void predict(const std::string& featureFileName, const std::string& modelFileName);

private:
	void setParam(double c, double g);
	void readTrainData(const std::string& featureFileName);
	void readTxt(const std::string& featureFileName);
	void readTxt2(const std::string& featureFileName);
	void svmScale(bool train_model);

private:
	svm_parameter param;
	svm_problem prob;//all the data for train
	std::vector<std::vector<double>>dataVec;//数据的特征
	std::vector<double>labels;//每个数据的分类标签
	int sampleNum;//样本数量
	int featureDim;//特征的数量
	int sampleTypes;//样本种类
	//bool* judgeRight;

};

