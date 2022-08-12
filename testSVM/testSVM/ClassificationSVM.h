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
	std::vector<std::vector<double>>dataVec;//���ݵ�����
	std::vector<double>labels;//ÿ�����ݵķ����ǩ
	int sampleNum;//��������
	int featureDim;//����������
	int sampleTypes;//��������
	//bool* judgeRight;

};

