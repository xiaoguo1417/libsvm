#pragma once
#include "svm.h"
#include <vector>
#include <iostream>
#include "data.h"

class CRecogSVM
{
public:
	CRecogSVM();
	~CRecogSVM();
	void train(const std::vector<stuFeatures>& trainData, const std::string& modelFileName);
	int predict(const stuFeatures &feature, const std::string& modelFileName);

private:
	void setParam(double c, double g);
	void readTrainData(const std::string& featureFileName);//��txt�ж�ȡ����
	void readTrainData2(const std::vector<stuFeatures>& trainData);
	void readTxt(const std::string& featureFileName);
	void readTxt2(const std::string& featureFileName);
	void svmScale(bool train_model);

private:
	svm_parameter param;
	svm_problem prob;//all the data for train
	std::vector<std::vector<double>>dataVec;//���ݵ�����
	std::vector<int>labels;//ÿ�����ݵķ����ǩ
	int sampleNum;//��������
	int featureDim;//����������
	int sampleTypes;//��������
	//bool* judgeRight;

};

