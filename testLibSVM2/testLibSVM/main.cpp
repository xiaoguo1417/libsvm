#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include "ClassificationSVM.h"
#include "data.h"
#include <sstream>
#include <fstream>
#include <time.h>

std::vector<stuFeatures> loadData(const std::string& featureFileName);

int main()
{
	//与官方工具对比验证
	CRecogSVM testSVM;
	std::vector<stuFeatures>trainData = loadData("E:\\LIB_SVM\\libsvm\\windows\\svmguide2.scale");
	testSVM.train(trainData, "E:\\LIB_SVM\\libsvm\\windows\\model2.txt");

	std::ofstream out("E:\\LIB_SVM\\libsvm\\tools\\predict_res.txt");
	int false_cnt = 0;
	std::vector<stuFeatures>testData = loadData("E:\\LIB_SVM\\libsvm\\windows\\svmguide2_test.scale");
	for (int i = 0; i < testData.size();i++) {
		int id = testSVM.predict(testData[i], "E:\\LIB_SVM\\libsvm\\windows\\model2.txt");
		out << id << std::endl;
		if (id != testData[i].id) false_cnt++;
	}
	std::cout << 1.0 * (testData.size() - false_cnt) / testData.size() << std::endl;

	std::cin.get();
	return 0;
}

//测试用，从txt读到vector<Features>
std::vector<stuFeatures> loadData(const std::string& featureFileName)
{
	int featureDim = -1;
	int sampleNum = 0;
	std::vector<std::vector<double>>dataVec;//数据的特征
	std::vector<double>labels;//每个数据的分类标签

	//官方标准样式
	std::ifstream fin;
	std::string rowData;//一行内容
	std::istringstream iss;
	fin.open(featureFileName);

	//保存特征数据
	std::string dataVal;
	while (std::getline(fin, rowData))
	{
		iss.clear();
		iss.str(rowData);
		bool first = true;
		std::vector<double>rowDataVec;
		// 逐词读取，遍历每一行中的每个词
		while (iss >> dataVal)
		{
			//第一个数据是label分类标识
			if (first) {
				first = false;
				labels.push_back(atof(dataVal.c_str()));
				sampleNum++;
			}
			else {
				//分割字符串得到冒号后数据
				for (int k = 0;k < dataVal.size();k++)
				{
					if (dataVal[k] == ':') {
						dataVal = dataVal.substr(k + 1);
						break;
					}
				}
				rowDataVec.push_back(atof(dataVal.c_str()));
			}
		}
		dataVec.push_back(rowDataVec);
	}
	featureDim = dataVec[0].size();

	std::vector<stuFeatures>res;
	for (int i = 0;i < sampleNum;i++)
	{
		stuFeatures feature;
		feature.id = labels[i];
		feature.data = dataVec[i];
		res.push_back(feature);
	}

	return res;
}