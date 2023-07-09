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
	//��ٷ����߶Ա���֤
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

//�����ã���txt����vector<Features>
std::vector<stuFeatures> loadData(const std::string& featureFileName)
{
	int featureDim = -1;
	int sampleNum = 0;
	std::vector<std::vector<double>>dataVec;//���ݵ�����
	std::vector<double>labels;//ÿ�����ݵķ����ǩ

	//�ٷ���׼��ʽ
	std::ifstream fin;
	std::string rowData;//һ������
	std::istringstream iss;
	fin.open(featureFileName);

	//������������
	std::string dataVal;
	while (std::getline(fin, rowData))
	{
		iss.clear();
		iss.str(rowData);
		bool first = true;
		std::vector<double>rowDataVec;
		// ��ʶ�ȡ������ÿһ���е�ÿ����
		while (iss >> dataVal)
		{
			//��һ��������label�����ʶ
			if (first) {
				first = false;
				labels.push_back(atof(dataVal.c_str()));
				sampleNum++;
			}
			else {
				//�ָ��ַ����õ�ð�ź�����
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