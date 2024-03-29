#include "ClassificationSVM.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <time.h>

//传入param的关键参数gamma和C
CRecogSVM::CRecogSVM()
{
	featureDim = -1;
	sampleNum = 0;
	sampleTypes = 2;
	setParam(1, 1);
}


CRecogSVM::~CRecogSVM()
{
}

void CRecogSVM::setParam(double c, double g)
{
	param.svm_type = C_SVC;//取值为前面提到的枚举类型中的值, c-SVC（多类别分类）
	param.kernel_type = RBF;//取值为前面提到的枚举类型中的值, RBF核函数 : exp(-gamma*|u-v|^2)
	param.degree = 3;//用于多项式核函数
	param.gamma = g;//用于多项式、径向基、S型核函数
	param.coef0 = 0;//用于多项式和S型核函数
	param.nu = 0.5;
	param.cache_size = 100;//核缓存大小，以MB为单位
	param.C = c;
	param.eps = 1e-5;
	param.shrinking = 1;
	param.probability = 0;//等于1代表模型的分布概率已知??输出每类对应的概率
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
}

void CRecogSVM::readTxt2(const std::string& featureFileName)
{
	dataVec.clear();
	labels.clear();
	featureDim = -1;
	sampleNum = 0;

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
				labels.push_back(std::stoi(dataVal.c_str()));
				sampleNum++;
			}
			else {
				//分割字符串得到冒号后数据
				for (int k = 0;k < dataVal.size();k++)
				{
					if (dataVal[k] == ':') {
						dataVal = dataVal.substr(k+1);
						break;
					}
				}
				rowDataVec.push_back(atof(dataVal.c_str()));
			}
		}
		dataVec.push_back(rowDataVec);
	}
	featureDim = dataVec[0].size();
}

//归一化到[-1, 1]
void CRecogSVM::svmScale(bool train_model)
{
	double *minVals = new double[featureDim];
	double *maxVals = new double[featureDim];

	if (train_model) {
		for (int i = 0;i < featureDim;i++)
		{
			minVals[i] = dataVec[0][i];
			maxVals[i] = dataVec[0][i];
		}

		for (int i = 0;i < dataVec.size();i++)
		{
			for (int j = 0;j < dataVec[i].size();j++)
			{
				if (dataVec[i][j] < minVals[j])
					minVals[j] = dataVec[i][j];
				if (dataVec[i][j] > maxVals[j])
					maxVals[j] = dataVec[i][j];
			}
		}

		std::ofstream out("scale_params.txt");
		for (int i = 0;i < featureDim;i++)
		{
			out << minVals[i] << " ";
		}
		out << std::endl;
		for (int i = 0;i < featureDim;i++)
		{
			out << maxVals[i] << " ";
		}
	}
	else {
		std::ifstream fin;
		std::string rowData;//一行内容
		std::istringstream iss;
		fin.open("scale_params.txt");
		std::getline(fin, rowData);
		iss.clear();
		iss.str(rowData);
		double dataVal;
		int count = 0;
		// 逐词读取，遍历每一行中的每个词
		while (iss >> dataVal)
		{
			minVals[count] = dataVal;
			count++;
		}
		count = 0;
		std::getline(fin, rowData);
		iss.clear();
		iss.str(rowData);
		while (iss >> dataVal)
		{
			maxVals[count] = dataVal;
			count++;
		}
	}

	for (int i = 0;i < dataVec.size();i++)
	{
		for (int j = 0;j < dataVec[i].size() && j < featureDim;j++)
		{
			dataVec[i][j] = -1 + 2 * (dataVec[i][j] - minVals[j]) / (maxVals[j] - minVals[j]);
		}
	}


	delete minVals;
	delete maxVals;
}

void CRecogSVM::readTxt(const std::string& featureFileName)
{
	dataVec.clear();
	labels.clear();
	featureDim = -1;
	sampleNum = 0;

	//空格分割的txt文件，第一行为标题
	std::ifstream fin;
	std::string rowData;//一行内容
	std::istringstream iss;
	fin.open(featureFileName);
	//读取首行判断特征维度
	std::getline(fin, rowData);
	iss.clear();
	iss.str(rowData);
	std::string title;
	while (iss >> title)
	{
		featureDim++;
	}

	//下面保存特征数据
	double dataVal;
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
				labels.push_back((int)dataVal);
				sampleNum++;
			}
			else {
				rowDataVec.push_back(dataVal);
			}
		}
		dataVec.push_back(rowDataVec);
	}

}

//读取特征信息保存
void CRecogSVM::readTrainData(const std::string& featureFileName)
{
	readTxt2(featureFileName);
	svmScale(true);
	//设置prob
	prob.l = sampleNum;   //训练样本数
	prob.x = new svm_node*[sampleNum];  //特征矩阵
	prob.y = new double[sampleNum];     //标签矩阵
	for (int i = 0; i < sampleNum; ++i)
	{
		prob.x[i] = new svm_node[featureDim + 1]; //
		for (int j = 0; j < featureDim; ++j)
		{
			prob.x[i][j].index = j + 1;
			prob.x[i][j].value = dataVec[i][j];
		}
		prob.x[i][featureDim].index = -1;
		prob.y[i] = labels[i];
	}

}

//从vector读取特征信息保存
void CRecogSVM::readTrainData2(const std::vector<stuFeatures>& trainData)
{
	dataVec.clear();
	labels.clear();
	//读取到dataVec
	featureDim = trainData[0].data.size();
	int sampleNum = trainData.size();//样本数量
	for (int i = 0;i < sampleNum;i++)
	{
		labels.push_back(trainData[i].id);
		dataVec.push_back(trainData[i].data);
	}
	//归一化
	svmScale(true);
	//设置prob
	prob.l = sampleNum;   //训练样本数
	prob.x = new svm_node*[sampleNum];  //特征矩阵
	prob.y = new double[sampleNum];     //标签矩阵
	for (int i = 0; i < sampleNum; ++i)
	{
		prob.x[i] = new svm_node[featureDim + 1]; //
		for (int j = 0; j < featureDim; ++j)
		{
			prob.x[i][j].index = j + 1;
			prob.x[i][j].value = dataVec[i][j];
		}
		prob.x[i][featureDim].index = -1;
		prob.y[i] = labels[i];
	}
}

//第一个参数为输入特征值的文件，第二个为要保存的model文件名
void CRecogSVM::train(const std::vector<stuFeatures>& trainData, const std::string& modelFileName)
{
	readTrainData2(trainData);

	double* target = new double[prob.l];
	int logG, logC;
	double bestG, bestC;//记录最好的参数值
	int minCount = prob.l;//记录错误的数量
	std::vector<double>rates;//记录每次组合对应的正确率
	for (logC = -5;logC <= 15;logC += 2)
		for (logG = -15;logG <= 3;logG += 2)
		{
			double c = pow(2, logC);
			double g = pow(2, logG);
			setParam(c, g);
			svm_cross_validation(&prob, &param, 5, target);
			int count = 0;
			for (int i = 0;i < prob.l;i++)
			{
				if (target[i] != labels[i])//修改0707
					count++;
			}
			if (count < minCount)
			{
				minCount = count;
				bestC = c;
				bestG = g;
			}
			rates.push_back(1.0*(prob.l - count) / prob.l * 100);
		}
	setParam(bestC, bestG);
	//输出每对参数及对应概率
	std::ofstream out("E:\\LIB_SVM\\libsvm\\tools\\rates.txt");
	int count1 = 0;
	for (logC = -5;logC <= 15;logC += 2)
	{
		for (logG = -15;logG <= 3;logG += 2)
		{
			std::string s1 = "log2c=";
			s1 += std::to_string(logC);
			std::string s2 = "log2g=";
			s2 += std::to_string(logG);
			std::string s3 = "rate=";
			s3 += std::to_string(rates[count1]);
			count1++;

			out << s1 << " " << s2 << " " << s3 << std::endl;
		}
	}
	out.close();

	//输出参数
	/*iter：迭代次数
		nu : 与前面的操作参数 - n nu 相同
		obj : 为SVM问题转换为的二次规划求解得到的最小值
		rho : 表示决策函数中的常数项的相反数（ - b）
		nSV : 标准支持向量个数, 就是在分类的边界上，松弛变量等于0，朗格朗日系数 0 = < ai < C
		nBSV : 边界的支持向量个数, 不在分类的边界上，松弛变量大于0，拉格郎日系数 ai = C
		Accuracy : 预测结果的准确率, 为3 * 1的向量，第一维就是准确率，显示为百分比，如95，精度就是95％*/

	std::cout << "start training" << std::endl;
	svm_model *svmModel = svm_train(&prob, &param);

	std::cout << "save model" << std::endl;
	//std::string modelFileName = "svm_model.txt";//默认保存位置，不再用户指定
	svm_save_model(modelFileName.c_str(), svmModel);
	std::cout << "done!" << std::endl;

	delete target;
	delete prob.x;
	delete prob.y;
	svm_destroy_param(&param);
}

int CRecogSVM::predict(const stuFeatures& feature, const std::string& modelFileName)
{
	//读取特征文件中的特征及保存模型
	//std::string modelFileName = "svm_model.txt";//从默认保存位置读取model文件
	svm_model *model = svm_load_model(modelFileName.c_str());
	
	featureDim = feature.data.size();
	dataVec.clear();
	dataVec.push_back(feature.data);
	svmScale(false);

	//从vector中构造prob
	int resultLabel2 = -1;
	for (int i = 0;i < dataVec.size();i++)
	{
		svm_node *sample = new svm_node[featureDim + 1];
		for (int j = 0; j < featureDim; ++j)
		{
			sample[j].index = j + 1;
			sample[j].value = dataVec[i][j];
		}
		sample[featureDim].index = -1;

		//double *probresut = new double[sampleTypes];
		//double resultLabel = svm_predict_probability(model, sample, probresut);
		//For a classification model, the predicted class for x is returned.
		resultLabel2 = round(svm_predict(model, sample));
		delete sample;
	}

	svm_free_and_destroy_model(&model);

	return resultLabel2;
}
