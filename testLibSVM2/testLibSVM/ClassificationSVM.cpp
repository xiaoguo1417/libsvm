#include "ClassificationSVM.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <time.h>

//����param�Ĺؼ�����gamma��C
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
	param.svm_type = C_SVC;//ȡֵΪǰ���ᵽ��ö�������е�ֵ, c-SVC���������ࣩ
	param.kernel_type = RBF;//ȡֵΪǰ���ᵽ��ö�������е�ֵ, RBF�˺��� : exp(-gamma*|u-v|^2)
	param.degree = 3;//���ڶ���ʽ�˺���
	param.gamma = g;//���ڶ���ʽ���������S�ͺ˺���
	param.coef0 = 0;//���ڶ���ʽ��S�ͺ˺���
	param.nu = 0.5;
	param.cache_size = 100;//�˻����С����MBΪ��λ
	param.C = c;
	param.eps = 1e-5;
	param.shrinking = 1;
	param.probability = 0;//����1����ģ�͵ķֲ�������֪??���ÿ���Ӧ�ĸ���
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
				labels.push_back(std::stoi(dataVal.c_str()));
				sampleNum++;
			}
			else {
				//�ָ��ַ����õ�ð�ź�����
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

//��һ����[-1, 1]
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
		std::string rowData;//һ������
		std::istringstream iss;
		fin.open("scale_params.txt");
		std::getline(fin, rowData);
		iss.clear();
		iss.str(rowData);
		double dataVal;
		int count = 0;
		// ��ʶ�ȡ������ÿһ���е�ÿ����
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

	//�ո�ָ��txt�ļ�����һ��Ϊ����
	std::ifstream fin;
	std::string rowData;//һ������
	std::istringstream iss;
	fin.open(featureFileName);
	//��ȡ�����ж�����ά��
	std::getline(fin, rowData);
	iss.clear();
	iss.str(rowData);
	std::string title;
	while (iss >> title)
	{
		featureDim++;
	}

	//���汣����������
	double dataVal;
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

//��ȡ������Ϣ����
void CRecogSVM::readTrainData(const std::string& featureFileName)
{
	readTxt2(featureFileName);
	svmScale(true);
	//����prob
	prob.l = sampleNum;   //ѵ��������
	prob.x = new svm_node*[sampleNum];  //��������
	prob.y = new double[sampleNum];     //��ǩ����
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

//��vector��ȡ������Ϣ����
void CRecogSVM::readTrainData2(const std::vector<stuFeatures>& trainData)
{
	dataVec.clear();
	labels.clear();
	//��ȡ��dataVec
	featureDim = trainData[0].data.size();
	int sampleNum = trainData.size();//��������
	for (int i = 0;i < sampleNum;i++)
	{
		labels.push_back(trainData[i].id);
		dataVec.push_back(trainData[i].data);
	}
	//��һ��
	svmScale(true);
	//����prob
	prob.l = sampleNum;   //ѵ��������
	prob.x = new svm_node*[sampleNum];  //��������
	prob.y = new double[sampleNum];     //��ǩ����
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

//��һ������Ϊ��������ֵ���ļ����ڶ���ΪҪ�����model�ļ���
void CRecogSVM::train(const std::vector<stuFeatures>& trainData, const std::string& modelFileName)
{
	readTrainData2(trainData);

	double* target = new double[prob.l];
	int logG, logC;
	double bestG, bestC;//��¼��õĲ���ֵ
	int minCount = prob.l;//��¼���������
	std::vector<double>rates;//��¼ÿ����϶�Ӧ����ȷ��
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
				if (target[i] != labels[i])//�޸�0707
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
	//���ÿ�Բ�������Ӧ����
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

	//�������
	/*iter����������
		nu : ��ǰ��Ĳ������� - n nu ��ͬ
		obj : ΪSVM����ת��Ϊ�Ķ��ι滮���õ�����Сֵ
		rho : ��ʾ���ߺ����еĳ�������෴���� - b��
		nSV : ��׼֧����������, �����ڷ���ı߽��ϣ��ɳڱ�������0���ʸ�����ϵ�� 0 = < ai < C
		nBSV : �߽��֧����������, ���ڷ���ı߽��ϣ��ɳڱ�������0����������ϵ�� ai = C
		Accuracy : Ԥ������׼ȷ��, Ϊ3 * 1����������һά����׼ȷ�ʣ���ʾΪ�ٷֱȣ���95�����Ⱦ���95��*/

	std::cout << "start training" << std::endl;
	svm_model *svmModel = svm_train(&prob, &param);

	std::cout << "save model" << std::endl;
	//std::string modelFileName = "svm_model.txt";//Ĭ�ϱ���λ�ã������û�ָ��
	svm_save_model(modelFileName.c_str(), svmModel);
	std::cout << "done!" << std::endl;

	delete target;
	delete prob.x;
	delete prob.y;
	svm_destroy_param(&param);
}

int CRecogSVM::predict(const stuFeatures& feature, const std::string& modelFileName)
{
	//��ȡ�����ļ��е�����������ģ��
	//std::string modelFileName = "svm_model.txt";//��Ĭ�ϱ���λ�ö�ȡmodel�ļ�
	svm_model *model = svm_load_model(modelFileName.c_str());
	
	featureDim = feature.data.size();
	dataVec.clear();
	dataVec.push_back(feature.data);
	svmScale(false);

	//��vector�й���prob
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
