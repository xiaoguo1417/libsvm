#pragma once
#include <iostream>
#include <vector>

struct stuFeatures
{
	int id;//分类号，预测时可随意输入
	std::vector<double>data;//特征的数组
};