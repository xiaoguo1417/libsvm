#include <opencv2\opencv.hpp>
#include "svm.h"
#include <string.h>
#include<iostream>
#include "ClassificationSVM.h"

using namespace std;

int main() {
	ClassificationSVM testSVM;

	testSVM.train("E:\\LIB_SVM\\libsvm\\source_txt\\vowel.txt", "E:\\LIB_SVM\\libsvm\\source_txt\\vowel_model_p.txt");
	//testSVM.predict("E:\\LIB_SVM\\libsvm\\source_txt\\vowel_test.txt", "E:\\LIB_SVM\\libsvm\\source_txt\\vowel_model_p.txt");

	return 0;
}