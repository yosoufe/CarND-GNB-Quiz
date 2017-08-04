#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {
	m_model = vector<class_model>(possible_labels.size());
	m_feature_size = 4;
	for( auto model_it= m_model.begin(); model_it < m_model.end(); model_it++ ){
		class_model& model = *model_it;
		model.mean = features(m_feature_size, 0.0);
		model.std = features(m_feature_size, 0.0);
		model.sum = features(m_feature_size, 0.0);
	}
}

GNB::~GNB() {}

void GNB::train ( vector<vector<double>> data, vector<string> labels ) {

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d,
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
	if (data.size() != labels.size() || data.size() == 0){
		std::cout << "Incompatible data size" << std::endl;
		std::exit(0);
	}
	
	int i = 0;
	for (auto data_it = data.begin() ; data_it < data.end(); data_it++, i++){
		features& sample = *data_it;
		int j = 0;
		for (auto class_it = possible_labels.begin(); class_it < possible_labels.end(); class_it++, j++){
			string& label = *class_it;
			if (label.compare(labels[i])==0){
				m_model[j].samples.push_back(sample);
				for(size_t i_features = 0; i_features < m_feature_size; i_features++){
					m_model[j].sum[i_features] += sample[i_features];
				}
				break;
			}
		}
	}

	// calculate the mean and std of the class samples
	for( auto model_it= m_model.begin(); model_it < m_model.end(); model_it++ ){
		class_model& model = *model_it;
		for(size_t i= 0; i<m_feature_size; i++){
			model.mean[i] = model.sum[i]/model.samples.size();
		}
		for(auto sample_it=model.samples.begin(); sample_it<model.samples.end();sample_it++){
			features& feats = *sample_it;
			int i = 0;
			for(auto feat_it = feats.begin(); feat_it< feats.end();feat_it++, i++){
				double& feat = *feat_it;
				model.std[i] += pow(feat - model.mean[i],2);
			}
		}
		for(size_t i= 0; i<m_feature_size; i++){
			model.std[i] = sqrt(model.std[i]/(model.samples.size()-1));
		}
	}
	
	cout << "Samples sorted in classes" << endl;	
}

string GNB::predict ( vector<double> ) {
	/*
		Once trained, this method is called and expected to return
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/

	return this->possible_labels[1];

}
