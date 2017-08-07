#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>


using namespace std;

class GNB
{
public:

    vector<string> possible_labels = {"left","keep","right"};
    
    typedef vector<double> features;
    typedef vector<features> featureList;
    
    typedef struct feature{
        featureList samples;
        features sum;
        features std;
        features mean;
				double prob;
    }class_model;
   
    vector<class_model> m_model;
    
    size_t m_feature_size;


    /**
    * Constructor
    */
    GNB();

    /**
    * Destructor
    */
    virtual ~GNB();

    void train ( vector<vector<double> > data, vector<string>  labels );

		string predict ( vector<double> states);

		features calculateFeatures(features &states);

};

#endif
