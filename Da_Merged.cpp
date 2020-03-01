//Stage 1: merge TKDE and KDD code from Zhe Jiang by using macro <--- done
//Stage 2: isolate out individual functions on each "Node"
//Stage 3: implement parallel algo

#define DEBUG //whether to check if probability values are valid (there are additional overheads, should comment this line after testing for reporting experimental results)
//#define TREE_REPORT //whether to report tree depth statistics and chain number
//#define INCLUDE_M //KDD18: comment,    TKDE: enable

int THREADS = 1;// number of threads
int CHUNK = 500;// how many elements per access in OMP parallel-for

#define WAIT_TIME_WHEN_IDLE 100

//##########################

#define Dim 3 //input data dimension
#define cNum 2
#define _USE_MATH_DEFINES
#define LOGZERO -INFINITY

#include<queue>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<numeric>
#include<vector>
#include<string>
#include <chrono>
#include<ctime>
#include<cmath>
#include<limits>
#include <cstdio>

#include <omp.h>

#include <set>
#include <map>

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <unistd.h>

#include <queue>
#include "conque.h"

//using  millisecond = chrono::milliseconds;
//using get_time = chrono::steady_clock;

using namespace std;

struct Node {
	float elevation;
	int nodeIndex; // original index from raster, index in data.allNodes
	int nodeChainID;
	struct Node* next;
	vector<Node*> parents;
	double curGain;
	bool visited = false;
	Node() {
		nodeChainID = -1;
		next = NULL;
	}
	Node(float value, int nodeindex) {
		elevation = value;
		nodeIndex = nodeindex;
		nodeChainID = -1;
		next = NULL;
	}

	//@@@@@@ Da Yan: added to track dependency of leaf2root msg propagation
	int counter = 0; //need to be cleared after use in each iteration
	mutex lock; //lock of "counter" above

	int add_child_counter() //returns -1 if child not ready (or current node is root), or the child's id for continued processing towards the root
	{
		int result = -1; //-1 means parent task is not ready, get the next task from task queue
		if(next)
		{
			next->lock.lock();
			next->counter++;
			if(next->counter == next->parents.size()) //all parents have been processed in leaf2root pass
			{
				result = next->nodeIndex; //return the next task node for processing by the current thread
				next->counter = 0; //restore the counter for use in the next iteration
			}
			next->lock.unlock();
		}
		return result;
	}

};


struct sParameter {
	double Mu[cNum][Dim] = { {0.0} }; // True value, Mu, Sigma are related to p(yi=0|X,theta) , p(yi=1|X,theta)
	double elnMu[cNum][Dim] = { {0.0} }; // eln value
	double Sigma[cNum][Dim][Dim] = { { {1.0} } };

	double Epsilon = 0.01; // Epsilon: p(zi=0|zpi=1) = Epsilon, is related to p(zi=0, zpi=1|X,theta}, p(zi=1, zpi=1|X,theta) when i has parents
	double Pi = 0.5; // Pi: p(zi = 1) = Pi, is related to p(zi=0|X,theta}, p(zi =1|X,theta) when i has no parents

#ifdef INCLUDE_M
	double M = 0.4; // M: M = y0GivenZ1, is related to p(yi=0,zi=1|X,theta), p(yi=1, zi=1|X,theta)
#endif

	double elnPy_z[cNum][cNum]; //transition probabilities
	double elnPz[cNum];
	double elnPz_zpn[cNum][cNum];

	int nonEmptySize = -1; //set a wrong default value, so it will report error if it's not initialized properly
	double THRESHOLD = 0.05;
	int maxIteratTimes = 30;
	int ROW = -1;
	int COLUMN = -1;
	int allPixelSize = -1;
};
//struct sParameter {
//	int Dim;
//	int cNum
//	double **Mu; // True value, Mu, Sigma are related to p(yi=0|X,theta) , p(yi=1|X,theta)
//	double **elnMu; // eln value
//	double **Sigma;
//
//	double Epsilon = 0.01; // Epsilon: p(zi=0|zpi=1) = Epsilon, is related to p(zi=0, zpi=1|X,theta}, p(zi=1, zpi=1|X,theta) when i has parents
//	double Pi = 0.5; // Pi: p(zi = 1) = Pi, is related to p(zi=0|X,theta}, p(zi =1|X,theta) when i has no parents
//					 //double M = 0.4; // M: M = y0GivenZ1, is related to p(yi=0,zi=1|X,theta), p(yi=1, zi=1|X,theta)
//
//	double **elnPy_z; //transition probabilities
//	double *elnPz;
//	double **elnPz_zpn;
//
//	int nonEmptySize = -1; //set a wrong default value, so it will report error if it's not initialized properly
//	double THRESHOLD = 0.05;
//	int maxIteratTimes = 30;
//	int ROW = -1;
//	int COLUMN = -1;
//	int allPixelSize = -1;
//};


struct sData {
	vector<int>index; // convert packed indexes to expanded (original) indexes
	vector<long int>old2newIdx; // convert expanded indexes to packed indexes
	vector<bool>NA;
	vector<float>features; // RGB +..., rowwise, long array
	vector<float>elevationVector;
	vector< pair<float, int> >elevationIndexPair; // here index means expanded index
	vector<Node*>allNodes;
	vector<int>missingIndicator;// 0 means missing data (not NoData), using new index system
};


struct sTree {
	vector<Node*> headPointers;
	vector<Node*> tailPointers;
	vector<int>nodeLevels;
	vector< pair<int, int> >nodeLevelIndexPair; // here index means expanded index
};


struct sInference {
	vector<double>waterLikelihood;  //-0.5(x-mu)T Sigma^-1 (x-mu) for class 1
	vector<double>dryLikelihood;  //-0.5(x-mu)T Sigma^-1 (x-mu) for class 0
	double lnCoefficient[cNum]; //log constanct of Gaussian distribution for two classes

#ifdef INCLUDE_M
	vector<double> lnTop2y;
	vector<double> lnz2top; //from z to y, outgoing of z, before factor node
	vector<double> lnbottom2y; //Vertical incoming message from z to y, after factor node
#endif

	vector<double> lnvi; // Top2z message
	vector<double> lnfi; // from zp to z except zp=NULL, after factor node
	vector<double> lnfo; // product of lnfi and lngi, before factor node
	vector<double> lngi; // Horizontal incoming message from z to zp, after factor node
	vector<double> lngo;

	// Below are all in eln form.
#ifdef INCLUDE_M
	vector<double> marginal_Yn;
#endif
	vector<double> marginal_YnZn; // only zn = 1 is useful to update parameters
	vector<double> marginal_ZnZpn; // only Zpn = 1 is useful to update parameters
	vector<double> marginal_Zn; // Separate from ZnZpn now. Those internal values are just for testing.

	vector<int>chainMaxGainFrontierIndex;
};

struct conMatrix {
	int TT;
	int TF;
	int FF;
	int FT;
};

class cFlood {
private:
	double featuresAbsoluteMin[Dim];
	struct sParameter parameter;
	struct sData data;
	struct sTree tree;
	struct sInference infer;
	vector<int>testIndex;
	vector<int>testLabel;
	vector<int>mappredictions;
	ofstream timeLogger;
	std::string HMTInputLocation;
	std::string HMTDem;
	std::string HMTFeature;
	std::string HMTIndex;
	std::string HMTMissingIndicator;
	std::string HMTPara;
	std::string HMTOutputLocation;
	std::string HMTPrediction;
	std::string HMTParaLog;
	std::string HMTTimeLog;
	std::string HMTTestIndex;
	std::string HMTTestLabel;

	//@@@@@@ added by yanda, they are to be updated in treeConstruct()
	int root_oid; //old index of root node
	vector<int> leaf_oids; //old index of leaves

public:
	void input(int argc, char *argv[]);
	void report();
	void treeConstruct();
	void UpdateTransProb(); //Update P(y|z), P(zn|zpn), P(zn|zpn=empty)
#ifdef INCLUDE_M
	void UpdatePX_Y(); //Update P(x|y) based on parameters
#else
	void UpdatePX_Z(); //Update P(x|z) based on parameters
#endif
	void UpdateParameters();
	void UpdateMarginalProb();

	void leaf2root(int curIdx);
	void leaf2root_thread(int curIdx);
	bool get_and_process_l2r_tasks(mutex & q_mtx, queue<int> & task_queue, int batch_size);
	void thread_run_l2r(mutex & q_mtx, queue<int> & task_queue, atomic<bool> & global_end_label, size_t & global_num_idle, mutex & mtx_go, condition_variable & cv_go, bool & ready_go);
	void Leaf2Root_MsgProp();

	void root2leaf(int curIdx);
	void root2leaf_thread(int curIdx, conque<int> & task_queue);
	bool get_and_process_r2l_tasks(conque<int> & task_queue, int batch_size);
	void thread_run_r2l(conque<int> & task_queue, atomic<bool> & global_end_label, size_t & global_num_idle, mutex & mtx_go, condition_variable & cv_go, bool & ready_go);
	void Root2Leaf_MsgProp();

	void learning();
	void inference();
	void output();
	struct conMatrix getConfusionMatrix();
	void updateMapPrediction();

};


void getCofactor(double mat[Dim][Dim], double temp[Dim][Dim], int p, int q, int n) {
	int i = 0, j = 0;
	// Looping for each element of the matrix
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			//  Copying into temporary matrix only those element
			//  which are not in given row and column
			if (row != p && col != q) {
				temp[i][j++] = mat[row][col];

				// Row is filled, so increase row index and
				// reset col index
				if (j == n - 1) {
					j = 0;
					i++;
				}
			}
		}
	}
}

//dynamic memory allocation,dimensional two dimension array
/* Recursive function for finding determinant of matrix.
n is current dimension of mat[][]. */
double determinant(double mat[Dim][Dim], int n) {
	double D = 0; // Initialize result

				  //  Base case : if matrix contains single element
	if (n == 1)
		return mat[0][0];

	double temp[Dim][Dim]; // To store cofactors
	int sign = 1;  // To store sign multiplier

				   // Iterate for each element of first row
	for (int f = 0; f < n; f++) {
		// Getting Cofactor of mat[0][f]
		getCofactor(mat, temp, 0, f, n);
		D += sign * mat[0][f] * determinant(temp, n - 1);

		// terms are to be added with alternate sign
		sign = -sign;
	}
	return D;
}

// Function to get adjoint of A[Dim][Dim] in adj[Dim][Dim].
void adjoint(double A[Dim][Dim], double adj[Dim][Dim]) {
	if (Dim == 1) {
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][]
	int sign = 1;
	double temp[Dim][Dim];

	for (int i = 0; i < Dim; i++) {
		for (int j = 0; j < Dim; j++) {
			// Get cofactor of A[i][j]
			getCofactor(A, temp, i, j, Dim);

			// sign of adj[j][i] positive if sum of row
			// and column indexes is even.
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the
			// transpose of the cofactor matrix
			adj[j][i] = (sign)*(determinant(temp, Dim - 1));
		}
	}
}

// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(double A[Dim][Dim], double inverse[Dim][Dim]) {
	// Find determinant of A[][]

	if (Dim == 1) {
		inverse[0][0] = 1.0 / A[0][0];
		return true;
	}

	double det = determinant(A, Dim);
	if (det == 0) {
		cout << "Singular matrix, can't find its inverse";
		return false;
	}

	// Find adjoint
	double adj[Dim][Dim];
	adjoint(A, adj);

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)"
	for (int i = 0; i < Dim; i++)
		for (int j = 0; j < Dim; j++)
			inverse[i][j] = adj[i][j] / double(det);
	return true;
}

// extended ln functions
double eexp(double x) {
	if (x == LOGZERO) {
		return 0;
	}
	else {
		return exp(x);
	}
}

double eln(double x) {
	if (x == 0) {
		return LOGZERO;
	}
	else if (x > 0) {
		return log(x);
	}
	else {
		cout << "Negative input error " << x << endl;
		int tmp;
		cin >> tmp;
	}
}

double elnsum(double x, double y) {
	if (x == LOGZERO) {
		return y;
	}
	else if (y == LOGZERO) {
		return x;
	}
	else if (x > y) {
		return x + eln(1 + eexp(y - x));
	}
	else {
		return y + eln(1 + eexp(x - y));
	}
}

double elnproduct(double x, double y) {
	if (x == LOGZERO || y == LOGZERO) {
		return LOGZERO;
	}
	else {
		return x + y;
	}
}


void cFlood::input(int argc, char *argv[]) {
	clock_t start_s = clock();

	string HMTInputLocation; // elevation, index and feature file are the same for both texture and notexture situation
	if (argc > 1) {
		ifstream config(argv[1]);
		string line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		if(line.back()=='/') line.push_back('/');
		HMTInputLocation = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTDem = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTFeature = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTIndex = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTMissingIndicator = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTPara = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTTestIndex = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTTestLabel = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		if(line.back()=='/') line.push_back('/');
		HMTOutputLocation = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTPrediction = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		HMTParaLog = line;
		getline(config, line);
		if(line.back()=='\r') line.pop_back();
		if (line != "") {
			HMTTimeLog = line;
		}
		else {
			std::cout << "config file error" << endl;
		}
		config.close();
	}
	else {
		std::cout << "Missing Configuration File!" << endl;
	}

	if (argc > 2) {
		THREADS = atoi(argv[2]);
	}

	cout << "# of threads: " << THREADS << endl;

	std::string timelogFile = HMTOutputLocation + HMTTimeLog;
	timeLogger.open(timelogFile.c_str(), std::ofstream::app);

	ifstream missingIndicatorFile(HMTInputLocation + HMTMissingIndicator);
	ifstream elevationFile(HMTInputLocation + HMTDem);
	ifstream indexFile(HMTInputLocation + HMTIndex);
	ifstream featuresFile(HMTInputLocation + HMTFeature);
	ifstream testIndexFile(HMTInputLocation + HMTTestIndex);
	ifstream testLabelFile(HMTInputLocation + HMTTestLabel);
	ifstream parameterFile(HMTInputLocation + HMTPara);


	if (!parameterFile) {
		std::cout << "Failed to open parameter!" << endl;
	}

	if (!elevationFile) {
		std::cout << "Failed to open elevationFile!" << endl;
	}

	if (!featuresFile) {
		std::cout << "Failed to open featuresFile!" << endl;
	}

	if (!indexFile) {
		std::cout << "Failed to open indexFile!" << endl;
	}

	if (!missingIndicatorFile) {
		std::cout << "Failed to open missingIndicatorFile!" << endl;
	}

	if (!testIndexFile) {
		std::cout << "Failed to open testIndexFile!" << endl;
	}

	if (!testLabelFile) {
		std::cout << "Failed to open testLabelFile!" << endl;
	}

	parameterFile >> parameter.THRESHOLD;
	parameterFile >> parameter.maxIteratTimes;
	parameterFile >> parameter.Epsilon;
	parameterFile >> parameter.Pi;
#ifdef INCLUDE_M
	parameterFile >> parameter.M;
#endif
	parameterFile >> parameter.ROW;
	parameterFile >> parameter.COLUMN;
	parameterFile >> parameter.nonEmptySize;


	int testIdx;
	while (testIndexFile >> testIdx)
	{
		testIndex.push_back(testIdx - 1); //testIdx extracted by R, starting from 1
	}
	testIndexFile.close();

	int testLbl;
	while (testLabelFile >> testLbl)
	{
		testLabel.push_back(testLbl);
	}
	testLabelFile.close();

#ifdef DEBUG
#ifdef INCLUDE_M
	if (parameter.Epsilon > 1 || parameter.Pi > 1 || parameter.M > 1) {
#else
	if (parameter.Epsilon > 1 || parameter.Pi > 1) {
#endif
		cout << "wrong parameter" << endl;
	}
#endif

#ifdef INCLUDE_M
	cout << "Input parameters:" << endl << "Epsilon: " << parameter.Epsilon << " Pi: " << parameter.Pi << " M: " << parameter.M << endl;
#else
	cout << "Input parameters:" << endl << "Epsilon: " << parameter.Epsilon << " Pi: " << parameter.Pi << endl;
#endif
	
	for (int c = 0; c < cNum; c++) {
		for (size_t i = 0; i < Dim; i++) {
			parameterFile >> parameter.Mu[c][i];
		}
	}

	for (int c = 0; c < cNum; c++) {
		for (size_t i = 0; i < Dim; i++) {
			for (size_t j = 0; j < Dim; j++) {
				parameterFile >> parameter.Sigma[c][i][j];
			}
		}
	}

	parameterFile.close();

	parameter.allPixelSize = parameter.ROW * parameter.COLUMN;

	infer.waterLikelihood.resize(parameter.nonEmptySize);
	infer.dryLikelihood.resize(parameter.nonEmptySize);

	data.missingIndicator.resize(parameter.nonEmptySize);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		missingIndicatorFile >> data.missingIndicator.at(i);
	}
	missingIndicatorFile.close();

	data.index.resize(parameter.nonEmptySize);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		indexFile >> data.index.at(i);
	}
	indexFile.close();

	//reverse map
	data.old2newIdx.resize(parameter.allPixelSize);
	std::fill(data.old2newIdx.begin(), data.old2newIdx.end(), -1);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		data.old2newIdx.at(data.index.at(i)) = i;
	}

	data.NA.resize(parameter.allPixelSize);
	std::fill(data.NA.begin(), data.NA.end(), true);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		data.NA.at(data.index.at(i)) = false;
	}

	data.features.resize(parameter.nonEmptySize * Dim);// RGB + ..., rowwise, long array

	for (size_t i = 0; i < parameter.nonEmptySize * Dim; i++) {
		featuresFile >> data.features.at(i);
	}
	featuresFile.close();

    // Calculate features absolute min values, then +1 to avoid 0
	int bandNum = 0;
	for (int i = 0; i < Dim; i++) {
		featuresAbsoluteMin[i] = 0;
	}
	for (int i = 0; i < parameter.nonEmptySize * Dim; i++) {
		bandNum = i % Dim;
		if (data.features.at(i) < featuresAbsoluteMin[bandNum]) {
			featuresAbsoluteMin[bandNum] = data.features.at(i);
		}
	}
	for (int i = 0; i < Dim; i++) {
		featuresAbsoluteMin[i] = fabs(featuresAbsoluteMin[i]) + 1;
	}


	data.elevationVector.resize(parameter.nonEmptySize);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		elevationFile >> data.elevationVector.at(i);
	}
	elevationFile.close();

	data.elevationIndexPair.resize(parameter.nonEmptySize);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		data.elevationIndexPair.at(i) = make_pair(data.elevationVector[i], data.index[i]);
	}
	sort(std::begin(data.elevationIndexPair), std::end(data.elevationIndexPair));

	data.allNodes.resize(parameter.allPixelSize);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) { // only nonempty nodes will be generated
														  //allNodes.at(elevationIndexPair[i].second) = new Node(elevationIndexPair[i].first, elevationIndexPair[i].second);
		data.allNodes.at(data.index.at(i)) = new Node(data.elevationVector[i], data.index.at(i));
	}

	clock_t stop_s = clock();
	std::cout << "Data prepare " << endl << "time: " << (stop_s - start_s) / float(CLOCKS_PER_SEC) << endl << endl;

	auto start = std::chrono::system_clock::now();

	treeConstruct();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	timeLogger << elapsed_seconds.count() << ",";

	//initialize likelihood and log constance in P(x|y);
	//Now we have all parameters and likelihood values needed for inference from initialization; 
	double determinantValue[cNum];
	for (int c = 0; c < cNum; c++) {
		determinantValue[c] = determinant(parameter.Sigma[c], Dim);
	}
	for (int c = 0; c < cNum; c++) {
		infer.lnCoefficient[c] = -0.5 * Dim * log(2 * M_PI) - 0.5 * log(fabs(determinantValue[c])); // |Sigma|^(-1/2), xiGivenYi_coefficient0
	}

	//convert parameter Pi, M, Epsilon to log form
	parameter.Pi = eln(parameter.Pi);
#ifdef INCLUDE_M
	parameter.M = eln(parameter.M);
#endif
	parameter.Epsilon = eln(parameter.Epsilon); //check if already eln form?

#ifdef INCLUDE_M
	UpdatePX_Y();
#else
	UpdatePX_Z();
#endif

	start = std::chrono::system_clock::now();
	learning();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	timeLogger << elapsed_seconds.count() << ",";

	start = std::chrono::system_clock::now();
	inference();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	timeLogger << elapsed_seconds.count() << std::endl;
	timeLogger.close();
	output();

	for(int i=0; i<data.allNodes.size(); i++)
	{
		delete data.allNodes[i];
	}
}

//@@@@@@ added by yanda
void cFlood::report()
{
	std::map<int, int> depth_count;
	std::set<int> chain_set;
	for(int i=0; i<data.allNodes.size(); i++)
	{
		Node * node = data.allNodes[i];
		int depth = tree.nodeLevels[i];
		depth_count[depth] ++;
		int chain = node->nodeChainID;
		chain_set.insert(chain);
	}

	for(auto it = depth_count.begin(); it != depth_count.end(); it++) std::cout<<"depth: "<<it->first<<", count: "<<it->second<<std::endl;
	std::cout<<"# chains = "<<chain_set.size()<<std::endl;
}

void cFlood::treeConstruct() {

	clock_t start_s = clock();

	vector<int>neighborChainID(8);
	vector<Node*>neighborTails(8);
	Node* tailNode = NULL;
	bool neighborEmpty, neighborIsNewList, neighborIsSameChainID;
	int curIdx, neighborIndex, newIdx;
	int row, column, minNeighborChainID;

	tree.nodeLevels.resize(parameter.nonEmptySize);
	std::fill(tree.nodeLevels.begin(), tree.nodeLevels.end(), -1);

	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		curIdx = data.elevationIndexPair[i].second;
		newIdx = data.old2newIdx.at(curIdx);
		row = curIdx / parameter.COLUMN;
		column = curIdx % parameter.COLUMN;

		neighborChainID.clear();
		neighborTails.clear();
		neighborEmpty = true;
		minNeighborChainID = parameter.nonEmptySize;
		tailNode = NULL;

		// check all 8 neighbors, record unique chain IDs and chain tails of its neighbors' chains
		for (int j = max(0, row - 1); j <= min(parameter.ROW - 1, row + 1); j++) {
			for (int k = max(0, column - 1); k <= min(parameter.COLUMN - 1, column + 1); k++) {
				neighborIndex = j * parameter.COLUMN + k;
				if(neighborIndex == curIdx) continue;

				if (data.NA.at(neighborIndex) == false) { // skip NA neighbor
					if (data.allNodes[neighborIndex]->nodeChainID != -1) { // -1 means unvisited
						tailNode = tree.tailPointers.at(data.allNodes[neighborIndex]->nodeChainID);
						while (tailNode->next != NULL) {
							if (tailNode->next->nodeChainID < tailNode->nodeChainID) {
								tailNode = tree.tailPointers.at(tailNode->next->nodeChainID);
							}
							else {
								tailNode = tailNode->next;
							}
						}// find tail node of neighbor node's chain

						neighborIsNewList = true;
						for (size_t n = 0; n < neighborTails.size(); n++) {
							if (neighborTails.at(n)->nodeIndex == tailNode->nodeIndex) {
								neighborIsNewList = false;
								break;
							}
						}// check if the tail has been recorded

						if (neighborIsNewList == true) {
							neighborTails.push_back(tailNode);
						}// record unique chain tail

						neighborIsSameChainID = false;
						for (size_t m = 0; m < neighborChainID.size(); m++) {
							if (tailNode->nodeChainID == neighborChainID.at(m)) {
								neighborIsSameChainID = true;
								break;
							}
						}
						if (neighborIsSameChainID == false) {
							neighborChainID.push_back(tailNode->nodeChainID);
						}// record unique chain ID
					}
				}

			}
		}// go throuth 8 neighbors

		int nodelevel = 0;
		if (neighborTails.size() != 0) {
			for (size_t m = 0; m < neighborTails.size(); m++) {
				neighborTails.at(m)->next = data.allNodes[curIdx];
				data.allNodes[curIdx]->parents.push_back(neighborTails.at(m));

				if (tree.nodeLevels.at(data.old2newIdx.at(neighborTails.at(m)->nodeIndex)) > nodelevel) {
					nodelevel = tree.nodeLevels.at(data.old2newIdx.at(neighborTails.at(m)->nodeIndex));
				}
			}
			tree.nodeLevels.at(newIdx) = nodelevel + 1; //update nodeLevles

			minNeighborChainID = neighborChainID.at(0);
			for (size_t n = 1; n < neighborChainID.size(); n++) {
				if (minNeighborChainID > neighborChainID.at(n)) {
					minNeighborChainID = neighborChainID.at(n);
				}
			}
			data.allNodes[curIdx]->nodeChainID = minNeighborChainID;
			tree.tailPointers.at(minNeighborChainID) = data.allNodes[curIdx];
		}
		else {// new local minimum
			tree.headPointers.push_back(data.allNodes[curIdx]);
			tree.tailPointers.push_back(data.allNodes[curIdx]);
			data.allNodes[curIdx]->nodeChainID = tree.headPointers.size() - 1; //id of the new chain
			tree.nodeLevels.at(newIdx) = 0;
			//@@@@@@
			leaf_oids.push_back(curIdx);
		}

		//std::cout << "i iterate" << i << endl;
		//for (size_t i = 0; i < headPointers.size(); i++) {
		//	Node* head = headPointers.at(i);
		//	std::cout << endl << "List " << i << endl << endl;
		//	while (head != NULL) {
		//		std::cout << head->nodeIndex << "  " << head->nodeChainID << "  " << head->elevation << endl;
		//		head = head->next;
		//	}
		//}
	}

	tree.nodeLevelIndexPair.resize(parameter.nonEmptySize);
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		tree.nodeLevelIndexPair.at(i) = make_pair(tree.nodeLevels.at(i), data.index.at(i));
	}
	sort(std::begin(tree.nodeLevelIndexPair), std::end(tree.nodeLevelIndexPair));

	//@@@@@@
	root_oid = tree.nodeLevelIndexPair.back().second;

	clock_t stop_s = clock();
	std::cout << "Tree generation " << endl;
	std::cout << "time: " << (stop_s - start_s) / float(CLOCKS_PER_SEC) << endl;

#ifdef TREE_REPORT
	//@@@@@@ added by yanda: check parallelism opportunities in the tree
	report();
#endif
};


void cFlood::UpdateTransProb() {
	if (cNum != 2) {
		cout << "cannot handle more than two classes now!" << endl;
		std::exit(1);
	}

	double eln(double);
#ifdef INCLUDE_M
	parameter.elnPy_z[0][0] = eln(1);
	parameter.elnPy_z[1][0] = eln(0);
	parameter.elnPy_z[0][1] = parameter.M;
	parameter.elnPy_z[1][1] = eln(1 - eexp(parameter.M));
#endif
	parameter.elnPz[0] = eln(1 - eexp(parameter.Pi));
	parameter.elnPz[1] = parameter.Pi;
	parameter.elnPz_zpn[0][0] = eln(1);
	parameter.elnPz_zpn[0][1] = parameter.Epsilon;
	parameter.elnPz_zpn[1][0] = eln(0);
	parameter.elnPz_zpn[1][1] = eln(1 - eexp(parameter.Epsilon));
	if (eexp(parameter.Epsilon) < 0 || eexp(parameter.Epsilon) > 1) {
		cout << "Epsilon Error: " << eexp(parameter.Epsilon) << endl;
	}
	if (eexp(parameter.Pi) < 0 || eexp(parameter.Pi) > 1) {
		cout << "Pi Error: " << eexp(parameter.Pi) << endl;
	}
#ifdef INCLUDE_M
	if (eexp(parameter.M) < 0 || eexp(parameter.M) > 1) {
		cout << "M Error: " << eexp(parameter.M) << endl;
	}
#endif
}

#ifdef INCLUDE_M
void cFlood::UpdatePX_Y() {
#else
void cFlood::UpdatePX_Z() {
#endif

	auto start = std::chrono::system_clock::now();

	// Calculate inverse of sigma
	//double adjointMatrix[cNum][Dim][Dim]; // To store adjoint of A[][] //commented by Da: not used
	double inverseMatrix[cNum][Dim][Dim]; // To store inverse of A[][]
	//for (int c = 0; c < cNum; c++) {
		//adjoint(parameter.Sigma[c], adjointMatrix[c]);  //commented by Da: not used
	//}
	for (int c = 0; c < cNum; c++) {
		if (!inverse(parameter.Sigma[c], inverseMatrix[c])) {
			cout << "Inverse error" << endl;
		}
	}

	//xiGivenZi_coefficient, log form
	for (int c = 0; c < cNum; c++) {// |Sigma|^(-1/2) 
		infer.lnCoefficient[c] = -0.5 * Dim * log(2 * M_PI) - 0.5 * log(fabs(determinant(parameter.Sigma[c], Dim)));
	}

	// Calculate p(x|y) if INCLUDE_M
	// Calculate p(x|z) if not INCLUDE_M

#pragma omp parallel for schedule(dynamic, CHUNK) num_threads(THREADS)
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		double intermediateValue[cNum][Dim] = { 0 };
		double likelihood[cNum] = { 0 };
		double xMinusMu[cNum][Dim] = { 0 };

		if (data.missingIndicator.at(i) == 1) { // Not missing data

			for (int c = 0; c < cNum; c++) {
				likelihood[c] = 0;
			}

			for (int c = 0; c < cNum; c++) {
				for (int d = 0; d < Dim; d++) {
					intermediateValue[c][d] = 0;
				}
			}

			// -0.5*(x-mu)' * Sigma^-1 * (x-mu), matrix multiply
			for (int c = 0; c < cNum; c++) {
				for (int d = 0; d < Dim; d++) {
					xMinusMu[c][d] = data.features.at(i * Dim + d) - parameter.Mu[c][d];
				}
			}

			for (int c = 0; c < cNum; c++) {
				for (int k = 0; k < Dim; k++) {
					for (int n = 0; n < Dim; n++) {
						intermediateValue[c][k] += xMinusMu[c][n] * inverseMatrix[c][n][k];
					}
					likelihood[c] += intermediateValue[c][k] * xMinusMu[c][k];
				}
			}

			infer.dryLikelihood[i] = -0.5 * likelihood[0];
			infer.waterLikelihood[i] = -0.5 * likelihood[1];

		}
		//else {// misssing data //p(x|z) will be assigned a constant directly, no need to calculate likelihood
		//	infer.dryLikelihood[i] = eln(0.5);
		//	infer.waterLikelihood[i] = eln(0.5);
		//}
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout<<"UpdatePX_Z(): "<<elapsed_seconds.count()<<" sec"<<endl;
}

void cFlood::leaf2root(int curIdx)
{
	int newIdx;
	Node* curNode;
	vector<int> ParentOfZ(4);
	vector<int> parentIndexes(4);
	//vector<double>tempTerms(8), tempTermsNew(8);
	//double message0 = 0, message1 = 0;

	curNode = data.allNodes.at(curIdx);

	newIdx = data.old2newIdx.at(curIdx);

#ifdef INCLUDE_M
	//vi: P(x|y)=a*exp()
	if (data.missingIndicator.at(newIdx) == 0) { // missing data
		infer.lnTop2y[newIdx * cNum] = eln(0.5);
		infer.lnTop2y[newIdx * cNum + 1] = eln(0.5);
	}
	else {
		infer.lnTop2y[newIdx * cNum] = infer.dryLikelihood[newIdx] + infer.lnCoefficient[0];
		infer.lnTop2y[newIdx * cNum + 1] = infer.waterLikelihood[newIdx] + infer.lnCoefficient[1];
	}
#ifdef DEBUG
	if (infer.lnTop2y[newIdx * cNum] > 0 || infer.lnTop2y[newIdx * cNum + 1] > 0) {
		cout << "wrong message: lnTop2y" << endl;
		//cout << "newIdx " << newIdx << endl;
	}
#endif
	////vi: P(x|z) = \sum_y P(x|y)P(y|z)
	for (int z = 0; z < cNum; z++) {
		infer.lnvi[newIdx * cNum + z] = LOGZERO;
		for (int y = 0; y < cNum; y++) {
			infer.lnvi[newIdx * cNum + z] = elnsum(infer.lnvi[newIdx * cNum + z],
				elnproduct(infer.lnTop2y[newIdx * cNum + y], parameter.elnPy_z[y][z]));
		}
	}
#else
	//vi: P(x|z)=a*exp()
	if (data.missingIndicator.at(newIdx) == 0) { // missing data
		infer.lnvi[newIdx * cNum] = eln(0.5);
		infer.lnvi[newIdx * cNum + 1] = eln(0.5);
	}
	else {
		infer.lnvi[newIdx * cNum] = infer.dryLikelihood[newIdx] + infer.lnCoefficient[0];
		infer.lnvi[newIdx * cNum + 1] = infer.waterLikelihood[newIdx] + infer.lnCoefficient[1];
	}
#endif

#ifdef DEBUG
	if (infer.lnvi[newIdx * cNum] > 0 || infer.lnvi[newIdx * cNum + 1] > 0) {
		cout << "curIdx: " << curIdx << endl;
		cout << "newIdx: " << newIdx << endl;
		cout << "wrong message: lnvi" << endl;
		cout << "infer.lnvi[newIdx * cNum] = " << infer.lnvi[newIdx * cNum] << endl;
		cout << "infer.lnvi[newIdx * cNum + 1] = " << infer.lnvi[newIdx * cNum + 1] << endl;
		cout << "infer.dryLikelihood[newIdx] = " << infer.dryLikelihood[newIdx] << endl;
		cout << "infer.waterLikelihood[newIdx] = " << infer.waterLikelihood[newIdx] << endl;
		cout << "infer.lnCoefficient[0] = " << infer.lnCoefficient[0] << endl;
		cout << "infer.lnCoefficient[1] = " << infer.lnCoefficient[1] << endl;
		int tmp;
		std::cin >> tmp;
	}
#endif

	if (curNode->parents.size() == 0) {// leaf nodes
									   //fi: P(z)
		for (int z = 0; z < cNum; z++) {
			infer.lnfi[newIdx * cNum + z] = parameter.elnPz[z];
		}

		//fo: fo=fi*vi
		for (int z = 0; z < cNum; z++) {
			infer.lnfo[newIdx * cNum + z] = elnproduct(infer.lnfi[newIdx * cNum + z], infer.lnvi[newIdx * cNum + z]);
		}
	}
	else if (curNode->parents.size() > 0) {  // internal nodes
											 //fi: \sum_zpn (\product_k fko) * P(zn|zpn)
		int combi = 1;
		for (int k = 0; k < curNode->parents.size(); k++) {
			combi *= cNum;
		}//number of combinations of all parent z status

		parentIndexes.clear();
		for (size_t k = 0; k < curNode->parents.size(); k++) {
			parentIndexes.push_back(data.old2newIdx.at(curNode->parents.at(k)->nodeIndex));
		}

		//Z status of all parents; enumerated through all "combi" combinations
		ParentOfZ.clear(); ParentOfZ.resize(curNode->parents.size());

		for (int z = 0; z < cNum; z++) {

			infer.lnfi[newIdx * cNum + z] = LOGZERO;

			//all combinations
			for (int c = 0; c < combi; c++) {
				//one possible combination c; get pzn parent z-value configuration
				int Zpn_product = 1;
				for (int p = 0; p < curNode->parents.size(); p++) {
					ParentOfZ.at(p) = (c >> p) & 1; //need check? decode status of parent p
					Zpn_product *= ParentOfZ.at(p);
				}
				//compute (\product_k fko) based on this specific combination
				double allProdfko = eln(1);
				for (int p = 0; p < curNode->parents.size(); p++) {
					allProdfko = elnproduct(allProdfko, infer.lnfo[parentIndexes.at(p) * cNum + ParentOfZ.at(p)]);
				}
				allProdfko = elnproduct(allProdfko, parameter.elnPz_zpn[z][Zpn_product]);

				//integration over zpn
				infer.lnfi[newIdx * cNum + z] = elnsum(infer.lnfi[newIdx * cNum + z], allProdfko);
			}
		}

		//fo: fo=fi*vi
		for (int z = 0; z < cNum; z++) {
			infer.lnfo[newIdx * cNum + z] = elnproduct(infer.lnfi[newIdx * cNum + z], infer.lnvi[newIdx * cNum + z]);
		}
	}

#ifdef DEBUG
	for (int c = 0; c < cNum; c++) {
		if (infer.lnfi[newIdx * cNum + c] > 0) {
			cout << "wrong message: lnfi" << endl;
			cout << " lnfi[newIdx * cNum] " << infer.lnfi[newIdx * cNum + c] << endl;
			int tmp;
			cin >> tmp;
		}
		if (infer.lnfo[newIdx * cNum + c] > 0) {
			cout << "wrong message: lnfo" << endl;
			cout << " lnfo[newIdx * cNum] " << infer.lnfo[newIdx * cNum + c] << endl;
			int tmp;
			cin >> tmp;
		}
	}
#endif

}

void cFlood::leaf2root_thread(int curIdx) //continue to process child towards the root (unless child is not ready)
{
	while(curIdx != -1){
		leaf2root(curIdx);
		curIdx = data.allNodes[curIdx]->add_child_counter();
	}
}

bool cFlood::get_and_process_l2r_tasks(mutex & q_mtx, queue<int> & task_queue, int batch_size = 1){
	queue<int> collector;
	q_mtx.lock();
	while(!task_queue.empty() && batch_size>0){
		collector.push(task_queue.front());
		task_queue.pop();
		batch_size--;
	}
	q_mtx.unlock();

	if(collector.empty()) return false;

	//process tasks in "collector"
	while(!collector.empty()){
		leaf2root_thread(collector.front());
		collector.pop();
	}

	return true;
}

void cFlood::thread_run_l2r(mutex & q_mtx, queue<int> & task_queue, atomic<bool> & global_end_label, size_t & global_num_idle, mutex & mtx_go, condition_variable & cv_go, bool & ready_go){
	while(global_end_label == false) //otherwise, thread terminates
	{
		bool busy = get_and_process_l2r_tasks(q_mtx, task_queue);
		if(!busy) //if task_queue is empty
		{
			unique_lock<mutex> lck(mtx_go);
			ready_go = false;
			global_num_idle++;
			while(!ready_go)
			{
				cv_go.wait(lck);
			}
			global_num_idle--;
		}
	}
}

void cFlood::Leaf2Root_MsgProp() {

	auto start = std::chrono::system_clock::now();

	/* old implementation by level
	//Horizontally leaf to root direction graph traversal
	for (int i = 0; i < parameter.nonEmptySize; i++)
		leaf2root(tree.nodeLevelIndexPair[i].second);
	*/

	/* new sinle-threaded implementation
	for (int i = 0; i < leaf_oids.size(); i++)
	{
		int seed_oid = leaf_oids[i];
		leaf2root_thread(seed_oid);
	}
	*/

	mutex mtx_go;
	condition_variable cv_go;
	bool ready_go = true; //protected by mtx_go

	atomic<bool> global_end_label(false); //end tag, to be set by main thread
	size_t global_num_idle = 0; // how many tasks are idle, protected by mtx_go

	mutex q_mtx;
	queue<int> task_queue;

	for(int i=0; i<leaf_oids.size(); i++)
		task_queue.push(leaf_oids[i]);

	//------------------------ create computing threads
	vector<thread> threads;
	for(int i=0; i<THREADS; i++)
	{
		threads.push_back(thread(&cFlood::thread_run_l2r, this, ref(q_mtx), ref(task_queue), ref(global_end_label), ref(global_num_idle), ref(mtx_go), ref(cv_go), ref(ready_go)));
	}
	//------------------------
	while(global_end_label == false)
	{
		usleep(WAIT_TIME_WHEN_IDLE); //avoid busy-checking
		//------
		q_mtx.lock();
		if(!task_queue.empty())
		{
			//case 1: there are tasks to process, wake up threads
			//the case should go first, since we want to wake up threads early, not till all are idle as in case 2
			mtx_go.lock();
			ready_go = true;
			cv_go.notify_all(); //release threads to compute tasks
			mtx_go.unlock();
		}
		else
		{
			mtx_go.lock();
			if(global_num_idle == THREADS)
			//case 2: every thread is waiting, guaranteed since mtx_go is locked
			//since we are in else-branch, task_queue must be empty
			{
				global_end_label = true;
				ready_go = true;
				cv_go.notify_all(); //release threads to their looping
			}
			//case 3: else, some threads are still processing tasks, check in next round
			mtx_go.unlock();
		}
		q_mtx.unlock();
	}
	//------------------------
	for(int i=0; i<THREADS; i++) threads[i].join();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout<<"Leaf2Root_MsgProp(): "<<elapsed_seconds.count()<<" sec"<<endl;
}

void cFlood::root2leaf(int curIdx)
{

	int newIdx;
	int childIdx; // Here the notation child is based on bottom up direcion.
	Node* curNode, *childNode;
	vector<int> SiblingOfZ(4);
	vector<int> siblingIndexes(4);
	//vector<double>tempTerms(8), tempTermsNew(8);
	//double message0 = 0, message1 = 0;


	////root to leaf traversal
	//int count = 0; //commented by Da; not used
	int childNewIndex;

	curNode = data.allNodes.at(curIdx);
	childNode = curNode->next;

	newIdx = data.old2newIdx.at(curIdx);

	if (childNode == NULL) { //gi: for root
		for (int z = 0; z < cNum; z++) {
			infer.lngi[newIdx * cNum + z] = eln(1); // 0 = eln(1), gi = 1
		}
	}
	else {
		//gi: for non-root nodes
		childIdx = childNode->nodeIndex;
		childNewIndex = data.old2newIdx.at(childIdx);
		siblingIndexes.clear();
		for (size_t k = 0; k < childNode->parents.size(); k++) {
			if (childNode->parents.at(k)->nodeIndex == curNode->nodeIndex) {
				continue;
			}
			else {
				siblingIndexes.push_back(data.old2newIdx.at(childNode->parents.at(k)->nodeIndex));
			}
		}


		int combi = 1;
		for (int k = 0; k < siblingIndexes.size(); k++) {
			combi *= cNum;
		}//number of combinations of all siblings of z status


		 //Z status of all siblings; enumerated through all "combi" combinations
		SiblingOfZ.clear(); SiblingOfZ.resize(siblingIndexes.size());

		for (int z = 0; z < cNum; z++) {

			infer.lngi[newIdx * cNum + z] = LOGZERO;
			for (int zc = 0; zc < cNum; zc++) {

				//all combinations
				for (int c = 0; c < combi; c++) {
					//one possible combination c; get siblings z-value configuration
					int Zpc_product = 1;
					for (int s = 0; s < siblingIndexes.size(); s++) {
						SiblingOfZ.at(s) = (c >> s) & 1; //decode status of sibling s, c used as binary here
						Zpc_product *= SiblingOfZ.at(s);
					}
					Zpc_product *= z; // don't miss zn its own status

									  //compute (\product_k fko) based on this specific combination
					double allProdfko = eln(1);
					for (int s = 0; s < siblingIndexes.size(); s++) {
						allProdfko = elnproduct(allProdfko, infer.lnfo[siblingIndexes.at(s) * cNum + SiblingOfZ.at(s)]);
					}
					allProdfko = elnproduct(allProdfko, parameter.elnPz_zpn[zc][Zpc_product]);
					allProdfko = elnproduct(allProdfko, infer.lngo[childNewIndex * cNum + zc]);

					//integration over siblings(Zsn), Zc(outer loop)
					infer.lngi[newIdx * cNum + z] = elnsum(infer.lngi[newIdx * cNum + z], allProdfko);
				}
			}
		}
	}

	for (int z = 0; z < cNum; z++) {
		infer.lngo[newIdx * cNum + z] = elnproduct(infer.lngi[newIdx * cNum + z], infer.lnvi[newIdx * cNum + z]);
	}
#ifdef INCLUDE_M
	//from z to top
	for (int z = 0; z < cNum; z++) {
		infer.lnz2top[newIdx * cNum + z] = elnproduct(infer.lnfi[newIdx * cNum + z], infer.lngi[newIdx * cNum + z]);
	}

	////from bottom to y, after factor node; \sum_z z2top * p(y|z)
	for (int y = 0; y < cNum; y++) {
		infer.lnbottom2y[newIdx * cNum + y] = LOGZERO;
		for (int z = 0; z < cNum; z++) {
			infer.lnbottom2y[newIdx * cNum + y] = elnsum(infer.lnbottom2y[newIdx * cNum + y],
				elnproduct(infer.lnz2top[newIdx * cNum + z], parameter.elnPy_z[y][z]));
		}
	}
#endif
#ifdef DEBUG
	for (int c = 0; c < cNum; c++) {
		if (infer.lngi[newIdx * cNum + c] > 0) {
			cout << " lngi[newIdx * cNum] " << infer.lngi[newIdx * cNum + c] << endl;
			int tmp;
			cin >> tmp;
		}
	}

	for (int c = 0; c < cNum; c++) {
		if (infer.lngo[newIdx * cNum + c] > 0) {
			cout << " lngo[newIdx * cNum] " << infer.lngo[newIdx * cNum + c] << endl;
			int tmp;
			cin >> tmp;
		}
	}
#endif
}

//continue to process a parent, while putting the other parents to task queue (unless current node is leaf)
void cFlood::root2leaf_thread(int curIdx, conque<int> & task_queue)
{
	while(true){
		root2leaf(curIdx);
		vector<Node*> & parents = data.allNodes[curIdx]->parents;
		if(parents.size() == 0) break;
		else
		{
			curIdx = parents[0]->nodeIndex;
			for(int i=1; i<parents.size(); i++)
			{
				task_queue.enqueue(parents[i]->nodeIndex);
			}
		}
	}
}

bool cFlood::get_and_process_r2l_tasks(conque<int> & task_queue, int batch_size = 1){
	queue<int> collector;
	int nid;
	while(batch_size>0){
		if(task_queue.dequeue(nid))
		{
			collector.push(nid);
			batch_size--;
		}
		else break;
	}

	if(collector.empty()) return false;

	//process tasks in "collector"
	while(!collector.empty()){
		root2leaf_thread(collector.front(), task_queue);
		collector.pop();
	}

	return true;
}

void cFlood::thread_run_r2l(conque<int> & task_queue, atomic<bool> & global_end_label, size_t & global_num_idle, mutex & mtx_go, condition_variable & cv_go, bool & ready_go){
	while(global_end_label == false) //otherwise, thread terminates
	{
		bool busy = get_and_process_r2l_tasks(task_queue);
		if(!busy) //if task_queue is empty
		{
			unique_lock<mutex> lck(mtx_go);
			ready_go = false;
			global_num_idle++;
			while(!ready_go)
			{
				cv_go.wait(lck);
			}
			global_num_idle--;
		}
	}
}

void cFlood::Root2Leaf_MsgProp() {

	auto start = std::chrono::system_clock::now();

	/* old single-threaded implementation
	for (int i = parameter.nonEmptySize - 1; i >= 0; i--)
		root2leaf(tree.nodeLevelIndexPair[i].second);
	*/

	mutex mtx_go;
	condition_variable cv_go;
	bool ready_go = true; //protected by mtx_go

	atomic<bool> global_end_label(false); //end tag, to be set by main thread
	size_t global_num_idle = 0; // how many tasks are idle, protected by mtx_go

	mutex q_mtx;
	conque<int> task_queue;

	task_queue.enqueue(root_oid);

	//------------------------ create computing threads
	vector<thread> threads;
	for(int i=0; i<THREADS; i++)
	{
		threads.push_back(thread(&cFlood::thread_run_r2l, this, ref(task_queue), ref(global_end_label), ref(global_num_idle), ref(mtx_go), ref(cv_go), ref(ready_go)));
	}
	//------------------------
	while(global_end_label == false)
	{
		usleep(WAIT_TIME_WHEN_IDLE); //avoid busy-checking
		//------
		q_mtx.lock();
		if(!task_queue.empty())
		{
			//case 1: there are tasks to process, wake up threads
			//the case should go first, since we want to wake up threads early, not till all are idle as in case 2
			mtx_go.lock();
			ready_go = true;
			cv_go.notify_all(); //release threads to compute tasks
			mtx_go.unlock();
		}
		else
		{
			mtx_go.lock();
			if(global_num_idle == THREADS)
			//case 2: every thread is waiting, guaranteed since mtx_go is locked
			//since we are in else-branch, task_queue must be empty
			{
				global_end_label = true;
				ready_go = true;
				cv_go.notify_all(); //release threads to their looping
			}
			//case 3: else, some threads are still processing tasks, check in next round
			mtx_go.unlock();
		}
		q_mtx.unlock();
	}
	//------------------------
	for(int i=0; i<THREADS; i++) threads[i].join();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout<<"Root2Leaf_MsgProp(): "<<elapsed_seconds.count()<<" sec"<<endl;
}

void cFlood::UpdateParameters() {
	auto start = std::chrono::system_clock::now();

	//// Calculate new parameter
	double topEpsilon = LOGZERO, bottomEpsilon = LOGZERO, topPi = LOGZERO, bottomPi = LOGZERO;
#ifdef INCLUDE_M
	double topM = LOGZERO, bottomM = LOGZERO;
#endif
	double bottomMu[cNum] = { LOGZERO };
	double tempMu[cNum][Dim] = { LOGZERO };
	double SigmaTemp[cNum][Dim][Dim] = { 0 };

	//@@@@@@ thread-local aggregators
	vector<double> topEps_t(THREADS, LOGZERO);
	vector<double> bottomEps_t(THREADS, LOGZERO);
	vector<double> topPi_t(THREADS, LOGZERO);
	vector<double> bottomPi_t(THREADS, LOGZERO);
#ifdef INCLUDE_M
	vector<double> topM_t(THREADS, LOGZERO);
	vector<double> bottomM_t(THREADS, LOGZERO);
#endif
	//------
	vector<vector<double> > bottomMu_t(THREADS);
	for(int i=0; i<THREADS; i++) bottomMu_t[i].resize(cNum, LOGZERO);
	vector<vector<vector<double> > > tempMu_t(THREADS);
	for(int i=0; i<THREADS; i++)
	{
		tempMu_t[i].resize(cNum);
		for(int j=0; j<cNum; j++) tempMu_t[i][j].resize(Dim, LOGZERO);
	}
	//------
	vector<vector<vector<vector<double> > > > SigmaTemp_t(THREADS);
	for(int i=0; i<THREADS; i++)
	{
		SigmaTemp_t[i].resize(cNum);
		for(int j=0; j<cNum; j++)
		{
			SigmaTemp_t[i][j].resize(Dim);
			for(int k=0; k<Dim; k++) SigmaTemp_t[i][j][k].resize(Dim, 0);
		}
	}

#pragma omp parallel for schedule(dynamic, CHUNK) num_threads(THREADS)
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		int curIdx = data.index[i]; // i: new index; curIdx: original index
		int thread_id = omp_get_thread_num();

#ifdef INCLUDE_M
		//// M,  go through all nodes
		for (int z = 0; z < cNum; z++) {
			for (int y = 0; y < cNum; y++) {
				topM_t[thread_id] = elnsum(topM_t[thread_id], elnproduct(eln(z * (1 - y)), infer.marginal_YnZn[i*cNum*cNum + y*cNum + z]));
				bottomM_t[thread_id] = elnsum(bottomM_t[thread_id], elnproduct(eln(z), infer.marginal_YnZn[i*cNum*cNum + y*cNum + z]));
			}
		}
		//topM = elnsum(topM, infer.marginal_YnZn[i * cNum]);
		//bottomM = elnsum(bottomM, elnsum(infer.marginal_YnZn[i * cNum], infer.marginal_YnZn[i * cNum + 1]));
#endif
		// Epsilon, zi has parents
		if (data.allNodes[curIdx]->parents.size() > 0) {
			for (int z = 0; z < cNum; z++) {
				for (int zp = 0; zp < cNum; zp++) {
					topEps_t[thread_id] = elnsum(topEps_t[thread_id], elnproduct(eln(zp*(1 - z)), infer.marginal_ZnZpn[i * cNum*cNum + z*cNum + zp]));
					bottomEps_t[thread_id] = elnsum(bottomEps_t[thread_id], elnproduct(eln(zp), infer.marginal_ZnZpn[i * cNum*cNum + z*cNum + zp]));
				}
			}
			//topEpsilon = elnsum(topEpsilon, infer.marginal_ZnZpn[i * cNum]);
			//bottomEpsilon = elnsum(bottomEpsilon, elnsum(infer.marginal_ZnZpn[i * cNum], infer.marginal_ZnZpn[i * cNum + 1]));
		}
		// Pi, zi is leaf node
		else {
			for (int z = 0; z < cNum; z++) {
				topPi_t[thread_id] = elnsum(topPi_t[thread_id], elnproduct(eln(1 - z), infer.marginal_Zn[i * cNum + z]));
				bottomPi_t[thread_id] = elnsum(bottomPi_t[thread_id], infer.marginal_Zn[i * cNum + z]);
			}

			//for (int z = 0; z < cNum; z++) {
			//	topPi = elnsum(topPi, elnproduct( eln(1-z), infer.marginal_ZnZpn[i * cNum*cNum + z*cNum]));
			//	bottomPi = elnsum(bottomPi, infer.marginal_ZnZpn[i * cNum*cNum + z*cNum]);
			//}
			//topPi = elnsum(topPi, infer.marginal_ZnZpn[i * cNum]);
			//bottomPi = elnsum(bottomPi, elnsum(infer.marginal_ZnZpn[i * cNum], infer.marginal_ZnZpn[i * cNum + 1]));
		}

		// Mu0, Mu1, go through all nodes

		//for (size_t j = 0; j < Dim; j++) { //data.features may contain negative values
		//	for (int c = 0; c < cNum; c++) {
		//		tempMu[c][j] += data.features[i * Dim + j] * eexp(infer.marginal_Yn[i * cNum + c]);
		//	}
		//}
		//for (size_t j = 0; j < Dim; j++) {
		//	for (int c = 0; c < cNum; c++) {
		//		tempMu[c][j] = eln(tempMu[c][j]);
		//	}
		//}

		for (size_t j = 0; j < Dim; j++) {
			for (int c = 0; c < cNum; c++) {
				if (data.features[i * Dim + j] + featuresAbsoluteMin[j] < 0) {
					cout << "Negative input feature value" << data.features[i * Dim + j] << endl;
				}
#ifdef INCLUDE_M
				tempMu_t[thread_id][c][j] = elnsum(tempMu_t[thread_id][c][j], elnproduct(eln(data.features[i * Dim + j] + featuresAbsoluteMin[j]), infer.marginal_Yn[i * cNum + c]));
#else
				tempMu_t[thread_id][c][j] = elnsum(tempMu_t[thread_id][c][j], elnproduct(eln(data.features[i * Dim + j] + featuresAbsoluteMin[j]), infer.marginal_Zn[i * cNum + c]));
#endif
			}
		}
		for (int c = 0; c < cNum; c++) {
#ifdef INCLUDE_M
			bottomMu_t[thread_id][c] = elnsum(bottomMu_t[thread_id][c], infer.marginal_Yn[i * cNum + c]);
#else
			bottomMu_t[thread_id][c] = elnsum(bottomMu_t[thread_id][c], infer.marginal_Zn[i * cNum + c]);
#endif
		}
	}

	//@@@@@@ summing thread-local aggregators
	for(int i=0; i<THREADS; i++)
	{
		topEpsilon = elnsum(topEpsilon, topEps_t[i]);
		bottomEpsilon = elnsum(bottomEpsilon, bottomEps_t[i]);
		topPi = elnsum(topPi, topPi_t[i]);
		bottomPi = elnsum(bottomPi, bottomPi_t[i]);
#ifdef INCLUDE_M
		topM = elnsum(topM, topM_t[i]);
		bottomM = elnsum(bottomM, bottomM_t[i]);
#endif
		for(int j=0; j<cNum; j++) bottomMu[j] = elnsum(bottomMu[j], bottomMu_t[i][j]);
		for(int j=0; j<cNum; j++)
		{
			for(int k=0; k<Dim; k++) tempMu[j][k] = elnsum(tempMu[j][k], tempMu_t[i][j][k]);
		}
	}


	parameter.Epsilon = elnproduct(topEpsilon, -1 * bottomEpsilon);
	parameter.Pi = elnproduct(topPi, -1 * bottomPi);
#ifdef INCLUDE_M
	parameter.M = elnproduct(topM, -1 * bottomM);
#endif

	// reserve eln(Mu) form
	for (size_t j = 0; j < Dim; j++) {
		for (int c = 0; c < cNum; c++) {
			parameter.elnMu[c][j] = elnproduct(tempMu[c][j], -1 * bottomMu[c]);
		}
	}

	// convert Mu to normal
	for (size_t j = 0; j < Dim; j++) {
		for (int c = 0; c < cNum; c++) {
			parameter.Mu[c][j] = eexp(parameter.elnMu[c][j]) - featuresAbsoluteMin[j];
		}
	}

#pragma omp parallel for schedule(dynamic, CHUNK) num_threads(THREADS)
	// Update Sigma
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		int thread_id = omp_get_thread_num();

		double xMinusMu[cNum][Dim];
		for (int c = 0; c < cNum; c++) {
			for (size_t j = 0; j < Dim; j++) {
				xMinusMu[c][j] = data.features.at(i * Dim + j) - parameter.Mu[c][j];
			}
		}

		for (int c = 0; c < cNum; c++) {
			for (size_t m = 0; m < Dim; m++) { // row
				for (size_t n = 0; n < Dim; n++) { // column
#ifdef INCLUDE_M
					SigmaTemp_t[thread_id][c][m][n] += xMinusMu[c][m] * xMinusMu[c][n] * eexp(infer.marginal_Yn[i * cNum + c]);
#else
					SigmaTemp_t[thread_id][c][m][n] += xMinusMu[c][m] * xMinusMu[c][n] * eexp(infer.marginal_Zn[i * cNum + c]);
#endif
				}
			}
		}
	}

	//@@@@@@ summing thread-local aggregators
	for(int i=0; i<THREADS; i++)
	{
		for(int j=0; j<cNum; j++)
		{
			for(int k=0; k<Dim; k++)
			{
				for(int kk=0; kk<Dim; kk++) SigmaTemp[j][k][kk] += SigmaTemp_t[i][j][k][kk];
			}
		}
	}

	for (int c = 0; c < cNum; c++) {
		for (size_t i = 0; i < Dim; i++) {
			for (size_t j = 0; j < Dim; j++) {
				parameter.Sigma[c][i][j] = SigmaTemp[c][i][j] / eexp(bottomMu[c]); // bottom is the same as Mu
			}
		}
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout<<"UpdateParameters(): "<<elapsed_seconds.count()<<" sec"<<endl;

}


void cFlood::UpdateMarginalProb() {
	// Calculate Marginal distribution

	auto start = std::chrono::system_clock::now();

#pragma omp parallel for schedule(dynamic, CHUNK) num_threads(THREADS)
	for (size_t i = 0; i < parameter.nonEmptySize; i++) {

		int curIdx, newIdx;
		Node* curNode;
		vector<int>ParentOfZ(4);
		vector<int>parentIndexes(4);
		//vector<double>tempTerms(8), tempTermsNew(8);
		//double message0 = 0, message1 = 0;
		double normFactor;

		curIdx = data.index[i];
		curNode = data.allNodes[curIdx];

		newIdx = data.old2newIdx.at(curIdx);
#ifdef INCLUDE_M
		////// p(y|X, theta) 
		normFactor = LOGZERO;
		for (int y = 0; y < cNum; y++) {
			infer.marginal_Yn[newIdx * cNum + y] = elnproduct(infer.lnbottom2y[newIdx * cNum + y], infer.lnTop2y[newIdx * cNum + y]);
			normFactor = elnsum(normFactor, infer.marginal_Yn[newIdx * cNum + y]);
		}

		for (int y = 0; y < cNum; y++) {
			infer.marginal_Yn[newIdx * cNum + y] = elnproduct(infer.marginal_Yn[newIdx * cNum + y], -1 * normFactor);
#ifdef DEBUG
			if (infer.marginal_Yn[newIdx * cNum + y] > 0) {
				cout << " marginal_Yn " << infer.lngi[newIdx * cNum + y] << endl;
				int tmp;
				cin >> tmp;
			}
#endif
		}


		////// p(y, z|X, theta) = gi(z) * fi(z) * Top2y * p(y|z) = Z2Top(z) * Top2y(y) * p(y|z)
		normFactor = LOGZERO;
		for (int z = 0; z < cNum; z++) {
			for (int y = 0; y < cNum; y++) {
				infer.marginal_YnZn[newIdx*cNum*cNum + y*cNum + z] = elnproduct(infer.lnz2top[newIdx * cNum + z], infer.lnTop2y[newIdx * cNum + y]);
				infer.marginal_YnZn[newIdx*cNum*cNum + y*cNum + z] = elnproduct(infer.marginal_YnZn[newIdx*cNum*cNum + y*cNum + z], parameter.elnPy_z[y][z]);
				normFactor = elnsum(normFactor, infer.marginal_YnZn[newIdx*cNum*cNum + y*cNum + z]);
			}
		}

		for (int z = 0; z < cNum; z++) {
			for (int y = 0; y < cNum; y++) {
				infer.marginal_YnZn[newIdx*cNum*cNum + y*cNum + z] = elnproduct(infer.marginal_YnZn[newIdx*cNum*cNum + y*cNum + z], -1 * normFactor);
#ifdef DEBUG
				if (infer.marginal_YnZn[newIdx*cNum*cNum + y * cNum + z] > 0) {
					cout << " marginal_YnZn " << infer.marginal_YnZn[newIdx*cNum*cNum + y * cNum + z] << endl;
					int tmp;
					cin >> tmp;
				}
#endif
			}
		}
#endif

		// p(z, zp|X, theta) = go(z) * \product_k{k belong to zp}(fko) * p(z|zp)
		normFactor = LOGZERO;
		if (curNode->parents.size() > 0) {

			int combi = 1;
			for (int k = 0; k < curNode->parents.size(); k++) {
				combi *= cNum;
			}//number of combinations of all parent z status

			parentIndexes.clear();
			for (size_t k = 0; k < curNode->parents.size(); k++) {
				parentIndexes.push_back(data.old2newIdx.at(curNode->parents.at(k)->nodeIndex));
			}

			//Z status of all parents; enumerated through all "combi" combinations
			ParentOfZ.clear(); ParentOfZ.resize(curNode->parents.size());

			int marginalSize = combi * cNum;
			double *tempZnZpn = new double[marginalSize];

			for (int z = 0; z < cNum; z++) {
				for (int zp = 0; zp < combi; zp++) {//all combinations
					tempZnZpn[z*combi + zp] = LOGZERO;

					//one possible combination c; get pzn parent z-value configuration
					int Zpn_product = 1;
					for (int p = 0; p < curNode->parents.size(); p++) {
						ParentOfZ.at(p) = (zp >> p) & 1; // decode status of parent p
						Zpn_product *= ParentOfZ.at(p);
					}
					//compute (\product_k fko) based on this specific combination
					double allProdfko = eln(1);
					for (int p = 0; p < curNode->parents.size(); p++) {
						allProdfko = elnproduct(allProdfko, infer.lnfo[parentIndexes.at(p) * cNum + ParentOfZ.at(p)]);
					}
					allProdfko = elnproduct(allProdfko, parameter.elnPz_zpn[z][Zpn_product]);
					allProdfko = elnproduct(allProdfko, infer.lngo[newIdx * cNum + z]);

					tempZnZpn[z*combi + zp] = allProdfko;
					normFactor = elnsum(normFactor, tempZnZpn[z*combi + zp]);
				}
			}

			//marginal_ZnZpn select the first and last term for each z
			for (int z = 0; z < cNum; z++) {
				for (int zp = 0; zp < combi; zp += combi - 1) {
					infer.marginal_ZnZpn[newIdx*cNum*cNum + z*cNum + zp] = elnproduct(tempZnZpn[z*combi + zp], -1 * normFactor);
#ifdef DEBUG
					if (infer.marginal_ZnZpn[newIdx*cNum*cNum + z * cNum + zp] > 0) {
						cout << " marginal_ZnZpn " << infer.marginal_ZnZpn[newIdx*cNum*cNum + z * cNum + zp] << endl;
						int tmp;
						cin >> tmp;
					}
#endif
				}
			}

			delete[] tempZnZpn;

		}


		// P(z|X, theta) = gi * fi * vi, Marginal Zn
		normFactor = LOGZERO;
		for (int z = 0; z < cNum; z++) {
			infer.marginal_Zn[newIdx * cNum + z] = elnproduct(elnproduct(infer.lnfi[newIdx * cNum + z], infer.lngi[newIdx * cNum + z]), infer.lnvi[newIdx * cNum + z]);
			normFactor = elnsum(normFactor, infer.marginal_Zn[newIdx * cNum + z]);
		}

		for (int z = 0; z < cNum; z++) {
			infer.marginal_Zn[newIdx * cNum + z] = elnproduct(infer.marginal_Zn[newIdx * cNum + z], -1 * normFactor);
#ifdef DEBUG
			if (infer.marginal_Zn[newIdx * cNum + z] > 0) {
				cout << " marginal_Zn " << infer.marginal_Zn[newIdx * cNum + z] << endl;
				int tmp;
				cin >> tmp;
			}
#endif
		}

#ifdef DEBUG
		for (int c = 0; c < cNum; c++) {
			if (infer.marginal_Zn[newIdx * cNum + c] > 0) {
				cout << "wrong message: marginal_Zn" << endl;
			}
			if (infer.marginal_ZnZpn[newIdx * cNum + c] > 0) {
				cout << "wrong message: marginal_ZnZpn" << endl;
			}
		}
#endif
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	cout<<"UpdateMarginalProb(): "<<elapsed_seconds.count()<<" sec"<<endl;
}


void cFlood::learning() {
	clock_t start_s = clock();
#ifdef INCLUDE_M
	infer.lnTop2y.resize(parameter.nonEmptySize * cNum);
	infer.lnbottom2y.resize(parameter.nonEmptySize * cNum); //Vertical incoming message from z to y, after factor node
	infer.lnz2top.resize(parameter.nonEmptySize * cNum); //from z to y, outgoing of z, before factor node
#endif
	infer.lnvi.resize(parameter.nonEmptySize * cNum);
	infer.lnfi.resize(parameter.nonEmptySize * cNum); // from zp to z except zp=NULL, after factor node
	infer.lnfo.resize(parameter.nonEmptySize * cNum); // product of lnfi and lngi, before factor node
	infer.lngi.resize(parameter.nonEmptySize * cNum); // Horizontal incoming message from z to zp, after factor node
	infer.lngo.resize(parameter.nonEmptySize * cNum);

#ifdef INCLUDE_M
	infer.marginal_Yn.resize(parameter.nonEmptySize * cNum);
	// the order is p(y=0,z=0), p(y=0,z=1), p(y=1,z=0), p(y=1,z=1), [index*cNum*cNum + y*cNum + z], only zi=1 is useful later
	infer.marginal_YnZn.resize(parameter.nonEmptySize * cNum * cNum);
#endif
	infer.marginal_ZnZpn.resize(parameter.nonEmptySize * cNum * cNum); // All except bottom nodes
	infer.marginal_Zn.resize(parameter.nonEmptySize * cNum); // Marginal Zn

	int iterateTimes = 0;
	bool iterator = true;

	double PiOld, EpsilonOld;
#ifdef INCLUDE_M
	double MOld;
#endif
	double MuOld[cNum][Dim], SigmaOld[cNum][Dim][Dim];
	ofstream parameterLog;

	std::string plogName = HMTOutputLocation + HMTParaLog;
	remove(plogName.c_str());
	parameterLog.open(plogName.c_str(), ofstream::app);

	while (iterator) {
		auto start = std::chrono::system_clock::now();

		UpdateTransProb();

		//copy current parameters to compare across iterations
#ifdef INCLUDE_M
		MOld = parameter.M;
#endif
		PiOld = parameter.Pi;
		EpsilonOld = parameter.Epsilon;
		for (int c = 0; c < cNum; c++) {
			for (int i = 0; i < Dim; i++) {
				MuOld[c][i] = parameter.Mu[c][i];
				for (size_t j = 0; j < Dim; j++) {
					SigmaOld[c][i][j] = parameter.Sigma[c][i][j];
				}
			}
		}

		Leaf2Root_MsgProp();
		Root2Leaf_MsgProp();

		UpdateMarginalProb();

		UpdateParameters();

#ifdef INCLUDE_M
		UpdatePX_Y();
#else
		UpdatePX_Z();
#endif
		/*inference();

		//output paramters
		parameterLog << eexp(parameter.Epsilon) << "," << eexp(parameter.Pi) << "," << eexp(parameter.M) << ",";
		for (int m = 0; m < cNum; m++) {
			for (int n = 0; n < Dim; n++) {
				parameterLog << parameter.Mu[m][n] << ",";
			}
		}
		for (int t = 0; t < cNum; t++) {
			for (int m = 0; m < Dim; m++) {
				for (int n = 0; n < Dim; n++) {
					parameterLog << parameter.Sigma[t][m][n] << ",";
				}
			}
		}

		struct conMatrix CM = getConfusionMatrix();
		float precision = (float)CM.TT / (CM.TT + CM.FT);
		float accuracy = (float)(CM.TT + CM.FF) / (CM.TT + CM.FF + CM.FT + CM.TF);
		float recall = (float)CM.TT / (CM.TT + CM.TF);
		float fscore = (float)2 * precision *recall / (precision + recall);

		float precisionDry = (float)CM.FF / (CM.FF + CM.TF);
		float recallDry = (float)CM.FF / (CM.FF + CM.FT);
		float fscoreDry = (float)2 * precisionDry *recallDry / (precisionDry + recallDry);
		fscore = (fscore + fscoreDry) / 2.0;
		parameterLog << precision << "," << recall << "," << fscore << "," << accuracy << "," << CM.TT << "," << CM.TF << "," << CM.FT << "," << CM.FF << std::endl;
		*/ 
		// Print and Log
		{
			clock_t stop_s = clock();
			std::cout << endl << "Iteration: " << iterateTimes << "  Total CPU-Time: " << (stop_s - start_s) / float(CLOCKS_PER_SEC);
#ifdef INCLUDE_M
			std::cout << endl << "Epsilon: " << eexp(parameter.Epsilon) << "  Pi: " << eexp(parameter.Pi) << "  M: " << eexp(parameter.M);
#else
			std::cout << endl << "Epsilon: " << eexp(parameter.Epsilon) << "  Pi: " << eexp(parameter.Pi);
#endif
			for (int c = 0; c < cNum; c++) {
				std::cout << endl << "Mu" << c;
				for (size_t i = 0; i < Dim; i++) {
					cout << " " << parameter.Mu[c][i] << " ";
				}
			}
			for (int c = 0; c < cNum; c++) {
				cout << endl << "Sigma" << c;
				for (size_t i = 0; i < Dim; i++) {
					for (size_t j = 0; j < Dim; j++) {
						cout << " " << parameter.Sigma[c][i][j] << " ";
					}
				}
			}

		}

		cout<<endl;//flush

		//check stop criteria
		{
			bool MuConverge = true, SigmaConverge = true;
			double thresh = parameter.THRESHOLD;

			for (int c = 0; c < cNum; c++) {
				for (int i = 0; i < Dim; i++) {
					if (fabs((parameter.Mu[c][i] - MuOld[c][i]) / MuOld[c][i])  > thresh) {
						MuConverge = false;
						break;
					}

					for (int j = 0; j < Dim; j++) {
						if (fabs((parameter.Sigma[c][i][j] - SigmaOld[c][i][j]) / SigmaOld[c][i][j]) > thresh) {
							SigmaConverge = false;
							break;
						}
					}
				}
			}

			double epsilonRatio = fabs((eexp(parameter.Epsilon) - eexp(EpsilonOld)) / eexp(EpsilonOld));
			double PiRatio = fabs((eexp(parameter.Pi) - eexp(PiOld)) / eexp(PiOld));
#ifdef INCLUDE_M
			double MRatio = fabs((eexp(parameter.M) - eexp(MOld)) / eexp(MOld));
#endif

#ifdef INCLUDE_M
			if (epsilonRatio < thresh &&  PiRatio < thresh && MRatio < thresh && MuConverge && SigmaConverge) {
#else
			if (epsilonRatio < thresh &&  PiRatio < thresh && MuConverge && SigmaConverge) {
#endif
				iterator = false;
			}

			iterateTimes++;
			if (iterateTimes > parameter.maxIteratTimes) {
				iterator = false;
			}
		}

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		cout << "### Iteration time: " << elapsed_seconds.count() << " sec" << endl;

	} // end while
	parameterLog.close();

}


void cFlood::inference() {
	//std::cout << endl << "Inference:" << endl;
	//std::cout << endl << "Epsilon: " << eexp(parameter.Epsilon) << "  Pi: " << eexp(parameter.Pi);
	//for (int c = 0; c < cNum; c++) {
	//	std::cout << endl << "Mu" << c;
	//	for (size_t i = 0; i < Dim; i++) {
	//		cout << " " << parameter.Mu[c][i] << " ";
	//	}
	//}
	//for (int c = 0; c < cNum; c++) {
	//	cout << endl << "Sigma" << c;
	//	for (size_t i = 0; i < Dim; i++) {
	//		for (size_t j = 0; j < Dim; j++) {
	//			cout << " " << parameter.Sigma[c][i][j] << " ";
	//		}
	//	}
	//}

	//convert parameter Pi, M, Epsilon back to normal form
	double lnPi = parameter.Pi;
	double Pi = eexp(parameter.Pi);
	double lnEpsilon = parameter.Epsilon;
	double Epsilon = eexp(parameter.Epsilon);
#ifdef INCLUDE_M
	//double lnM = parameter.M; //Commented by Da: not used
	double M = eexp(parameter.M);
#endif
	//Calculate Gain
	bool parentsAllVisited = true;
	double curWaterProb, curDryProb, subTreeGain = 0, curMaxGain = 0;
	vector<double>chainMaxGain(tree.headPointers.size(), 0);

	infer.chainMaxGainFrontierIndex.resize(tree.headPointers.size());
	std::fill(infer.chainMaxGainFrontierIndex.begin(), infer.chainMaxGainFrontierIndex.end(), -1);

	//define temporary variables
	int curIdx = -1, newIdx = -1;
	Node* curNode = NULL;

	for (size_t i = 0; i < parameter.nonEmptySize; i++) {
		curIdx = tree.nodeLevelIndexPair[i].second;
		curNode = data.allNodes.at(curIdx);
		newIdx = data.old2newIdx.at(curIdx);

		if (data.missingIndicator.at(newIdx) == 0) { // missing data
			curDryProb = eln(0.5);
			curWaterProb = eln(0.5);
		}
		else {
			curDryProb = infer.dryLikelihood.at(newIdx) + infer.lnCoefficient[0]; // log form
			curWaterProb = infer.waterLikelihood.at(newIdx) + infer.lnCoefficient[1];
		}


		curNode->visited = true;
		parentsAllVisited = true;
#ifdef INCLUDE_M
		if (curWaterProb + log(1 - M) >= curDryProb + log(M)) {
			curWaterProb += log(1 - M);
		}
		else {
			curWaterProb = curDryProb + log(M);
		}
#endif
		if (curNode->parents.size() == 0) { // leaf node

			if (curNode->next != NULL) {
				if (curNode->next->parents.size() == 1) {
					curNode->curGain = curWaterProb - curDryProb + lnPi - log(1 - Pi) + lnEpsilon;
				}
				else {
					for (size_t a = 0; a < curNode->next->parents.size(); a++) {
						if (curNode->next->parents.at(a)->visited == false) {
							parentsAllVisited = false;
							break;
						}
					}
					if (parentsAllVisited == true) {
						curNode->curGain = curWaterProb - curDryProb + lnPi - log(1 - Pi) + lnEpsilon;
					}
					else {
						curNode->curGain = curWaterProb - curDryProb + lnPi - log(1 - Pi);
					}
				}
			}
			else {//false node which is seperated from other nodes, chain length = 1, may due to mosaic or data error
				curNode->curGain = 0;
			}

		}
		else { // internal node
			if (curNode->next != NULL) {
				if (curNode->next->parents.size() == 1) {
					curNode->curGain = curWaterProb - curDryProb + log(1 - Epsilon);
				}
				else {
					for (size_t a = 0; a < curNode->next->parents.size(); a++) {
						if (curNode->next->parents.at(a)->visited == false) {
							parentsAllVisited = false;
							break;
						}
					}
					if (parentsAllVisited == true) {
						curNode->curGain = curWaterProb - curDryProb + log(1 - Epsilon);
					}
					else {
						curNode->curGain = curWaterProb - curDryProb + log(1 - Epsilon) - lnEpsilon;
					}
				}
			}
			else { // root node
				curNode->curGain = curWaterProb - curDryProb + log(1 - Epsilon) - lnEpsilon;
			}
		}


		if (curNode->parents.size() == 0) { // bottom node, only treated as flood if Gain > 0
			if (curNode->curGain > 0) { // if curNode->curGain < 0, chainMaxGain remains 0
				chainMaxGain.at(curNode->nodeChainID) = curNode->curGain;
				infer.chainMaxGainFrontierIndex.at(curNode->nodeChainID) = curIdx;
				curMaxGain += curNode->curGain;
			}
		}
		else if (curNode->parents.size() == 1) {
			curNode->curGain += curNode->parents.at(0)->curGain;
			if (curNode->curGain > chainMaxGain.at(curNode->nodeChainID)) {
				curMaxGain += curNode->curGain - chainMaxGain.at(curNode->nodeChainID);
				chainMaxGain.at(curNode->nodeChainID) = curNode->curGain;
				infer.chainMaxGainFrontierIndex.at(curNode->nodeChainID) = curIdx;
			}
		}
		else if (curNode->parents.size() > 1) {
			subTreeGain = 0;
			for (size_t i = 0; i < curNode->parents.size(); i++) {
				curNode->curGain += curNode->parents.at(i)->curGain;
				subTreeGain += chainMaxGain.at(curNode->parents.at(i)->nodeChainID);
			}
			chainMaxGain.at(curNode->nodeChainID) = subTreeGain; //combine parents' curMaxGain to one chain

			if (curNode->curGain > subTreeGain) {
				curMaxGain += curNode->curGain - subTreeGain;
				chainMaxGain.at(curNode->nodeChainID) = curNode->curGain;
				infer.chainMaxGainFrontierIndex.at(curNode->nodeChainID) = curIdx;
			}
		}

	}
	updateMapPrediction();

};

void cFlood::updateMapPrediction() {
	mappredictions.resize(parameter.nonEmptySize);
	std::fill(mappredictions.begin(), mappredictions.end(), 1); // all water
	size_t rootId = tree.nodeLevelIndexPair[parameter.nonEmptySize - 1].second;
	std::queue<size_t> myqueue;
	myqueue.push(rootId);
	while (!myqueue.empty()) {
		//traverse current node
		size_t curId = myqueue.front();
		myqueue.pop();
		if (infer.chainMaxGainFrontierIndex[data.allNodes.at(curId)->nodeChainID] != curId) {
			mappredictions.at(data.old2newIdx.at(curId)) = 0; //Attention!
			for (int i = 0; i < data.allNodes.at(curId)->parents.size(); i++) {
				myqueue.push(data.allNodes.at(curId)->parents[i]->nodeIndex);
			}
		}
	}
}

void cFlood::output() {
	//std::ofstream outputIndexFile;
	//outputIndexFile.open(outputFile, std::ofstream::trunc);
	//size_t rootId = tree.nodeLevelIndexPair[parameter.nonEmptySize - 1].second;
	//std::queue<size_t> myqueue;
	//myqueue.push(rootId);
	//while (!myqueue.empty()) {
	//	//traverse current node
	//	size_t curId = myqueue.front();
	//	myqueue.pop();
	//	if (infer.chainMaxGainFrontierIndex[data.allNodes.at(curId)->nodeChainID] != curId) {
	//		outputIndexFile << curId << "\n";
	//		for (int i = 0; i < data.allNodes.at(curId)->parents.size(); i++) {
	//			myqueue.push(data.allNodes.at(curId)->parents[i]->nodeIndex);
	//		}
	//	}
	//}
	//outputIndexFile.close();

	std::ofstream outputIndexFile;
	std::string outputFile = HMTOutputLocation + HMTPrediction;
	outputIndexFile.open(outputFile, std::ofstream::trunc);
	for (int i = 0; i < mappredictions.size(); i++) {
		outputIndexFile << mappredictions.at(i) << "\n";
	}
	outputIndexFile.close();

};


int main(int argc, char *argv[]) {
	cFlood flood;
	flood.input(argc, argv);
}

struct conMatrix cFlood::getConfusionMatrix() {
	struct conMatrix confusionMatrix;
	confusionMatrix.TT = 0;
	confusionMatrix.TF = 0;
	confusionMatrix.FF = 0;
	confusionMatrix.FT = 0;

	for (int i = 0; i < testIndex.size(); i++) {
		if (testLabel[i] == 0 && mappredictions[data.old2newIdx.at(testIndex[i])] == 0) {
			confusionMatrix.FF++;
		}
		else if (testLabel[i] == 0 && mappredictions[data.old2newIdx.at(testIndex[i])] == 1) {
			confusionMatrix.FT++;
		}
		else if (testLabel[i] == 1 && mappredictions[data.old2newIdx.at(testIndex[i])] == 1) {
			confusionMatrix.TT++;
		}
		else if (testLabel[i] == 1 && mappredictions[data.old2newIdx.at(testIndex[i])] == 0) {
			confusionMatrix.TF++;
		}
	}

	return confusionMatrix;
}



