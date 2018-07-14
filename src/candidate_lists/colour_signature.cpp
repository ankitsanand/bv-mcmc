#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>
#include <thread>
#include <functional>
#include <random>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include "saucy.h"
#include "amorph.h"
#include "util.h"
#include "platform.h"
#include "saucyio.c"

using std::string;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::getline;
using std::map;
using std::pair;
using std::make_pair;
using std::set;

string saucy_output;
// saucy variable for handling time out
static sig_atomic_t timeout_flag = 0;

// Method for running a parallel for loop
// Source: https://stackoverflow.com/questions/36246300/parallel-loops-in-c
static
void parallel_for(unsigned nb_elements,
                  std::function<void (int start, int end)> functor,
                  bool use_threads = true){

    unsigned nb_threads_hint = std::thread::hardware_concurrency();
    //unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);
    unsigned nb_threads = 4;
    unsigned batch_size = nb_elements / nb_threads;
    unsigned batch_remainder = nb_elements % nb_threads;

    std::vector< std::thread > my_threads(nb_threads);

    if( use_threads )
    {
        // Multithread execution
        for(unsigned i = 0; i < nb_threads; ++i)
        {
            int start = i * batch_size;
            my_threads[i] = std::thread(functor, start, start+batch_size);
        }
    }
    else
    {
        // Single thread execution (for easy debugging)
        for(unsigned i = 0; i < nb_threads; ++i){
            int start = i * batch_size;
            functor( start, start+batch_size );
        }
    }

    // Deform the elements left
    int start = nb_threads * batch_size;
    functor( start, start+batch_remainder);

    // Wait for the other thread to finish their task
    if( use_threads )
        std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}

// Function to tokenize a string
void Tokenize(
	const string& str,
	vector<string>& tokens,
	const string& delimiters = " "){

	tokens.clear();
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

// Class to define a clause in a PGM
// A clause is a simply a set of variable assignments and an associated weight
// A PGM is simply a set of clauses
class Clause
{
public:
	vector< pair<int,int> > variable_assignments;
	map<int, int> assignment_map;
	double weight;

	Clause(
		vector< pair<int,int> > assgn,
		double w): weight(w){

		for (int i=0;i<assgn.size();i++) {
			variable_assignments.push_back(assgn[i]);
			assignment_map[assgn[i].first] = assgn[i].second;
		}
	};

	bool is_valid(
		vector< pair<int,int> > partial_assgn){

		for (int i=0;i<partial_assgn.size();i++){
			if (assignment_map.find(partial_assgn[i].first)!=assignment_map.end()){
				if (assignment_map[partial_assgn[i].first] == partial_assgn[i].second) return true;
			}
		}
		return false;
	}

	string as_string(){
		
		string out, temp1, temp2;
		for (int i=0;i<variable_assignments.size();i++){
			temp1 = std::to_string(variable_assignments[i].first);
			temp2 = std::to_string(variable_assignments[i].second);
			out += temp1+": "+temp2+", ";
		}
		out = out+"\n";
		return out;
	}


	double evaluate_clause(
		vector< pair<int,int> > partial_assgn){

		for (int i=0;i<partial_assgn.size();i++){
			if (assignment_map.find(partial_assgn[i].first)!=assignment_map.end()){
				if (assignment_map[partial_assgn[i].first] == partial_assgn[i].second) return weight;
			}
		}
		return 0.0;
	}
};

// Function to compare two clauses by weight
bool compare_clauses(
	Clause const & a,
	Clause const & b){

	return (a.weight<b.weight);
}

// Helper function to compare two vectors by size
bool compare_vectors(
	vector< pair<int, int> >& a,
	vector< pair<int, int> >& b){

	return(a.size() > b.size());
}

// Modification of an existing saucy function for handling output
static void
amorph_automorphism_string(
	int n,
	const int *gamma,
	int nsupp,
	const int *support,
	struct amorph_graph *g,
	char *marks){

	int i, j, k;
	std::stringstream ss;
	/* We presume support is already sorted */
	for (i = 0; i < nsupp; ++i) {
		k = support[i];

		/* Skip elements already seen */
		if (marks[k]) continue;

		/* Start an orbit */
		marks[k] = 1;
		ss << "("<<(k+1);
		//printf("(%d", k+1);

		/* Mark and notify elements in this orbit */
		for (j = gamma[k]; j != k; j = gamma[j]) {
			marks[j] = 1;
			ss << ", "<<(j+1);
			//printf(", %d", j+1);
		}

		/* Finish off the orbit */
		ss << ")";
		//printf(")");
	}
	ss << ",\n";
	//printf(",\n");
	
	saucy_output += ss.str();
	/* Clean up after ourselves */
	for (i = 0; i < nsupp; ++i) {
		marks[support[i]] = 0;
	}

}

// Modification of an existing saucy function to handle string output
static int
string_on_automorphism(
	int n,
	const int *gamma,
	int k,
	int *support,
	void *arg,
	char *marks){

	struct amorph_graph *g = (amorph_graph*) arg;
	qsort_integers(support, k);
	g->consumer(n, gamma, k, support, g, marks);
	return !timeout_flag;
}

// Function to generate a saucy graph object from an existing graph object
struct amorph_graph* amorph_graph_generator(
	int nodes,
	int edges,
	int colours, 
	vector<int>& filtered_colour_splits,
	vector<pair<int, int>>& edge_list){

	int saucy_i, saucy_j, saucy_k, saucy_n, saucy_e, saucy_p;
	int *saucy_aout, *saucy_eout, *saucy_ain, *saucy_ein, *saucy_colors;
	struct amorph_graph *saucy_g = NULL;

	saucy_n = nodes;
	saucy_e = edges;
	saucy_p = colours;
	// Allocate memory
	saucy_g = (amorph_graph*) malloc(sizeof(struct amorph_graph));
	saucy_aout = (int*) calloc(saucy_n+1, sizeof(int));
	saucy_eout = (int*) malloc(2 * saucy_e * sizeof(int));
	saucy_colors = (int*) malloc(saucy_n * sizeof(int));
	if (!saucy_g || !saucy_aout || !saucy_eout || !saucy_colors) return NULL;

	saucy_g->sg.n = saucy_n;
	saucy_g->sg.e = saucy_e;
	saucy_g->sg.adj = saucy_aout;
	saucy_g->sg.edg = saucy_eout;
	saucy_g->colors = saucy_colors;

	saucy_ain = saucy_aout;
	saucy_ein = saucy_eout;

	/* Initial coloring with provided splits */
	for (saucy_i = saucy_j = 0; saucy_i < saucy_p - 1; ++saucy_i) {
		saucy_k = filtered_colour_splits[saucy_i];
		while (saucy_j < saucy_k) {
			saucy_colors[saucy_j++] = saucy_i;
		}
	}
	while (saucy_j < saucy_n) {
		saucy_colors[saucy_j++] = saucy_i;
	}

	/* Count the size of each adjacency list */
	for (saucy_i = 0; saucy_i < saucy_e; ++saucy_i) {
		saucy_j = edge_list[saucy_i].first;
		saucy_k = edge_list[saucy_i].second;
		++saucy_aout[saucy_j]; ++saucy_ain[saucy_k];
	}

	init_fixadj1(saucy_n, saucy_aout);

	// Add edges from the adjacency list
	for (saucy_i = 0; saucy_i < saucy_e; ++saucy_i) {
		saucy_j = edge_list[saucy_i].first;
		saucy_k = edge_list[saucy_i].second;

		/* Simple input validation: check vertex values */
		if (saucy_j >= saucy_n || saucy_j < 0) {
			warn("invalid vertex in input: %d", saucy_j);
			return NULL;
		}
		if (saucy_k >= saucy_n || saucy_k < 0) {
			warn("invalid vertex in input: %d", saucy_k);
			return NULL;
		}

		saucy_eout[saucy_aout[saucy_j]++] = saucy_k;
		saucy_ein[saucy_ain[saucy_k]++] = saucy_j;
	}

	init_fixadj2(saucy_n, 2 * saucy_e, saucy_aout);

	/* Check for duplicate edges */
	if (dupe_check(saucy_n, saucy_aout, saucy_eout)) return NULL;
	saucy_g->consumer = amorph_automorphism_string;
	saucy_g->free = amorph_graph_free;
	saucy_g->stats = NULL;
	return saucy_g;
}

// Function to generate an auxilliary graph from a PGM to run graph 
// isomorphism on
pair<string, string> generate_auxilliary_graph(
	int variables,
	vector <pair<int, int> >& possible_block,
	vector<Clause>& features,
	vector< vector<int> >& nodes_to_features){

	// Initialize statistics about the graph
	int nodes=0, edges=0, colours=0;
	int covered_nodes[variables];
	vector<int> colour_splits; // Nodes where new colours begin in the graph
	std::memset(covered_nodes, 0, sizeof(covered_nodes));
	for (pair<int, int> block : possible_block){
		covered_nodes[block.first] = 1;
		covered_nodes[block.second] = 1;
	}

	pair<int, int> node_convention[4]; // Array to simplify assignment to blocks
	node_convention[0] = make_pair(0,0);
	node_convention[1] = make_pair(0,1);
	node_convention[2] = make_pair(1,0);
	node_convention[3] = make_pair(1,1);

	vector<int> node_type, sentinel_blocks;
	vector<pair<int, int>> node_to_variables;
	vector<pair<int, int>> edge_list; // All edges in the graph
	/* Node type convention:
	0 for sentinel nodes of Blocks
	1 for sentinel nodes of Variables
	2,3,4,5 for 00,01,10,11 for BV pairs
	6,7 for 0,1 for VV pairs
	8 for clause nodes
	*/
	int bv_nodes_begin, vv_nodes_begin, bv_pairs_begin, vv_pairs_begin;
	int clauses_begin;

	// Add all sentinel nodes for BV pairs to the graph
	bv_nodes_begin = 0;
	colour_splits.push_back(nodes); // The colour of BV nodes starts from 0
	for (pair<int, int> block : possible_block){
		node_to_variables.push_back(block);
		node_type.push_back(0);
		sentinel_blocks.push_back(-1); // These nodes are themselves sentinel
		nodes++;
	}
	// nodes denotes the next empty node_number in the graph
	// Add all sentinel nodes for VV pairs to the graph
	vv_nodes_begin = nodes;
	colour_splits.push_back(nodes);
	for (int i=0;i<variables;i++){
		if (!covered_nodes[i]){
			node_to_variables.push_back(make_pair(i, -1));
			node_type.push_back(1);
			sentinel_blocks.push_back(-1);
			nodes++;
		}
	}

	// Add all BV pairs to the graph
	bv_pairs_begin = nodes;
	colour_splits.push_back(nodes);
	for (int i=bv_nodes_begin;i<vv_nodes_begin;i++){
		for (int j=0;j<4;j++){
			node_type.push_back(j+2);
			sentinel_blocks.push_back(i);
			edge_list.push_back(make_pair(i,nodes));
			nodes++;
		}
	}

	// Add all VV pairs to the graph
	vv_pairs_begin = nodes;
	colour_splits.push_back(nodes);
	for (int i=vv_nodes_begin;i<bv_pairs_begin;i++){
		for (int j=0;j<2;j++){
			node_type.push_back(j+6);
			sentinel_blocks.push_back(i);
			edge_list.push_back(make_pair(i,nodes));
			nodes++;
		}
	}

	// Add all clauses to the graph
	clauses_begin = nodes;
	int current_colour = 4;
	map<double,int> weight_to_colour;
	for (int i=0;i<features.size();i++){
		if (weight_to_colour.find(features[i].weight) == weight_to_colour.end()){
			weight_to_colour[features[i].weight] = current_colour;
			current_colour++;
			colour_splits.push_back(nodes);
			current_colour++;
		}
		nodes++;
	}

	// blocks_string contains the mapping which would be given as input to GAP
	// Maps block nodes and assignments to node numbers
	string blocks_string = "[\n";
	// Adding edges between BV pairs and clauses
	for (int i=bv_pairs_begin;i<vv_pairs_begin;i++){
		// Iterate over all blocks and add edges to each feature which is on
		// Partial assignment is the assignment of all variables in the clause
		vector<pair<int,int>> partial_assignment;
		partial_assignment.push_back(make_pair(node_to_variables[sentinel_blocks[i]].first, node_convention[node_type[i]-2].first));
		partial_assignment.push_back(make_pair(node_to_variables[sentinel_blocks[i]].second, node_convention[node_type[i]-2].second));
		/********** String manipulations for creating blocks_string ***********/ 
		blocks_string += "[ [ [";
		blocks_string += std::to_string(node_to_variables[sentinel_blocks[i]].first);
		blocks_string += ",";
		blocks_string += std::to_string(node_to_variables[sentinel_blocks[i]].second);
		blocks_string += "], ";
		blocks_string += "[";
		blocks_string += std::to_string(node_convention[node_type[i]-2].first);
		blocks_string += ",";
		blocks_string += std::to_string(node_convention[node_type[i]-2].second);
		blocks_string += "] ], ";
		blocks_string += std::to_string(i+1); // Saucy outputs graphs with nodes numbers incremented
		blocks_string += " ],\n";
		/****************** Ending string manipulation ************************/
		set<int> current_features(nodes_to_features[node_to_variables[sentinel_blocks[i]].first].begin(),
			nodes_to_features[node_to_variables[sentinel_blocks[i]].first].end());
		current_features.insert(nodes_to_features[node_to_variables[sentinel_blocks[i]].second].begin(),
			nodes_to_features[node_to_variables[sentinel_blocks[i]].second].end());

		for (auto it = current_features.begin();it != current_features.end(); ++it){
			if (features[*it].is_valid(partial_assignment)) {
				edge_list.push_back(make_pair(i, *it + clauses_begin));
			}
		}
	}

	// Add edges for VV pairs and clauses
	for (int i=vv_pairs_begin;i<clauses_begin;i++){
		// Iterate over all VV pairs and add edges between VV pairs and features
		vector<pair<int,int>> partial_assignment;
		partial_assignment.push_back(make_pair(node_to_variables[sentinel_blocks[i]].first, node_type[i]-6));
		// set<int> current_features(nodes_to_features[node_to_variables[sentinel_blocks[i]].first].begin(),
		// 	nodes_to_features[node_to_variables[sentinel_blocks[i]].first].end());
		/********** String manipulations for creating blocks_string ***********/
		blocks_string += "[ [ [";
		blocks_string += std::to_string(node_to_variables[sentinel_blocks[i]].first);
		blocks_string += "], ";
		blocks_string += "[";
		blocks_string += std::to_string(node_type[i]-6);
		blocks_string += "] ], ";
		blocks_string += std::to_string(i+1); // Saucy outputs graphs with nodes numbers incremented
		blocks_string += " ],\n";
		/****************** Ending string manipulation ************************/
		for(int j=0;j<nodes_to_features[node_to_variables[sentinel_blocks[i]].first].size();j++){
			if (features[nodes_to_features[node_to_variables[sentinel_blocks[i]].first][j]].is_valid(partial_assignment)){
				edge_list.push_back(make_pair(i, nodes_to_features[node_to_variables[sentinel_blocks[i]].first][j] + clauses_begin));
			}
		}
	}
	
	blocks_string += "]\n";

	// Need to delete repeated colours in colour_splits
	vector<int> filtered_colour_splits;
	for (int i=1;i<colour_splits.size();i++){
		if (colour_splits[i-1] != colour_splits[i]){
			filtered_colour_splits.push_back(colour_splits[i]);
		}
	}
	colours = filtered_colour_splits.size() + 1;
	edges = edge_list.size();
	/***************************************************/
	/* Creating saucy graph object					   */
	/***************************************************/
	struct amorph_graph *saucy_g = NULL;
	// Initialize variables specific to saucy
	// These are named saucy_* to avoid confusion with variables already used
	saucy_g = amorph_graph_generator(nodes, edges, colours,
			 filtered_colour_splits, edge_list);
	int saucy_n;
	saucy_n = saucy_g->sg.n;
	struct saucy *saucy_s;
	char* marks;
	saucy_s = saucy_alloc(saucy_n);
	marks = (char*) calloc(saucy_n, sizeof(char));
	struct saucy_stats stats;
	saucy_search(saucy_s, &saucy_g->sg, 0, saucy_g->colors, string_on_automorphism, saucy_g, &stats, marks);
	cout<<"Finished running saucy\n";
	string permutation_output = saucy_output; 

	return make_pair(blocks_string, permutation_output);
	
}

// Check if two pairs of ints are conflicting or not
bool in_conflict(
	pair<int,int>& p1,
	pair<int,int>& p2){

	if ((p1.first == p2.first) || (p1.first == p2.second)) return true;
	if ((p1.second == p2.first) || (p1.second == p2.second)) return true;
	return false;
}

// Try to fit a node in a list of candidates
void try_to_fit(
	int nodes,
	int starting_index, 
	vector< vector< pair<int,int> > >& final_list, 
	pair<int, int>& block,
	vector< vector<int> >& covered_nodes_in_final_list){

	if (starting_index == final_list.size()){
		vector< pair<int,int> > new_list;
		new_list.push_back(block);
		final_list.push_back(new_list);
		vector< int > covered_by_this(nodes);
		covered_by_this[block.first] = 1;
		covered_by_this[block.second] = 1;
		covered_nodes_in_final_list.push_back(covered_by_this);
		return;
	}
	for (int i=starting_index;i<final_list.size();i++){
		bool found = false;
		found = covered_nodes_in_final_list[i][block.first] || covered_nodes_in_final_list[i][block.second];
		if (!found){
			final_list[i].push_back(block);
			covered_nodes_in_final_list[i][block.first] = 1;
			covered_nodes_in_final_list[i][block.second] = 2;
			return;
		}
	}
	vector< pair<int,int> > new_list;
	new_list.push_back(block);
	final_list.push_back(new_list);
	vector< int > covered_by_this(nodes);
	covered_by_this[block.first] = 1;
	covered_by_this[block.second] = 1;
	covered_nodes_in_final_list.push_back(covered_by_this);
	return;
}

// Converts a list of weights into a string signature
string weights_to_signature(
	vector<double>& active_weights){

	sort(active_weights.begin(), active_weights.end());
	if (active_weights.size()==0){
		return " ";
	}
	string out;
	double last_weight_seen = active_weights[0];
	int count_of_current_weight = 0;
	for (int i=0;i<active_weights.size();i++){
		if (last_weight_seen == active_weights[i]){
			count_of_current_weight ++;
		}
		else{
			std::stringstream ss;
			ss << last_weight_seen;
			out += ss.str();
			out += ":";
			out += std::to_string(count_of_current_weight);
			out += " ";
			last_weight_seen = active_weights[i];
			count_of_current_weight = 1;
		}
	}
	// Adding the last seen weight to the string
	std::stringstream ss;
	ss << last_weight_seen;
	out += ss.str();
	out += ":";
	out += std::to_string(count_of_current_weight);

	return out;
}

// Random heuristic for generating candidate lists
// Used for sanity checking with the actual heuristic
void get_random_lists(
	int nodes,
	vector<Clause> &features,
	vector< vector<int> >& nodes_to_features,
	vector< vector< pair<int, int> > > &final_list,
	int num_lists){

	for (int i=0;i<features.size();i++){
		for (int j=0;j<features[i].variable_assignments.size();j++){
			nodes_to_features[features[i].variable_assignments[j].first].push_back(i);
		}
	}

	vector<int> current_nodes(nodes);
	for (int i=0;i<nodes;i++) current_nodes[i] = i;
	// Genarating random lists
	for (int i=0;i<num_lists;i++){
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(current_nodes.begin(), current_nodes.end(), g);
		vector< pair<int, int> > current_list;
		// Add all pairs to the list
		for (int j=0;j < current_nodes.size() - 1;j+=2){
			current_list.push_back(make_pair(current_nodes[j], current_nodes[j+1]));
		}
		final_list.push_back(current_list);
	}

}

// Heuristic to generate candidate lists
void get_candidate_lists(
	int nodes,
	vector<Clause> &features,
	vector< vector<int> >& nodes_to_features,
	vector< vector< pair<int, int> > > &final_list){
	
	final_list.clear();
	vector< pair<int, int> > blocks_list;
	vector< vector<int> > current_nodes(nodes);
	// Mapping nodes to which features are present for this node
	for (int i=0;i<features.size();i++){
		vector<int> current_variables;
		for (int j=0;j<features[i].variable_assignments.size();j++){
			nodes_to_features[features[i].variable_assignments[j].first].push_back(i);
			current_variables.push_back(features[i].variable_assignments[j].first);
		}
		// Iterate through current_variables and add all blocks in the graph
		sort(current_variables.begin(), current_variables.end());
		for (int j=0;j<current_variables.size();j++){
			for (int k=j+1;k<current_variables.size();k++){
				current_nodes[current_variables[j]].push_back(current_variables[k]);
			}
		}
	}
	// Iterate over current_nodes to get the blocks
	for (int i=0;i<nodes;i++){
		if (current_nodes[i].size()==0){
			//cout<<"Empty vector for: "<<i<<endl;
			continue;
		}
		sort(current_nodes[i].begin(), current_nodes[i].end());
		for (int j=0;j<current_nodes[i].size()-1;j++){
			if (current_nodes[i][j] != current_nodes[i][j+1]){
				blocks_list.push_back(make_pair(i, current_nodes[i][j]));
			}
		}
		blocks_list.push_back(make_pair(i, current_nodes[i][ current_nodes[i].size() - 1 ]));
	}
	cout<<"Size of blocks list is: "<<blocks_list.size()<<endl;
	// Algorithm for generating blocks:
	// Iterate over all BV pairs and all features which are present in the 
	// Markov Blanket of both the variables combined
	map< string, vector<int> > signature_mapping;
	// signature_mapping is a map from strings to a list of block value pairs
	// which have this signature
	// The key is a string of the form: "feature_weight:Number "...

	cout<<"Number of blocks is: "<<blocks_list.size()<<endl;
	// Covered blocks would be used later to get the candidate lists in the 
	// graph.
	vector<pair<int,int>> node_convention;
	node_convention.push_back(make_pair(0,0));
	node_convention.push_back(make_pair(0,1));
	node_convention.push_back(make_pair(1,0));
	node_convention.push_back(make_pair(1,1));

	// Create colour signatures
	// block number is an integer = 4*i + j
	for (int i=0;i<blocks_list.size();i++){
		// Get the list of features which are in the Markov Blanket of these
		// variables. Iterate over all the features and get the candidate lists
		set<int> current_features(nodes_to_features[blocks_list[i].first].begin(),nodes_to_features[blocks_list[i].first].end());
		current_features.insert(nodes_to_features[blocks_list[i].second].begin(),nodes_to_features[blocks_list[i].second].end());
		vector<pair<int,int>> partial_assignment;
		partial_assignment.push_back(make_pair(blocks_list[i].first, node_convention[0].first));
		partial_assignment.push_back(make_pair(blocks_list[i].second, node_convention[0].second));
		for (int j=0;j<4;j++){
			partial_assignment[0].second = node_convention[j].first;
			partial_assignment[1].second = node_convention[j].second;
			vector<double> current_weights;
			for (auto it = current_features.begin();it != current_features.end(); ++it){
				if (features[*it].is_valid(partial_assignment)) {
					current_weights.push_back(features[*it].weight);
				}
			}
			string signature = weights_to_signature(current_weights);
			signature_mapping[signature].push_back(4*i + j);
		}
	}
	cout<<"Number of distinct colours is: "<<signature_mapping.size()<<endl;
	vector< vector< pair<int, int> > >  possible_candidates;
	for (auto it = signature_mapping.begin(); it != signature_mapping.end(); ++it){
		if (it->second.size()==1) continue;
		else{
			vector< pair<int, int> > current_candidate;
			int covered_blocks[blocks_list.size()];
			std::memset(covered_blocks, 0, sizeof(covered_blocks));
			// Iterate over all integers present in this block
			// Add their respective blocks to current_candidates
			for (int i=0;i<it->second.size();i++){
				int current_block = it->second[i]/4;
				if (!covered_blocks[current_block]){
					current_candidate.push_back(blocks_list[current_block]);
					covered_blocks[current_block] = 1;
				}
			}
			possible_candidates.push_back(current_candidate);
		}
	}
	cout<<"Finished cleaning lists\n";
	sort(possible_candidates.begin(), possible_candidates.end(), compare_vectors);
	cout<<"Done sorting lists\n";
	vector< vector<int> > covered_nodes_in_final_list; 
	// Iterate over possible_candidates
	// Each element can contribute at most #variables candidate lists
	int num_iterations = possible_candidates.size();
	for (int i=0;i<possible_candidates.size() && final_list.size() < 3*nodes;i++){
		int starting_index = final_list.size();
		// The current list must have some elements left
		// And total elements contributed must be less than nodes
		// int iter = possible_candidates[i].size();
		//iter = std::min(iter, 100*nodes);
		for (int j=0;j<possible_candidates[i].size();j++){
			try_to_fit(nodes, 0, final_list, possible_candidates[i][j], covered_nodes_in_final_list);
		}
	}
	cout<<"Final size of list is: "<<final_list.size()<<endl;
}

// Reads a PGM file, generates candidate lists and writes them to file
void create_graph(
	string file, 
	int num_candidate_lists){

	// Reads a file and creates a graph based on the PGM defined in the file
	// blocks is a vector of vectors which define the blocks present in the 
	// graph. In case any integer is not present in this vector, it is added
	// as a single node in the graph.
	ifstream infile;
	infile.open(file);
	if (!infile.is_open()){
		std::cerr<<"Error opening file\n";
		return;
	}
	std::cerr<<"File loaded\n";
	string variable_names, temp;
	getline(infile, variable_names);
	variable_names = variable_names.substr(5);
	vector<string> variable_name_vector;
	Tokenize(variable_names, variable_name_vector, ",");
	// Now create a hash map of variable names and assign some numbers for your notation
	map<string, int> mapping;
	for (int i=0;i<variable_name_vector.size();i++) mapping[variable_name_vector[i]] = i;
	// Now read the file completely.
	getline(infile, temp);
	// Next n lines contain the domains of the variables
	// For now, we ignore these lines, since our domains is assumed to be binary
	int nodes = variable_name_vector.size();
	for (int i=0;i<nodes;i++) getline(infile, temp);
	getline(infile, temp);
	// Next lines contain the clauses in the graphical model
	// Each clause is defined over some variables and has a weight associated
	// How should we store each clause? Define a class for a clause. 
	// Each clause has a weight and a vector of pairs which denote 
	// whether a particular feature is true in on or off state.
	vector<Clause> features;
	double weight;
	string first, second, variable_num, parity;
	string::size_type location;
	vector<string> tokens;
	int v_num, par;
	while(getline(infile, temp)) {
		// Weights, features are separated by tab
		location = temp.find_last_of("\t");
		first = temp.substr(0,location);
		weight = atof(first.c_str()); // Get the weight for the clause
		second = temp.substr(location+1);
		Tokenize(second, tokens, ","); // Need to separate this by comma
		vector< pair<int,int> > v_assgn;
		for (int i=0;i<tokens.size();i++){
			location = tokens[i].find_last_of(" ");
			variable_num = tokens[i].substr(0,location);
			parity = tokens[i].substr(location+1);
			v_num = atoi(variable_num.c_str());
			par = atoi(parity.c_str());
			pair<int, int> feature(v_num, par);
			v_assgn.push_back(feature);
		}
		features.push_back(Clause(v_assgn, weight));
	}
	infile.close();
	sort(features.begin(), features.end(), compare_clauses);
	// Clauses are generated. Now, need to create auxilliary graph corresponding
	// to blocks.
	// First, for each node, first add all the variables present as blocks to 
	// the graph. Then add all variables not covered in blocks as single nodes to 
	// the graph. Then add BV pairs corresponding to each block
	// Then add nodes for clauses
	// Define edges between sentinel nodes and BV pairs 
	// Add edges between BV pairs and clauses.
	// Creating a new graph for the first variation
	vector< vector< pair<int, int> > > possible_blocks;
	string blocks_out, permutations_out;
	blocks_out += "node_list := [\n";
	std::cerr<<"Starting to find candidate lists\n";
	vector< vector<int> > nodes_to_features(nodes);
	get_candidate_lists(nodes, features, nodes_to_features, possible_blocks);
	//get_random_lists(nodes, features, nodes_to_features, possible_blocks, 12);
	int max_candidates = possible_blocks.size();
	max_candidates = std::min(max_candidates, num_candidate_lists);
	for (int candidate = 0;candidate<max_candidates;candidate++){
		saucy_output = "";
		cout<<"********************* Starting candidate: "<<candidate<<endl;
		pair<string, string> t = generate_auxilliary_graph(nodes, 
			possible_blocks[candidate], features, nodes_to_features);
		if (!t.second.empty()){
			// Remove the last comma in the list of generators
			t.second.pop_back();
			t.second.pop_back();
			blocks_out += t.first;
			blocks_out += ',';
			permutations_out += "g := Group( \n";
			permutations_out += t.second;
			permutations_out += "\n";
			permutations_out +=");;\n";
			permutations_out += "prpl := ProductReplacer(g);;\n";
			permutations_out += "Next(prpl);;\n";
			permutations_out += "Add(replacers, prpl);;\n";
		}
		else{
			cout<<"Empty string\n"; 
		}
	}
	// A parallelized implementation

	// parallel_for(possible_blocks.size(), [&](int start, int end){ 
 	//		for(int i = start; i < end; ++i){
	// 		saucy_output = "";
	// 		cout<<"********************* Starting candidate: "<<i<<endl;
	// 		pair<string, string> t = generate_auxilliary_graph(nodes, 
	// 			possible_blocks[i], features, nodes_to_features);
	// 		cout<<t.first<<endl;
	// 		cout<<t.second<<endl;
	// 		if (!t.second.empty()){
	// 			blocks_out += t.first;
	// 			blocks_out += ',';
	// 			permutations_out += "g := Group( \n";
	// 			permutations_out += t.second;
	// 			permutations_out +=");;\n";
	// 			permutations_out += "prpl := ProductReplacer(g);;\n";
	// 			permutations_out += "Next(prpl);;\n";
	// 			permutations_out += "Add(replacers, prpl);;\n";
	// 		}
	// 		else{
	// 			cout<<"Empty string\n"; 
	// 		}
	//    	}
 	//    } );

	blocks_out += "];;\n";
	ofstream permutations_file;
	permutations_file.open("permutations.g");
	if (permutations_file.is_open()){
		permutations_file<<permutations_out;
		permutations_file.close();
	}
	else cout<<"Unable to open file";
	ofstream blocks_file;
	blocks_file.open("blocks.g");
	if (blocks_file.is_open()){
		blocks_file<<blocks_out;
		blocks_file.close();
	}
	else cout<<"Unable to open file";

}

int main(int argc, char** argv) {
	//string file_name(argv[1]);
	int num_candidate_lists;
	if (argv[1]){
		num_candidate_lists = atoi(argv[1]);
		cout<<"Number of candidate lists is: "<<num_candidate_lists<<endl;
	}
	else{
		num_candidate_lists = 10000;
		cout<<"No restriction on number of candidate lists"<<endl;
	}
	int conc = std::thread::hardware_concurrency();
	cout<<"Number of threads supported: "<<conc<<endl;
	create_graph("test", num_candidate_lists);
	return 0;
}
