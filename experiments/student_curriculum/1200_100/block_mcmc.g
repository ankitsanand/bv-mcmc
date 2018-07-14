LoadPackage("orb");;

runtimeSum := function()
	local t;;
	t := Runtimes();;
	return t.user_time + t.user_time_children;;
end;;
iterations := 20;;  # Number of Iterations
num_sample := 450000;;
inter_sz := 1000;;	# Interval size when kl is computed
take_all_till := 1;;	# Computes marginal for all samples till ___
alpha := 0.02;; # The probability with which we take an orbital step

dom_size := [];;    # Domain size of each variable
num := -1;;         # Number of variables
feat := [];;        # A list of features. Each feature is given as (w, clause) where clause is a list of var-val pairs (var,val)
var_to_feat := [];;	# For each variable, has list of features containing that variable

state := [];;   # A list containing assignment of each variable

counts := [];;	# For each variable we have the count for each value in it's domain
marginals := [];;	# For each variable we have the count for each value in it's domain

# smoothing constant for computing KL-divergence is 0.000000001
smooth := 0.000000001;;

# load the random source
rs1 := RandomSource(IsMersenneTwister);;#, runtimeSum());;

Read("marginals.g");

start_time := runtimeSum();;

# Add the exec part of the code here
Exec( "./colour_signature 10" );;

Read("blocks.g");;
# blocks.g initialises a list, node_list
# i th index of node_list contains the mapping (from blocks to node numbers in
# the graph) for the ith permutation found

replacers := [];;
# replacers is a list of product replacer objects
# ith element of replacers is the product replacer for the ith permutation

Read("permutations.g");;
# replacers.g defines groups and initialises product replacer objects
# and adds them to replacers

nodes_to_states := [];;
# nodes_to_states is a list of dictionaries
# ith element of this list is a dictionary which maps nodes to states in the 
# ith permutation.

# A node is simply an integer corresponding to the node number in the graph
# A state is a list of the for [[variables], [assignments]]
# Eg: state could be [ [ 1, 99 ], [ 0, 0 ] ] which means that the variable
# numbers 1 and 99 have assignment 0 and 0 respectively 

states_to_nodes := [];;
# states_to_nodes is a list of dictionaries
# ith element of this list is a dictionaory which maps states to nodes in the 
# ith permutation

blocks_list := [];;
# blocks_list[i] is a list of blocks in the ith permutation
# Eg: ith element would look like [[1,2], [3]]
# This means that there are two blocks, one with two variables and one with
# a single variable

bv_start_index := [];;
vv_start_index := [];;
vv_start_number := [];;
first_block_number := [];;
last_block_number := [];;

for i in [1..Length(node_list)] do
	#current1 := NewDictionary(node_list[1][1][1], true);;
	#current2 := NewDictionary(node_list[1][1][2], true);;
	current_blocks := [];;
	blocks_seen := NewDictionary(node_list[1][1][1][1], false);;
	if (Length(node_list[i][1][1][1]) =2) then
		Add(bv_start_index, node_list[i][1][2]);
	else
		Add(bv_start_index, -1);
	fi;;
	found_vv := false;;

	for j in [1..Length(node_list[i])] do
		#AddDictionary(current1, node_list[i][j][1], node_list[i][j][2]);
		#AddDictionary(current2, node_list[i][j][2], node_list[i][j][1]);
		if not( KnowsDictionary(blocks_seen, node_list[i][j][1][1]) ) then
			Add(current_blocks, node_list[i][j][1][1]);;
			AddDictionary(blocks_seen, node_list[i][j][1][1]);;
			if (not(found_vv) and Length(node_list[i][j][1][1]) = 1) then
				Add(vv_start_index, node_list[i][j][2]);;
				Add(vv_start_number, Length(current_blocks));;
				found_vv := true;;
			fi;;
		fi;;
	od;;
	if (not(found_vv)) then 
		Add(vv_start_index, node_list[i][Length(node_list[i])][2] + 1);;
		Add(vv_start_number, Length(current_blocks)+1);;
		found_vv := true;;
	fi;;
	#Add(states_to_nodes, current1);;
	#Add(nodes_to_states, current2);;
	Add(blocks_list, current_blocks);;
	Add(last_block_number, node_list[i][Length(node_list[i])][2]);;
	Add(first_block_number, node_list[i][1][2]);;
od;;

sym_discovery_time := (runtimeSum() -start_time)/1000.0;;
Print("Sym discovery time: ", sym_discovery_time, "sec \n");;
get_graph_from_state := function(permutation_number)
	# states_to_nodes[i]: Dictionary from [block, values] to node num
	# nodes_to_states[i]: Dictionary from node num to [block, values]
	local current_state, current_assignment;
	current_state := Set([]);;
	# Iterate over all blocks in the list of blocks and keep checking which 
	# node number in the graph this corresponds to and add to the list
	for i in [1..Length(blocks_list[permutation_number])] do
		current_assignment := [];
		for j in [1..Length(blocks_list[permutation_number][i])] do 
			Add(current_assignment, state[ blocks_list[permutation_number][i][j] +1 ]);;
		od;
		Add(current_state, LookupDictionary(states_to_nodes[permutation_number],
			[ blocks_list[permutation_number][i], current_assignment ]));;
	od;;
	return current_state;;
end;;

get_graph_from_state_2 := function(permutation_number)
	local current_state, current_assignment;
	current_state := Set([]);;
	for i in [1..Length(blocks_list[permutation_number])] do
		current_assignment := 0;;
		for j in [1..Length(blocks_list[permutation_number][i])] do
			current_assignment := 2*current_assignment + state[ blocks_list[permutation_number][i][j] +1 ];;
		od;;
		if (i >= vv_start_number[permutation_number]) then 
			Add(current_state, (i - vv_start_number[permutation_number])*2 + vv_start_index[permutation_number] + current_assignment );
		else
			Add(current_state, 4*(i-1) + bv_start_index[permutation_number] + current_assignment);;
		fi;;
	od;;
	return current_state;;
end;;



get_state_from_graph := function(permutation_number, current_nodes)
	# Iterate over all nodes in the list and update the state accordingly
	local current_block;
	for i in [1..Length(current_nodes)] do
		current_block := LookupDictionary( nodes_to_states[permutation_number], current_nodes[i] );;
		for j in [1..Length(current_block[1])] do
			state[ current_block[1][j] + 1] := current_block[2][j];;
		od;;
	od;;
end;;

assignments_4 := [[0,0], [0,1], [1,0], [1,1]];;


get_state_from_graph_2 := function(permutation_number, current_nodes)
	local current_block, current_assignment;
	for i in [1..Length(current_nodes)] do
		if (current_nodes[i] >= vv_start_index[permutation_number]) then
			current_block := QuoInt(current_nodes[i] - vv_start_index[permutation_number], 2);;
			current_assignment := RemInt(current_nodes[i] - vv_start_index[permutation_number], 2);;
			current_block := blocks_list[permutation_number][ vv_start_number[permutation_number] + current_block];;
			state[ current_block[1] + 1 ] := current_assignment;;
		else
			current_block := QuoInt(current_nodes[i] - bv_start_index[permutation_number], 4);;
			current_assignment := RemInt(current_nodes[i] - bv_start_index[permutation_number], 4);;
			current_block := blocks_list[permutation_number][ 1 + current_block];;
			current_assignment := assignments_4[ current_assignment + 1 ];;
			state[ current_block[1] +1 ] := current_assignment[1];;
			state[ current_block[2] +1 ] := current_assignment[2];;
		fi;;
	od;;
end;;
	



jump_in_orbit := function(permutation_number)
	# First convert the current state into the set of nodes in the graph
	local current_state;;
	current_state := get_graph_from_state_2(permutation_number);;
	# Then apply permutation to this set
	current_state := OnSets(current_state, Next(replacers[permutation_number]));; 
	# Recover the original state back from this new state
	get_state_from_graph_2(permutation_number, current_state);
end;;


jump_in_orbit_2 := function(permutation_number, p)
	local current_node, current_assignment, current_block, old_state, new_state, new_node;;
	#p := Next(replacers[permutation_number]);;
	if (p=()) then 
		return;;
	fi;;
	old_state := ShallowCopy(state);;
	#new_state := [];;
	for i in [1..Length(blocks_list[permutation_number])] do
		current_assignment := 0;;
		# Get the current assignment of variables in this block
		for j in [1..Length(blocks_list[permutation_number][i])] do
			current_assignment := 2*current_assignment + old_state[ blocks_list[permutation_number][i][j] +1 ];;
		od;;
		# Convert this to a node in the graph
		# Apply permutation over this
		# Update the state with the new assignments of the variables
		if (i >= vv_start_number[permutation_number]) then 
			# Get the current_node in the graph
			current_node := (i - vv_start_number[permutation_number])*2 + vv_start_index[permutation_number] + current_assignment;;
			# Apply permutation to the current node in the graph
			new_node := OnPoints(current_node, p);;
			if (new_node = current_node) then
				continue;;
			fi;;	
			# Convert the current_node to states and update them accordingly
			current_block := QuoInt(new_node - vv_start_index[permutation_number], 2);;
			current_assignment := RemInt(new_node - vv_start_index[permutation_number], 2);;
			current_block := blocks_list[permutation_number][ vv_start_number[permutation_number] + current_block];;
			state[ current_block[1] + 1 ] := current_assignment;;
		else
			# Get the current node in the graph
			current_node := 4*(i-1) + bv_start_index[permutation_number] + current_assignment;;
			# Apply the permutation to the current node in the graph
			new_node := OnPoints(current_node, p);;
			if (new_node = current_node) then 
				continue;;
			fi;;
			# Convert the node to states and update them accordingly
			current_block := QuoInt(new_node - bv_start_index[permutation_number], 4);;
			current_assignment := RemInt(new_node - bv_start_index[permutation_number], 4);;
			current_block := blocks_list[permutation_number][ 1 + current_block];;
			current_assignment := assignments_4[ current_assignment + 1 ];;
			state[ current_block[1] +1 ] := current_assignment[1];;
			state[ current_block[2] +1 ] := current_assignment[2];;
		fi;;
	od;;
	#state := new_state;;
end;;

jump_in_orbit_3 := function(permutation_number, p)
	# Iterates over the elements in the permutation instead of the blocks
	# May or may not perform better than jump_in_orbit_2. Depends on the kinds
	# of permutation present in the model
	local moved_points, old_state, current_point, current_block, current_assignment;;
	# p := Next(replacers[permutation_number]);;
	# if (p=()) then 
	# 	return;;
	# fi;;
	moved_points := MovedPoints(p);;
	# Iterate over the moved points and keep updating the state accordingly
	old_state := ShallowCopy(state);;
	for i in [1..Length(moved_points)] do
		current_point := moved_points[i];;
		# Get the node from current_point and check if this is active
		# Check if this is a VV node or a BV node
		# Also check that this is not a feature node in the aux graph
		if (current_point < first_block_number[permutation_number] or current_point > last_block_number[permutation_number]) then 
			continue;;
		fi;;
		if (current_point >= vv_start_index[permutation_number]) then
			# Check if this point is on in the current state
			current_block := QuoInt(current_point - vv_start_index[permutation_number], 2);;
			current_assignment := RemInt(current_point - vv_start_index[permutation_number], 2);;
			current_block := blocks_list[permutation_number][ vv_start_number[permutation_number] + current_block];;
			if (old_state[ current_block[1] + 1 ] = current_assignment) then 
				current_point := OnPoints(current_point, p);;
				current_block := QuoInt(current_point - vv_start_index[permutation_number], 2);;
				current_assignment := RemInt(current_point - vv_start_index[permutation_number], 2);;
				current_block := blocks_list[permutation_number][ vv_start_number[permutation_number] + current_block];;
				state[ current_block[1] + 1 ] := current_assignment;;
			else
				continue;;
			fi;;
		else 
			current_block := QuoInt(current_point - bv_start_index[permutation_number], 4);;
			current_assignment := RemInt(current_point - bv_start_index[permutation_number], 4);;
			current_block := blocks_list[permutation_number][ 1 + current_block];;
			current_assignment := assignments_4[ current_assignment + 1 ];;
			if (old_state[ current_block[1] +1 ] = current_assignment[1] and old_state[ current_block[2] +1 ] = current_assignment[2]) then
				current_point := OnPoints(current_point, p);;
				current_block := QuoInt(current_point - bv_start_index[permutation_number], 4);;
				current_assignment := RemInt(current_point - bv_start_index[permutation_number], 4);;
				current_block := blocks_list[permutation_number][ 1 + current_block];;
				current_assignment := assignments_4[ current_assignment + 1 ];;
				state[ current_block[1] +1 ] := current_assignment[1];;
				state[ current_block[2] +1 ] := current_assignment[2];;
			else
				continue;;
			fi;;
		fi;;
	od;;
end;;

jump_in_orbit_4 := function(permutation_number, p)
	# Iterates over the elements in the permutation instead of the blocks
	local moved_points, old_state, i, current_point, current_start_perm,
		current_end_perm, vv_start, current_block_number, current_block, 
		current_assignment, new_point, current_end;;

	# p := Next(replacers[permutation_number]);;
	# if (p=()) then 
	# 	return;;
	# fi;;
	moved_points := MovedPoints(p);;
	# TODO: moved_points must be a sorted list. This was verified experimentally
	# Although it might be better to add a check here specifically
	old_state := ShallowCopy(state);;
	# TODO: Need to check if this performs any better than keeping a global
	# new_state and assigning values to that
	i := 1;;
	# First get to the first point which is a valid block
	current_point := 0;;
	current_start_perm := first_block_number[permutation_number];;
	current_end_perm := last_block_number[permutation_number];;
	vv_start := vv_start_index[permutation_number];;
	while (current_point < current_start_perm and i<= Length(moved_points)) do
		current_point := moved_points[i];;
		i := i + 1;;
	od;;
	i := i-1;;
	# Check if the point found is useful
	if (current_point < current_start_perm) then
		return;;
	fi;;
	# Start iterating over blocks with moved point i
	while ( i< Length(moved_points)) do
		current_point := moved_points[i];;
		# Check if current_point is not useful for applying BV permutation
		if ( current_point > current_end_perm) then
			break;;
		fi;;
		# Find the current block this belongs
		if (current_point >= vv_start) then
			current_block_number := QuoInt(current_point - vv_start, 2);;
			# Get the assignment for this block and update i to the next block
			current_block := blocks_list[permutation_number][ vv_start_number[permutation_number] + current_block_number];;
			current_assignment := old_state[ current_block[1] + 1 ];;
			current_point := current_block_number*2 + vv_start + current_assignment;;
			current_end := current_block_number*2 + vv_start + 1;;
			# Would need to find the value in moved_list strictly larger than this
			new_point := OnPoints(current_point, p);;
			if (new_point <> current_point) then
				current_block := QuoInt(new_point - vv_start, 2);;
				current_assignment := RemInt(new_point - vv_start, 2);;
				current_block := blocks_list[permutation_number][ vv_start_number[permutation_number] + current_block];;
				state[ current_block[1] + 1 ] := current_assignment;;
			fi;;
			# Move i to the next suitable point
			while(i < Length(moved_points) and moved_points[i] <= current_end) do
				i := i + 1;;
			od;;
		else
			current_block_number := QuoInt(current_point - bv_start_index[permutation_number], 4);;
			current_block := blocks_list[permutation_number][ 1 + current_block_number];;
			current_assignment := 0;;
			# Get the current assignment of variables in this block
			for j in [1..Length(current_block)] do
				current_assignment := 2*current_assignment + old_state[ current_block[j] +1 ];;
			od;;
			current_point := 4*(current_block_number) + bv_start_index[permutation_number] + current_assignment;;
			current_end := 4*(current_block_number) + bv_start_index[permutation_number] + 3;;
			new_point := OnPoints(current_point, p);;
			if (new_point <> current_point) then
				current_block := QuoInt(new_point - bv_start_index[permutation_number], 4);;
				current_assignment := RemInt(new_point - bv_start_index[permutation_number], 4);;
				current_block := blocks_list[permutation_number][ 1 + current_block];;
				current_assignment := assignments_4[ current_assignment + 1 ];;
				state[ current_block[1] +1 ] := current_assignment[1];;
				state[ current_block[2] +1 ] := current_assignment[2];;
			fi;;
			while(i < Length(moved_points) and moved_points[i] <= current_end) do
				i := i + 1;;
			od;;
		fi;;
	od;;
end;;

jump_in_orbit_5 := function(permutation_number)
	local p;;
	p := Next(replacers[permutation_number]);;
	if (p=()) then 
		return;;
	fi;;
	if (NrMovedPoints(p) < Length(blocks_list[permutation_number])) then
		jump_in_orbit_4(permutation_number, p);;
	else
		jump_in_orbit_2(permutation_number, p);;
	fi;;
end;;


read_file := function(file_name)
	local inp, file_str, file_lines, i, f, temp, var, sz, w, cl, l, temp2, this_feat, include;;

	inp := InputTextFile(file_name);;
	file_str := ReadAll(inp);;
	file_lines := SplitString(file_str, "\n");;

	num := Int(file_lines[1]);;

	for i in [3..(3+num-1)] do
		temp := SplitString(file_lines[i], ":");;
		var := Int(temp[1]);;
		sz := Int(temp[2]);;
		dom_size[var] := sz;;
	od;;

	for f in [(4+num)..Length(file_lines)] do

		if file_lines[f] = "" then
			continue;;
		fi;;

		temp := SplitString(file_lines[f], "\t");;

		w := Float(temp[1]);;
		temp := SplitString(temp[2], ",");;

		cl := [];;

		for l in temp do
			temp2 := SplitString(l, " ");;
			Add(cl, [Int(temp2[1]), Int(temp2[2])]);;
		od;;

		Add(feat, [w, cl]);;
	od;;

	for i in [1..num] do
		this_feat := [];;

		for f in feat do
			include := false;;
			cl := f[2];;
			for l in cl do
				if l[1] = i then
					include := true;;
					break;;
				fi;;
			od;;

			if include then
				Add(this_feat, f);;
			fi;;
		od;;

		Add(var_to_feat, this_feat);;
	od;;
end;;


init_state := function()
	local i, r;;

	state := [];;
	for i in [1..num] do
		r := Random(rs1, 0, dom_size[i]-1);;
		Add(state, r);;
	od;;
end;;

init_counts := function()
	local i, j;;

	for i in [1..num] do
		counts[i] := [];;
		marginals[i] := [];;
		for j in [1..dom_size[i]] do
			Add(counts[i], 0);;
			Add(marginals[i], 0.0);;
		od;;
	od;;
end;;


sample_var := function(var_to_sample)
	local up, sum_worlds, old_val, d, tot_w, f, w, cl, satisfied, l, r, Sum;;

	up := [];;	# Unnormalized probability for each assignment of var
	sum_worlds := 0.0;;	# This is the sum of the exponentials of each assignment possible

	old_val := state[var_to_sample];;

	for d in [0..(dom_size[var_to_sample]-1)] do
		state[var_to_sample] := d;;

		tot_w := 0.0;;	# Will hold the sum of weights of satisfied features in this assignment

		for f in var_to_feat[var_to_sample] do
			w := f[1];;
			cl := f[2];;

			satisfied := false;;

			for l in cl do
				if state[l[1]] = l[2] then
					satisfied := true;;
					break;;
				fi;;
			od;;

			if satisfied then
				tot_w := tot_w + w;;
			fi;;
		od;;

		Add(up, Exp(tot_w));;
		sum_worlds := sum_worlds + up[d+1];;
	od;;

	state[var_to_sample] := old_val;;

	r := Random(rs1, 0, 100000000000) / 100000000000.0 * sum_worlds;;
	Sum := 0;;
	for d in [0..(dom_size[var_to_sample]-1)] do
		Sum := Sum + up[d+1];;
		if r <= Sum then
			return d;;
		fi;;
	od;;
end;;


update_counts := function()
	local i;;

	for i in [1..num] do
		counts[i][state[i]+1] := counts[i][state[i]+1] + 1;;
	od;;
end;;


update_marginals := function(n)
	local i, d;;

	for i in [1..num] do
		for d in [0..(dom_size[i]-1)] do
			marginals[i][d+1] := counts[i][d+1]/Float(n);;
		od;;
	od;;
end;;


kullback := function(m1, m2, type)	# m1 is current marginals, m2 is true marginals
	local sum1, sum2, i, d;;

	sum1 := 0.0;;
	sum2 := 0.0;;

	for i in [1..num] do
		for d in [0..(dom_size[i]-1)] do
			if(m1[i][d+1] > 0.0) then
				if(m2[i][d+1] > 0.0) then
					sum1 := sum1 + (m1[i][d+1] * Log(m1[i][d+1]/m2[i][d+1]));;
				else
					sum1 := sum1 + (m1[i][d+1] * Log(m1[i][d+1]/smooth));;
				fi;;
			fi;;
		od;;
	od;;


	for i in [1..num] do
		for d in [0..(dom_size[i]-1)] do
			if(m2[i][d+1] > 0.0) then
				if(m1[i][d+1] > 0.0) then
					sum2 := sum2 + (m2[i][d+1] * Log(m2[i][d+1]/m1[i][d+1]));;
				else
					sum2 := sum2 + (m2[i][d+1] * Log(m2[i][d+1]/smooth));;
				fi;;
			fi;;
		od;;
	od;;

	if type = 1 then
		return sum1 + sum2;;
	elif type = 2 then
		return sum1 / Float(Length(m1));;
	elif type = 3 then
		return sum1 / Float(num);;
	else
		return sum2 / Float(Length(m2));;
	fi;;
end;;


gibbs := function(num_sample,iterations)
	local n, var_to_sample, new_val, start_time, klPerInterval, entries_per_iter, timeArr, kl_vals, i, iter, variance, current_kl, standard_error, timeTaken, current_permutation, random_sample;;
	entries_per_iter := num_sample/inter_sz;;
	klPerInterval := [];;
	timeArr := [];;
	kl_vals := [];;
	current_permutation := 0;;

	for i in [1..entries_per_iter] do
		timeArr[i] := 0 ;;
		klPerInterval[i] :=0;;
	od;;

	for iter in [1.. iterations] do 
		start_time := runtimeSum();;

		init_state();;
		init_counts();;
		PrintTo("block_gibbs.csv", "");;

		for n in [1..num_sample] do

			# Gibbs Step
			var_to_sample := Random(rs1, 1, num);;
			new_val := sample_var(var_to_sample);;
			state[var_to_sample] := new_val;;
			

			# Orbital step
			# With probability alpha take an orbital step
			random_sample := Random(rs1, 0, 10000) / 10000.00;;
			if (random_sample < alpha) then 
				current_permutation := Random(rs1, 1, Length(blocks_list));;
				jump_in_orbit_5(current_permutation);;
				#current_permutation := RemInt(current_permutation+1, Length(blocks_list));;
			fi;;
			# Update counts
			update_counts();;

			# Compute KL-Divergence every inter_sz samples
			if (RemInt(n,inter_sz) = 0)  then
				update_marginals(n);;
				current_kl := kullback(marginals, true_marginals, 3);
				timeArr[n/inter_sz] := timeArr[n/inter_sz] + (runtimeSum()-start_time)/1000.0 ;;
				kl_vals[iter*entries_per_iter+n/inter_sz] := current_kl;;
				klPerInterval[n/inter_sz]:= klPerInterval[n/inter_sz] + current_kl;;
				if iter = iterations then
					timeArr[n/inter_sz] := timeArr[n/inter_sz]/iterations;;
					klPerInterval[n/inter_sz] := klPerInterval[n/inter_sz]/iterations;;
					variance := 0;
					for i in [1..iterations] do
						variance := variance + (kl_vals[i*entries_per_iter+n/inter_sz] - klPerInterval [n/inter_sz])*(kl_vals[i*entries_per_iter+n/inter_sz] - klPerInterval [n/inter_sz]);
					od;;
					standard_error := 2*Sqrt(variance)/iterations;;
					timeTaken := timeArr[n/inter_sz] + sym_discovery_time;;
					#timeTaken := timeArr[n/inter_sz];;
					AppendTo("block_gibbs.csv",String(timeTaken),",",String(klPerInterval[n/inter_sz]),",",String(n),",",String(standard_error),"\n");;
				fi;;

				
			fi;;
			if (RemInt(n, 100000) = 0) then
				Print(iter, ": ", n, "\n");;
			fi;;
		od;;
	od;;
end;;


read_file("test.num");;

gibbs(num_sample,iterations);;

PrintTo("block_marginals.txt", "block_marginals := [");;

for i in [1..num] do
	AppendTo("block_marginals.txt", "\n[");;
	for d in [0..(dom_size[i]-1)] do
		AppendTo("block_marginals.txt", counts[i][d+1]/Float(num_sample), ",");;
	od;;
	AppendTo("block_marginals.txt", "],");;
od;;

AppendTo("block_marginals.txt", "\n];;\n");;
