from sys import argv
from random import sample

evidence = float(argv[1]) # Evidence to be added to the model
file_input = open(argv[2])
output = argv[3]

v = file_input.readline().strip()
v = v[5:].split(',')
# v is now a list of variables in the model

file_input.readline() # Empty line
# Next |v| lines contain the domain
# We assume the domain is boolean for now and ignore all these lines
for i in xrange(len(v)):
	file_input.readline()

file_input.readline() # Empty line

# Pick a set of evidence*|v| as the evidence variables in the model
evidence = int(evidence * len(v))
evidence_variables = sample(v[100:], evidence) #Only sample variables which represent friends

var_mapping = {}
current_variable = 0
# Assign a new numbering to the domain
for i in v:
	if i not in evidence_variables:
		var_mapping[i] = current_variable
		current_variable+=1

# Output the variables to the new file
file_output = open(output, 'w')
all_var = [var_mapping[i] for i in var_mapping]
all_var.sort()
all_var = [str(i) for i in all_var]
file_output.write("Vars:"+','.join(all_var)+'\n')
# Empty line
file_output.write("\n")
# Output all the domains of the variables (Boolean only)
for i in all_var:
	file_output.write(i + "=0,1\n")

# Empty line
file_output.write("\n")

# Iterate over all the features present in the model
for line in file_input:
	feat = line.strip().split('\t')
	weight = feat[0]
	var = feat[1].split(',')
	var = [i.split(" ") for i in var]
	# Check if any variable is an evidence variable
	if len(filter(lambda x: x[0] not in var_mapping, var))==0:
		var = map(lambda x: [str(var_mapping[x[0]]), x[1]], var)
		var = [" ".join(i) for i in var]
		feat = weight + '\t' + ','.join(var)
		file_output.write(feat+'\n')

file_input.close()
file_output.close()


	










