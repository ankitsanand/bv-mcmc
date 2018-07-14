import random
from sys import argv
from itertools import permutations

num = int(argv[1]) # Number of students
# Number of variables is 2*num

friends = int(argv[2])
# No of friends nodes to be added to the graph

out = open(argv[3], 'w')

out.write("Vars:")
for i in range(0, 2*num-1):
	out.write(str(i) + ',')
out.write(str(2*num-1) + '\n\n')

for i in range(0, 2*num):
	out.write(str(i) + '=0,1\n')

out.write('\n')

base_weights = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
assignments = [('0','0'), ('0','1'), ('1','0'), ('1','1')]
current_variable = 0

for i in range(num):
	weights = random.sample(base_weights, 4)
	weights = ['{0:.1f}'.format(j) for j in weights]
	for j in xrange(4):
		out.write(weights[j] + '\t' + str(current_variable) + " " + 
			assignments[j][0] + "," + str(current_variable+1) + " " +
			assignments[j][1] + "\n")
	current_variable+=2

# Adding edges for friends to the list of features
for i in range(friends):
	p = random.randint(0,num-1)
	q = random.randint(0,num-1)
	while( q == p):
		q = random.randint(0,num -1)
	w = 0.01
	out.write('{0:.2f}'.format(w) + '\t' + str(2*p) + " 0," + str(2*q) + " 1" +"\n")
	out.write('{0:.2f}'.format(w) + '\t' + str(2*p+1) + " 0," + str(2*q+1) + " 1" +"\n")
	out.write('{0:.2f}'.format(w) + '\t' + str(2*q) + " 0," + str(2*p) + " 1" +"\n")
	out.write('{0:.2f}'.format(w) + '\t' + str(2*q+1) + " 0," + str(2*p+1) + " 1" +"\n")

out.close()
