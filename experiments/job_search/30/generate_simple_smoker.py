import random
from sys import argv
from itertools import permutations

num = int(argv[1]) # Number of people
# Number of variables is 2*num + num*(num-1)/2

out = open(argv[2], 'w')

out.write("Vars:")
for i in range(0, 2*num + num*(num-1) - 1):
	out.write(str(i) + ',')
out.write(str(2*num + num*(num-1) - 1) + '\n\n')

for i in range(0, 2*num + num*(num-1)):
	out.write(str(i) + '=0,1\n')

out.write('\n')

current_variable = 0

friends_weight = 0.1

# Adding clauses for smokes(x) => cancer(x) and smokes(x) => not cancer(x)

for i in range(num):
	weight = random.uniform(1,2)
	w = '{0:.2f}'.format(weight)
	out.write(w + '\t' + str(current_variable) + " 0,"
		+ str(current_variable+1) + " 1\n")
	weight = random.uniform(1,2)
	w = '{0:.2f}'.format(weight)
	out.write(w + '\t' + str(current_variable) + " 0,"
		+ str(current_variable+1) + " 0\n")
	current_variable+=2

# Adding edges for friends in the network

for i in range(num):
	for j in range(num):
		if (i != j):
			w = '{0:.1f}'.format(friends_weight)
			out.write(w + '\t' + str(current_variable) + " 0,"+ str(2*i) + " 0," +
				str(2*j) + " 1" + "\n")
			current_variable+=1

out.close()
