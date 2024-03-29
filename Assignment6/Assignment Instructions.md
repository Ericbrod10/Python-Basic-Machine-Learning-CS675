# Python-Basic-Machine-Learning-CS675

Write a Python program that determines the column with the
best split for the CART decision tree algorithm. You don't
have to write the CART algorithm in its entirety. You just
have to write a program that will traverse all columns in the
data and output the column and the threshold that gives the
lowest gini index.

The input should be the data file and labels as in previous
assignments. The output is the column number k and the split
value s.

We will test it on a simple example to determine if your program 
gives the correct output. Test your program with a simple XOR example.

Submit your assignment by copying it into the directory
/afs/cad/courses/ccs/S19/cs/675/850/<ucid>.
For example if your ucid is abc12 then copy your programs
/afs/cad/courses/ccs/S19/cs/675/850/abc12.

Your completed script is due midnight on July 28th 2019. 

High level pseudocode:

    (1) For each column j:
	        
	(1) Find the value that gives the minimum gini split of 
	the data d into a partition of two sets

	(2) To evaluate the gini of a split use the formula

	gini = (lsize/rows)*(lp/lsize)*(1 - lp/lsize) + (rsize/rows)*(rp/rsize)*(1 - rp/rsize);

	where lsize is the size of the left partition, lp is the 
	proportion of -1 labels in the left partition, rsize is the 
	size of the right partition, rp is the proportion of -1 
	labels in the right partition, and rows is the total number of
	datapoints in the dataset d (passed to the function)

    (2) Let column k give the best split s. Output k and s.
     