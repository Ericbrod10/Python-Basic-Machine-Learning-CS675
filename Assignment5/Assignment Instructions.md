# Python-Basic-Machine-Learning-CS675

Modify your solution for assignments 2 and 3 to do an adaptive
eta setting. Between the compute dellf and updatew code portions
insert the following pseudocode (see least squares in Perl code on
course website for reference). 

eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
bestobj = 1000000000000
for k in range(0, len(eta_list), 1):

  eta = eta_list[k]
  
  ##update w
  ##insert code here for w = w + eta*dellf

  ##get new error
  error = 0
  for i in range(0, rows, 1):
    if(trainlabels.get(i) != None):
      ##update error
      ##insert code to update the loss (which we call error here)

  obj = error

  ##update bestobj and best_eta
  ##insert code here

  ##remove the eta for the next
  ##insert code here for w = w - eta*dellf

eta = best_eta

After you have the adapative step size solutions working obtain
the average test error of least squares and hinge on the six
datasets on the course website. For this use the avg_test_error
script from https://web.njit.edu/~usman/courses/cs675_summer19/avg_test_error.pl.