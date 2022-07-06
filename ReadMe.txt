This document contains details on how to run the experiments.


Python Dependencies:

1. PuLP:  “pip install pulp”

2. MatplotLib, Numpy, Pandas



How to Run:

The code contains subfolders, named FactoredCMDP, InventoryControl, and MediaControl. These correspond to experiments for each of those environments in the paper.

In each of these subfolders,

1. First run model*.py. This will create solution files to compute regret and generate baseline policies for OptPessLP and DOPE.

2. Run OptCMDP*.py for OptCMDP, DOPE*.py for DOPE, OptPessLP*.py for OptPessLP, alwayssafe.py for AlwaysSafe. The code will generate .pckl files which contain cumulative regret for each environment. 

3. Change the RUN_NUMBER field to run the above for a different seed. The plots in the paper are averaged over 20 runs each.

4. For plots with different baseline policies, change the C_b value in the model*.py