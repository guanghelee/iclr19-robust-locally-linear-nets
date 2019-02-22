# Package version:

PyTorch0.4.1

python3.6.1

# Reproducing the MNIST experiment:

 * Please execute the shell files (gamma100_fc.sh) to reproduce the experiment on MNIST dataset with gamma=100. The results will be in the folder fc_log

 * parse_log.py is a utility script. After you run all the models using gamma100_fc.sh. Use the following comment:

   * ls fc_log > fc_log.list
   * cd fc_log
   * python ../parse_log.py --file-list ../fc_log.list

 * To see the best model in terms of \hat \epsilos\_{2, 50} given each validation accuracy. Then we recommend to look into the log file to finally see the testing accuracy on the desired model.

