# Towards Robust, Locally Linear Deep Networks:

This repository is for the paper

 * "[Towards Robust, Locally Linear Deep Networks](https://openreview.net/pdf?id=SylCrnCcFX)" by [Guang-He Lee](https://people.csail.mit.edu/guanghe/), [David Alvarez Melis](https://people.csail.mit.edu/davidam/), and [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/) in ICLR 19.
 * [Project page](http://people.csail.mit.edu/guanghe/locally_linear)

## Package version:

 * PyTorch0.4.1
 * python3.6.1

## Reproducing the MNIST experiment:

 * Please execute the shell files (`gamma100_fc.sh`) to reproduce the experiment on MNIST dataset with gamma=100. The results will be in the folder `fc_log/`

 * `parse_log.py` is a utility script. After you run all the models using `gamma100_fc.sh`. Use the following comment:

   * `ls fc_log > fc_log.list`
   * `cd fc_log`
   * `python ../parse_log.py --file-list ../fc_log.list`

 * To see the best model in terms of the median of L2 margin given each validation accuracy. Then we recommend to look into the log file to finally see the testing accuracy on the desired model.

## Other experiments:

 * Unfortunately we don't plan to release the codes for other experiments.

## Citation:

If you find this repo useful for your research, please cite the paper

```
@inproceedings{
lee2018towards,
title={Towards Robust, Locally Linear Deep Networks},
author={Guang-He Lee and David Alvarez-Melis and Tommi S. Jaakkola},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=SylCrnCcFX},
}
```
