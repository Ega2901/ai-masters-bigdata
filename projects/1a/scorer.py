#!/opt/conda/envs/dsenv/bin/python

#
# This is a log loss scorer
#

import sys
import os
import logging
from sklearn.metrics import log_loss
import numpy as np

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

y_true = []
y_pred = []

for line in sys.stdin:
    true_label, prob_0, prob_1 = map(float, line.strip().split(","))

    y_true.append(true_label)
    y_pred.append([prob_0, prob_1])

score = log_loss(y_true, y_pred)

print(score)

sys.exit(0)
