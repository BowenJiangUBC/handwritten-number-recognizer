#
# Created by bowenjiang on 3/19/18.
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from mnist import MNIST


def main():

    mndata = MNIST('./data')
    mndata.gz = True
    images, labels = mndata.load_training()