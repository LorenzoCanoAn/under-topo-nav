#!/usr/bin/python3
import argparse
from training_utils import Dataset
import pathlib


##############################################################
#	Configuration of the parser
##############################################################
parser = argparse.ArgumentParser(description='Tests a gallery detection Neural Network.')
parser.add_argument("path_to_nn", type=pathlib.Path, action="store")
parser.add_argument("path_to_dataset", type=pathlib.Path, action="store")
parser.add_argument("--fraction_to_test", action="store", type=int)

##############################################################
#	Import the network
##############################################################

