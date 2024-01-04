"""
This module contains costants used across all the repository
"""

import os

## PATHS

PROJECT_PATH = os.path.abspath(".")
DATASET_PATH = os.path.join(PROJECT_PATH, "../Data")

## ORIGINAL HIGH RES
#ORIGINAL_DS = os.path.join(DATASET_PATH, "HR")
ORIGINAL_DS = os.path.join(DATASET_PATH, "HR2")
ORIGINAL_DS_TRAIN = os.path.join(ORIGINAL_DS, "DIV2K_train_HR")
ORIGINAL_DS_TEST = os.path.join(ORIGINAL_DS, "DIV2K_test_HR")
ORIGINAL_DS_VAL = os.path.join(ORIGINAL_DS, "DIV2K_val_HR")


# LR
LR = os.path.join(DATASET_PATH, "X8")
LR_TRAIN = os.path.join(LR, "DIV2K_train_LR")
LR_TEST = os.path.join(LR, "DIV2K_test_LR")
LR_VAL = os.path.join(LR, "DIV2K_val_LR")


