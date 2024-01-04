"""
This module contains the functions used to log information
"""
import logging
import time


def print_name_stage_project(stage):
    '''
    At the beginning of a stage (like Data preparation, data cleaning, ...) it prints
    the name of the stage.

    Args:
        - stage (str): name of the stage

    Returns:
        - None

    '''
    width = 50+len(stage)
    logging.info("\n")
    logging.info("-"*(width))
    logging.info("-"*(width))
    logging.info(f"-----------------------  {stage}  -----------------------")
    logging.info("-"*(width))
    logging.info(("-"*(width))+"\n\n")
    time.sleep(3)
