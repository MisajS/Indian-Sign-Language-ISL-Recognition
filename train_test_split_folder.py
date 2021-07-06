# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:21:50 2020

@author: Lenovo
"""

import splitfolders  # or import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# splitfolders.ratio("D:\\Misaj\\Mini Project\\Dataset300", output="D:\\Misaj\\Mini Project\\Dataset300", seed=1337, ratio=(.7, .3), group_prefix=None) # default values
splitfolders.ratio("D:\\Misaj\\S3 MTech\\Project\\Datasets\\Data\\Data_class_2018", output="D:\\Misaj\\S3 MTech\\Project\\Datasets\\Data\\Data_class_2018", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values


# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# splitfolders.fixed("C:\\Users\\misaj\\Downloads\\Dataset", output="D:\\Misaj\\Mini Project\\Dataset", seed=1337, fixed=(100, 25), oversample=False, group_prefix=None) # default values