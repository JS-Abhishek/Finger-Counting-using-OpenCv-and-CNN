# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:29:50 2020

@author: jsabh
"""


import glob
import shutil
import os

if not os.path.exists("images"):
    os.makedirs("images")
    os.makedirs("images/train")
    os.makedirs("images/test")
    os.makedirs("images/train/0")
    os.makedirs("images/train/1")
    os.makedirs("images/train/2")
    os.makedirs("images/train/3")
    os.makedirs("images/train/4")
    os.makedirs("images/train/5")
    os.makedirs("images/test/0")
    os.makedirs("images/test/1")
    os.makedirs("images/test/2")
    os.makedirs("images/test/3")
    os.makedirs("images/test/4")
    os.makedirs("images/test/5")

test_dir = "C:/Users/jsabh/OneDrive/Documents/ML programs/images/test/FIVE"
des_test = "images/test/5"

for jpgfile in glob.iglob(os.path.join(test_dir,"*.png")):
    shutil.copy(jpgfile,des_test)
