#!/usr/bin/env python
# coding: utf-8

# In[0]

import numpy as np

def calculate_vergance_angle(angle_l, angle_r):
    return 0.5*(angle_l - angle_r)

def calculate_version_angle(angle_l, angle_r):
    return 0.5*(angle_l + angle_r)