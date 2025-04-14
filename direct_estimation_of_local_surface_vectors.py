#!/usr/bin/env python
# coding: utf-8

# In[0]

import numpy as np
import sympy as sp

def calculate_vergance_angle(angle_l, angle_r):
    return 0.5*(angle_l - angle_r)

def calculate_version_angle(angle_l, angle_r):
    return 0.5*(angle_l + angle_r)

# ********* Disparity Gradiant ********* #
def calculate_disparity_gradient(vergence_angle, version_angle, surface_z):
    X, Y = sp.symbols('X Y', real=True)

    P = sp.diff(surface_z, X)  # ∂Z/∂X
    Q = sp.diff(surface_z, Y)  # ∂Z/∂Y

    cos_vergence_angle = sp.cos(vergence_angle)
    sin_vergence_angle = sp.sin(vergence_angle)

    denom = (cos_vergence_angle - P * sin_vergence_angle)

    a11 = (cos_vergence_angle + P * sin_vergence_angle) / denom
    a12 = (2 * Q * cos_vergence_angle * sin_vergence_angle) / denom
    a21 = 0
    a22 = 1

    scale = sp.cos(version_angle - vergence_angle) / sp.cos(version_angle + vergence_angle)

    M = scale * sp.Matrix([
        [a11, a12],
        [a21, a22]
    ])

    return M