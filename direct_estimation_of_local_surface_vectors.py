#!/usr/bin/env python
# coding: utf-8

# In[0]

import numpy as np
import sympy as sp
import cv2

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

# ********* Disparity Gradiant (With viewing geometry known) ********* #
# Normalizing the disparity gradiant eliminates the gaze angle variable.

def calcualte_disparity_gradiant_known_geometry(vergance_angle, surface_z):
    X, Y = sp.symbols('X Y', real=True)

    P = sp.diff(surface_z, X)  # ∂Z/∂X
    Q = sp.diff(surface_z, Y)  # ∂Z/∂Y

    denominator = (np.cos(vergance_angle) - P * np.sin(vergance_angle))

    m_11 = (np.cos(vergance_angle) + P * np.sin(vergance_angle)) / denominator
    m_12 = (2*Q*np.cos(vergance_angle)*np.sin(vergance_angle)) / denominator

    P = ((m_11 - 1) * np.cos(vergance_angle)) / (m_11 + 1) * np.sin(vergance_angle)
    Q = m_12/ (m_11 + 1) * np.sin(vergance_angle)

    return P, Q

# ********* Window Second Moment Matrix ********* #
def calcualte_window_second_moment_matrix(image, window_size, sigma):
    #In this case, the averaging operator is a gaussian blur (this can be changed for others types as well)
    image = image.astype(np.float32)

    Lx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Ly = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    Lx2 = Lx ** 2
    Ly2 = Ly ** 2
    LxLy = Lx * Ly

    mu11 = cv2.GaussianBlur(Lx2, (window_size, window_size), sigma)
    mu22 = cv2.GaussianBlur(Ly2, (window_size, window_size), sigma)
    mu12 = cv2.GaussianBlur(LxLy, (window_size, window_size), sigma)

    return mu11, mu12, mu22 # matrix is symetric so m12 is used twice

def compute_trace(mu11, mu22):
    return mu11 + mu22

# ********* directional statistics (C~, S~) from the second moment matrix. ********* #
"""
- C_tilde: anisotropy (difference between directions)
- S_tilde: orientation measure
"""
def compute_directional_statistics(mu11, mu12, mu22):
    trace = compute_trace(mu11, mu22)
    epsilon = 1e-6
    trace_safe = trace + epsilon

    C_tilde = (mu11 - mu22) / trace_safe
    S_tilde = (2 * mu12) / trace_safe

    return C_tilde, S_tilde
