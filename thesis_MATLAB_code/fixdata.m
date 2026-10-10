% Script to clean up data after using graph digitizer.
% Written by Lev Thomson for MATLAB version 24.2.0.2773142 (R2024b)

clear all
close all

halfpix = 5;

t = @(x) double(x>=-halfpix & x<halfpix);

nums = linspace(-10, 10, 20);

tophat = t(nums);

s = 0.0525;
m = 0.001167;

gaussian = @(x) ((1/(2*pi*s^2)^(0.5)) * exp(-((x-m)^2 / 2*(s)^2)));

convolve = @(x) (t(x) * gaussian(x));

func = convolve(nums);