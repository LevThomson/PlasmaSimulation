% Add noise to Anders raw data to determine error resulting from
% digitisation of Anders et al's graphs
seed = 0; % fixed seed
plotfig = true;
fig_4_width = 1480; % pixels
fig_4_height = 958; % pixels
fig_4_scale_y = 1.2; % no unit
fig_4_scale_x = 0.16; % m, centred on 0
fig_4_x_per_pixel = fig_4_scale_x / fig_4_width;
fig_4_y_per_pixel = fig_4_scale_y / fig_4_height;

fig_8_width = 286; % pixels
fig_8_height = 183; % pixels
fig_8_scale_y = 1.1; % no unit
fig_8_scale_x = 0.08; % m, centred on 0
fig_8_x_per_pixel = fig_8_scale_x / fig_8_width;
fig_8_y_per_pixel = fig_8_scale_y / fig_8_height;

% Read in figure raw data
rawdata_uncollimated = readmatrix("anders_raw_data/Ander1996Fig4.txt");
rawdata_collimated = readmatrix("anders_raw_data/Anders1996Fig8.txt");

% Split into elements & centre at zero
Pt = rmmissing(rawdata_uncollimated(:, 1:2));
Pt(:, 1) = (Pt(:, 1) - mean(Pt(:,1))) * 1e-2;
Pd = rmmissing(rawdata_uncollimated(:, 3:4));
Pd(:, 1) = (Pd(:, 1) - mean(Pd(:,1))) * 1e-2;
Co = rmmissing(rawdata_uncollimated(:, 5:6));
Co(:, 1) = (Co(:, 1) - mean(Co(:,1))) * 1e-2;
Mo = rmmissing(rawdata_uncollimated(:, 7:8));
Mo(:, 1) = (Mo(:, 1) - mean(Mo(:,1))) * 1e-2;
Al = rmmissing(rawdata_uncollimated(:, 9:10));
Al(:, 1) = (Al(:, 1) - mean(Al(:,1))) * 1e-2;

% This data is already centered
Fe_collimated = rmmissing(rawdata_collimated(:, 1:2));
Fe_collimated(:, 1) = Fe_collimated(:, 1) * 1e-2; 
Co_collimated = rmmissing(rawdata_collimated(:, 3:4));
Co_collimated(:, 1) = Co_collimated(:, 1) * 1e-2;
Ta_collimated = rmmissing(rawdata_collimated(:, 5:6));
Ta_collimated(:, 1) = Ta_collimated(:, 1) * 1e-2;
Al_collimated  = rmmissing(rawdata_collimated(:, 7:8));
Al_collimated(:, 1) = Al_collimated(:, 1) * 1e-2;

raw_data_4_uncollimated = Co(:, 2);
raw_data_8_collimated = Co_collimated(:, 2);

n_data_points_4 = length(raw_data_4_uncollimated);
n_data_points_8 = length(raw_data_8_collimated);

n_tests = 500;

raw_data_4 = repmat(raw_data_4_uncollimated.', n_tests, 1);
raw_data_8 = repmat(raw_data_8_collimated.',   n_tests, 1);

% Perturb the data by +- a few pixels using random gaussian noise:
seed = seed + 1;
rng(seed) % reseed
r_4 = normrnd(0,1,[n_tests,n_data_points_4]) * fig_4_y_per_pixel * 16; % +- 16 pixels, roughly the size of a cursor
perturbed_data_4 = raw_data_4 + r_4;

seed = seed + 1;
rng(seed) % reseed
r_8 = normrnd(0,1,[n_tests,n_data_points_8]) * fig_8_y_per_pixel * 4; % much lower resolution on this image (used a different screen when screenshotting it), so the image was zoomed in -> mouse covers fewer image pixels. 
perturbed_data_8 = raw_data_8 + r_8;

% PLOT ALL DATA POINTS ------------------------------------s

function normalised_array = normalise(arr)
% Normalise distribution
total_arr = sum(arr);
normalised_array = arr / total_arr;
end

if plotfig
    fig0 = figure();
    plot(Co(:, 1), normalise(raw_data_4_uncollimated(:)))
    hold on
    for i=1:n_tests
        plot(Co(:, 1), normalise(perturbed_data_4(i, :)))
    end
    title({'Normalised distribution of the film thickness',' for an uncollimated cathodic arc deposition system'})
    % legend("Platinum", "Palladium", "Cobalt", "Molybdenum", "Aluminium")
    xlabel("Displacement (m)")
    ylabel("Normalised Film Thickness (no units)")
    xlim([-0.08 0.08])
    ylim([0 8e-3])
    fontsize(16,"points")
    hold off
    saveas(fig0, "images/AndersUncollimated_Random_uncollimated", "png");

    fig1 = figure();
    plot(Co_collimated(:, 1), normalise(raw_data_8_collimated(:)))
    hold on
    for i=1:n_tests
        plot(Co_collimated(:, 1), normalise(perturbed_data_8(i, :)))
    end
    title({'Normalised distribution of the film thickness',' for a collimated cathodic arc deposition system'})
    xlabel("Displacement (m)")
    ylabel("Normalised Film Thickness (no units)")
    xlim([-0.08 0.08])
    ylim([0 8e-3])
    fontsize(16,"points")
    hold off
    saveas(fig1, "images/AndersUncollimated_Random_collimated", "png");
end

% minimum and maximum for all parameters of fit
piecewise_fit = fittype('piecewise_ft(xL, xR, a, b, d, e, g, h, k, x)', 'independent', 'x', 'dependent', 'y', 'coefficients', {'xL', 'xR', 'a', 'b', 'd', 'e', 'g', 'h', 'k'});
opts = fitoptions(piecewise_fit);
opts.StartPoint = [-0.015, 0.015, 41, 100, 41, -100, -1.8, 0, 0.007];  % [a b d e g h k]
opts.Lower      = [-1, -0.003, 0, 0, 0, -1000, -100, -100, -100]; % xL and xR constraints are the diameter of the cathode = 6mm
opts.Upper      = [0.003, 1, 100, 100, 100, 100, 100, 100, 100];

params = ["xL", "xR", "a", "b", "d", "e", "g", "h", "k"];
n_params = numel(params);

coeffs_col = zeros(n_tests, n_params);
coeffs_unc = zeros(n_tests, n_params);
gof_collimated   = cell(n_tests, 1);
gof_uncollimated = cell(n_tests, 1);

% Fit piecewise function to each perturbed dataset
for i = 1:n_tests
    [f_col, gof_collimated{i}]   = fit(Co_collimated(:, 1), normalise(perturbed_data_8(i, :)).', piecewise_fit, opts);
    [f_unc, gof_uncollimated{i}] = fit(Co(:, 1),            normalise(perturbed_data_4(i, :)).', piecewise_fit, opts);

    coeffs_col(i, :) = coeffvalues(f_col);
    coeffs_unc(i, :) = coeffvalues(f_unc);
end

% Min and max of each parameter across all tests
for j = 1:n_params
    p = params(j);
    unc.(p) = [min(coeffs_unc(:, j)), max(coeffs_unc(:, j))];
    col.(p) = [min(coeffs_col(:, j)), max(coeffs_col(:, j))];
end

% CALCULATE THRUST FROM ALL THESE DISTRIBUTIONS ----------------


s_accel = 1e-3; % m, distance over which acceleration takes place
m_Mo = 95.94 * 1.66054e-27; % kg. Mo atomic mass in amu from NIST (Linstrom, 2019)
t_arc = 5e-3; % s, arc duration (Anders et. al, 1996) 
I_arc = 160; % A, arc current. 
C_pulse = t_arc * I_arc; % C, total charge of one pulse
erosion_rate = 35e-9; % kilograms per Coulomb, erosion rate of molybdenum 
% (Neumann, 2015)
m_ions = C_pulse * erosion_rate; % kg, typical mass of molybdenum ionised 
% in one pulse 
n_ions = m_ions / m_Mo; % Number of Molybdenum atoms ionised in one pulse
v = 1.74e4; % ms^-1, ion velocity of Molybdenum (Yushkov et. al, 2000)
t_accel = s_accel / v; % s, time over which acceleration takes place
t_pulse = 1; % s, repetition rate of one pulse
s_y = 9e-2; % m, distance from cathode to transparency film in Anders et. al 1996

x = linspace(-0.1, 0.1, 100); % Between -10 to 10 cm
dists = zeros(n_tests, max(size(x)));
theta = atan(x/s_y); % Rad, angle between y-axis and 
% plasma velocity vector

for i = 1:n_tests
    dists(i, :) = normalise(piecewise_fit(coeffs_unc(i, :), x));
end

p_total = m_ions * v; % kg ms-1, total momentum of molybdenum ions 
% ejected in one pulse
F_total = p_total / t_pulse; % kg ms^-2=N, total force on thruster by 
% ejected Mo ions (directed away from ejected ions, assuming that 
% initial momentum of both ions and thruster is 0)

% Evaluate force at different locations
[F_x, F_y, F] = deal(zeros(n_tests, max(size(x))));
[F_x_total, F_y_total, F_total_sum] = deal(zeros(n_tests, 1));

for i = 1:n_tests
    F_y(i, :) = abs(F_total * cos(theta)) .* dists(i, :);
    % N, y-component of force for given (x,y) coordinate.
    F_x(i, :) = abs(F_total * sin(theta)) .* dists(1, :);
    % N, x-component of force for given (x,y) coordinate.
    F(i, :) = (F_y(i, :).^2 + F_x(i, :).^2).^(1/2);
    % N, total force (at angle theta to y-axis) for given (x,y) coordinate

    F_y_total(i) = sum(F_y(i , :));
    F_x_total(i) = sum(F_x(i , :));
    F_total_sum(i) = sum(F(i, :));
end

fprintf("Mean total thrust of uncollimated plasma plume: %7.6e Newtons.\n " + ...
    "Standard deviation = %7.6e Newtons.\n", mean(F_y_total), std(F_y_total));

% Repeat for collimated tests
for i = 1:n_tests
    dists(i, :) = normalise(piecewise_fit(coeffs_col(i, :), x));
end

[F_x, F_y, F] = deal(zeros(n_tests, max(size(x))));
[F_x_total, F_y_total, F_total_sum] = deal(zeros(n_tests, 1));

for i = 1:n_tests
    F_y(i, :) = abs(F_total * cos(theta)) .* dists(i, :);
    F_x(i, :) = abs(F_total * sin(theta)) .* dists(1, :);
    F(i, :) = (F_y(i, :).^2 + F_x(i, :).^2).^(1/2);
    
    F_y_total(i) = sum(F_y(i , :));
    F_x_total(i) = sum(F_x(i , :));
    F_total_sum(i) = sum(F(i, :));
end

fprintf("Mean total thrust of collimated plasma plume: %7.6e Newtons.\n " + ...
    "Standard deviation = %7.6e Newtons.\n", mean(F_y_total), std(F_y_total));