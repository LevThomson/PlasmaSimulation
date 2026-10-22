% Reads in data from figures 4 (uncollimated plasma) and 8 (collimated plasma) from Anders et. al, 1996
% then attempts to fit a curve to both plots, and interpolate a number of
% distributions between them, to determine how the normalised film thickness might
% change due to collimation. 

% Curve fit data from Fig 4, Normalised distribution of film thickness for
% various cathode materials. Plasma source without solenoid, filter and
% multicusp. 
% Plasma Distribution of Cathodic Arc Deposition System
% Anders et. al, 1996

% Code written by Lev Thomson for MATLAB version 24.2.0.2773142 (R2024b)

% VARIABLES --------------------------------

clear all;
close all;

% Force imparted by 1 Mo ion accelerating to velocity V over distance ~ 1mm
% from cathode spot (Ion velocity roughly constant in ion thruster -
% almost all acceleration takes place within the vicinity of the cathode
% spot. Ref: C. Wieckert, Contrib. Plasma Phys. 27, 309 ~1987)

s_accel = 1e-3; % m, distance over which acceleration takes place

% Molybdenum mean charge is +3.1 (Yushkov et. al, 2000)

m_Mo = 95.94 * 1.66054e-27; % kg. Mo atomic mass in amu from NIST, J. 
% Emsley, The Elements, Oxford Chemistry Guides (Oxford Univ. Press, New 
% York, NY, 1995).

t_arc = 5e-3; % s, arc duration (Anders et. al, 1996) 
% NOTE: This is the arc duration but not the pulse duration. 
% Arc just kicks off the plasma pulse. 
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

plotfig = true; % Optional, set to 0 to hide plots. 

s_y = 9e-2; % m, distance from cathode to transparency film in Anders et. al 1996

mode = "piecewise_fit"; % "piecewise_fit", "poly5" or "sin4" - curve fitter type. 

n_dists = 10; % Number of partially-collimated plasma distributions to create


% FUNCTIONS ---------------------------------

function normalised_array = normalise(arr)
    % Normalise distribution
    total_arr = sum(arr);
    normalised_array = arr / total_arr;
end

function [trimmed_fx, trimmed_x] = trim(fx, x)
    % Removes all entries from a distribution other than the centre part. This
    % prevents errors in the FindFWHM function

    middle_index = floor(max(size(fx)) / 2);
    [~, middle_peak_index] = max(fx(middle_index-20:middle_index+20)); % Identify the central peak of fx (may not be exactly aligned with the centre)
    middle_peak_index = middle_index - 20 + middle_peak_index;

    check_val = middle_peak_index; % Value to check
    middle_section = []; % Array indices corresponding to the central peak of fx

    % Define right side bound of middle section
    while check_val < max(size(fx))-20 && fx(check_val) > 0 % Next value is appended if it's smaller than the last - should stop at next local minimum
        [~, local_min_idx] = min(fx(check_val-20:check_val+20)); % Find local min
        
        if check_val == middle_peak_index % if middle (1st loop)
            middle_section = [middle_section, check_val]; % Append
        elseif local_min_idx == check_val % Found local minimum on right side 
            break;
        else % haven't reached next local minimum yet
            middle_section = [middle_section, check_val]; % Append
        end
        check_val = check_val + 1;
    end

    check_val = middle_peak_index;
    % Define left side bound of middle section
    while check_val > 20 && fx(check_val) > 0
        [~, local_min_idx] = min(fx(check_val-20:check_val+20)); % Find local min

        if check_val == middle_peak_index % If middle
            middle_section = [check_val, middle_section];
        elseif check_val == local_min_idx % Found local minimum on left side
            break;
        else % haven't reached next local minimum yet
            middle_section = [check_val, middle_section];
        end
        check_val = check_val - 1;
    end

    % Cut fx to include only middle section
    trimmed_fx = fx(middle_section);
    trimmed_x = x(middle_section);

end

function FWHM = findFWHM(x, fx)

    % Function to find full-width at half maximum of any function
    % Adapted from code written by:
    % (Wyatt, A, 28 Aug 2007, FindFWHM, (version 1.0.0.0), 
    % https://au.mathworks.com/matlabcentral/fileexchange/16130-findfwhm?s_tid=srchtitle


    [fx, x] = trim(fx, x);
    
    [m, n] = max(fx);		%	Find maximum value and index
    %FWHM = interp1(fx(end:-1:n), x(end:-1:n), m/2, 'spline') - interp1(fx(1:n), x(1:n), m/2, 'spline');
    ind = find(fx>=m/2);	%	Find indices where I>=max(I)/2
    nl = min(ind);			%	Leftmost index
    nr = max(ind);			%	Rightmost index
    
    %	Linear interpolate x positions
    xl = (x(nl)-x(nl-1))*(m/2-fx(nl-1))/(fx(nl)-fx(nl-1)) + x(nl-1);
    xr = (x(nr)-x(nr-1))*(m/2-fx(nr-1))/(fx(nr)-fx(nr-1)) + x(nr-1);
    
    %	Get FWHM
    FWHM = abs(xr-xl);
end


% RAW DATA & CURVE FITTING -------------------

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

% Need to use either Co or Al as these are the only two that are in both
% the collimated and uncollimated plots in Anders et. al, 1996. 
% Chose Co as it's closer in mass to Mo, but we don't expect it to behave
% exactly the same. Just trying to get an idea of how to model this
% mathematically. 

if plotfig % Plot raw data

    fig0 = figure();
    plot(Pt(:, 1), normalise(Pt(:, 2)))
    hold on
    plot(Pd(:, 1), normalise(Pd(:, 2)))
    plot(Co(:, 1), normalise(Co(:, 2)), "red")
    plot(Mo(:, 1), normalise(Mo(:, 2)))
    plot(Al(:, 1), normalise(Al(:, 2)), "blue")
    title({'Normalised distribution of the film thickness',' for an uncollimated cathodic arc deposition system'})
    legend("Platinum", "Palladium", "Cobalt", "Molybdenum", "Aluminium")
    xlabel("Displacement (m)")
    ylabel("Normalised Film Thickness (no units)")
    xlim([-0.08 0.08])
    ylim([0 8e-3])
    fontsize(16,"points")
    hold off
    saveas(fig0, "images/AndersUncollimated", "png");

end

% Also plot the collimated data:
if plotfig % Plot raw data

    fig = figure();
    plot(Fe_collimated(:, 1), normalise(Fe_collimated(:, 2)))
    hold on
    plot(Co_collimated(:, 1), normalise(Co_collimated(:, 2)), "red")
    plot(Ta_collimated(:, 1), normalise(Ta_collimated(:, 2)))
    plot(Al_collimated(:, 1), normalise(Al_collimated(:, 2)), "blue")
    title({'Normalised distribution of the film thickness',' for a collimated cathodic arc deposition system'})
    legend("Iron", "Cobalt", "Tantalum", "Aluminium")
    xlabel("Displacement (m)")
    ylabel("Normalised Film Thickness (no units)")
    xlim([-0.08 0.08])
    ylim([0 8e-3])
    fontsize(16,"points")
    hold off
    saveas(fig, "images/AndersCollimated", "png");

end



% Normalise both so area under curve is the same
Co(:, 2) = normalise(Co(:,2));

Co_collimated(:, 2) = normalise(Co_collimated(:, 2));

if plotfig % Plot collimated and uncollimated raw data for one element
    fig1 = figure();
    plot(Co_collimated(:, 1), Co_collimated(:, 2))
    hold on
    plot(Co(:, 1), Co(:, 2))
    legend("Cobalt, collimated", "Cobalt, uncollimated")
    xlabel("Displacement (m)")
    ylabel("Normalised Film Thickness (no units)")
    title({'Normalised distribution of the film thickness',' for a cathodic arc deposition system'})
    fontsize(16,"points")
    hold off
    saveas(fig1, "images/AndersCollimatedComparison", "png");
end

% Fit data to a curve

if mode == "piecewise_fit"
    % Define fit
    piecewise_fit = fittype('piecewise_ft(xL, xR, a, b, d, e, g, h, k, x)', 'independent', 'x', 'dependent', 'y', 'coefficients', {'xL', 'xR', 'a', 'b', 'd', 'e', 'g', 'h', 'k'});
    opts = fitoptions(piecewise_fit);
    opts.StartPoint = [-0.015, 0.015, 41, 100, 41, -100, -1.8, 0, 0.007];  % [a b d e g h k]
    opts.Lower      = [-1, -0.003, 0, -Inf, 0, -Inf, -Inf, -Inf, -Inf]; % xL and xR constraints are the diameter of the cathode = 6mm
    opts.Upper      = [0.003, 1, Inf, Inf, Inf, Inf, Inf, Inf, Inf];

    [fit_collimated, gof_collimated] = fit(Co_collimated(:, 1), Co_collimated(:, 2), piecewise_fit, opts);
    [fit_uncollimated, gof_uncollimated] = fit(Co(:, 1), Co(:, 2), piecewise_fit, opts);
else
    [fit_collimated, gof_collimated] = fit(Co_collimated(:, 1), Co_collimated(:, 2), mode);
    [fit_uncollimated, gof_uncollimated] = fit(Co(:, 1), Co(:, 2), mode);
end

if plotfig
    % Plot fit with data.
    fig3 = figure("Name", "Fit for Collimated Cobalt Data");
    subplot( 2, 1, 1 );
    h = plot(fit_collimated, Co_collimated(:, 1), Co_collimated(:, 2));
    legend( h, 'Co Collimated Data', 'Co Collimated Curve - '+mode, 'Location', 'NorthEast', 'Interpreter', 'none' );
    xlabel( 'X (m)', 'Interpreter', 'none' );
    ylabel( 'Normalised Film Thickness (no units)', 'Interpreter', 'none' );
    title(fig3.Name)
    grid on
    
    % Plot residuals.
    subplot( 2, 1, 2 );
    h = plot(fit_collimated, Co_collimated(:, 1), Co_collimated(:, 2), 'residuals' );
    legend( h, 'Co Collimated Curve - Residuals', 'Zero Line', 'Location', 'NorthEast', 'Interpreter', 'none' );
    xlabel( 'X (m)', 'Interpreter', 'none' );
    ylabel( 'Normalised Film Thickness (no units)', 'Interpreter', 'none' );
    grid on
    saveas(fig3, "images/CollimatedCurveFit", "png");
    
    % Plot fit with data.
    fig4 = figure("Name", "Fit for Uncollimated Cobalt Data");
    subplot( 2, 1, 1 );
    h = plot(fit_uncollimated, Co(:, 1), Co(:, 2));
    legend( h, 'Co Uncollimated Data', 'Co Uncollimated Curve - '+mode, 'Location', 'NorthEast', 'Interpreter', 'none' );
    xlabel( 'X (m)', 'Interpreter', 'none' );
    ylabel( 'Normalised Film Thickness (no units)', 'Interpreter', 'none' );
    title(fig4.Name)
    grid on
    
    % Plot residuals.
    subplot( 2, 1, 2 );
    h = plot(fit_uncollimated, Co(:, 1), Co(:, 2), 'residuals' );
    legend( h, 'Co Uncollimated Curve - Residuals', 'Zero Line', 'Location', 'NorthEast', 'Interpreter', 'none' );
    xlabel( 'X (m)', 'Interpreter', 'none' );
    ylabel( 'Normalised Film Thickness (no units)', 'Interpreter', 'none' );
    grid on
    saveas(fig4, "images/UncollimatedCurveFit", "png");

end

% Calculate FWHM of collimated and uncollimated fits

FWHM_uncol = findFWHM(Co(:,1), Co(:,2));
FWHM_uncol_angle = atan((FWHM_uncol/2)/s_y);
FWHM_col = findFWHM(Co_collimated(:, 1),Co_collimated(:, 2));
disp(fit_uncollimated)
fprintf("Full-width at half maximum of the uncollimated data: %5.4f m. \n" + ...
    "This corresponds to an angle of %5.4f radians from normal " + ...
    "(y-axis). \n", FWHM_uncol, FWHM_uncol_angle);

FWHM_col = findFWHM(Co_collimated(:,1), Co_collimated(:,2));
FWHM_col_angle = atan((FWHM_col/2)/s_y);
disp(fit_collimated)
fprintf("Full-width at half maximum of the uncollimated data: %5.4f m. \n" + ...
    "This corresponds to an angle of %5.4f radians from normal " + ...
    "(y-axis). \n", FWHM_col, FWHM_col_angle);

% INTERPOLATE PARTIALLY-COLLIMATED DISTS ---------------

% Define x-values. This is the set of points at which each fit will be
% evaluated. Also calculate corresponding angles theta between y and x
% axis.
x = linspace(-0.1, 0.1, 100); % Between -10 to 10 cm

% Define arrays of curve coefficients from uncollimated to collimated array
dists = zeros(n_dists, max(size(x)));
FWHM_dists = zeros(n_dists, 1);

if mode == "sin4"

    % fit = a1*sin(b1*x+c1) + a2*sin(b2*x+c2) + a3*sin(b3*x+c3) + a4*sin(b4*x+c4)

    a1 = linspace(fit_uncollimated.a1, fit_collimated.a1, n_dists);
    b1 = linspace(fit_uncollimated.b1, fit_collimated.b1, n_dists);
    c1 = linspace(fit_uncollimated.c1, fit_collimated.c1, n_dists);
    
    a2 = linspace(fit_uncollimated.a2, fit_collimated.a2, n_dists);
    b2 = linspace(fit_uncollimated.b2, fit_collimated.b2, n_dists);
    c2 = linspace(fit_uncollimated.c2, fit_collimated.c2, n_dists);
    
    a3 = linspace(fit_uncollimated.a3, fit_collimated.a3, n_dists);
    b3 = linspace(fit_uncollimated.b3, fit_collimated.b3, n_dists);
    c3 = linspace(fit_uncollimated.c3, fit_collimated.c3, n_dists);
    
    a4 = linspace(fit_uncollimated.a4, fit_collimated.a4, n_dists);
    b4 = linspace(fit_uncollimated.b4, fit_collimated.b4, n_dists);
    c4 = linspace(fit_uncollimated.c4, fit_collimated.c4, n_dists);

    % Evaluate each distribution at each value of x
    for i = 1:n_dists
        dists(i, :) = normalise(a1(i)*sin(b1(i)*x+c1(i)) + a2(i)*sin(b2(i)*x+c2(i)) + a3(i)*sin(b3(i)*x+c3(i)) + a4(i)*sin(b4(i)*x+c4(i)));
    end

elseif mode == "poly5"

    % fit = p1*x^5 + p2*x^4 + p3*x^3 + p4*x^2 + p5*x + p6
    
    p1 = linspace(fit_uncollimated.p1, fit_collimated.p1, n_dists);
    p2 = linspace(fit_uncollimated.p2, fit_collimated.p2, n_dists);
    p3 = linspace(fit_uncollimated.p3, fit_collimated.p3, n_dists);
    p4 = linspace(fit_uncollimated.p4, fit_collimated.p4, n_dists);
    p5 = linspace(fit_uncollimated.p5, fit_collimated.p5, n_dists);
    p6 = linspace(fit_uncollimated.p6, fit_collimated.p6, n_dists);

    % Polynomial coefficients of fit i can be accessed as row i of array dists
    poly_values = [p1; p2; p3; p4; p5; p6];

    % Evaluate each distribution at each value of x
    for i = 1:n_dists
        dists(i, :) = normalise(polyval(poly_values(:, i), x));
    end
elseif mode == "piecewise_fit"

    % fit = exponential, then poly2, then exponential

    % Make linspace for each parameter (Note c & f absent - they are found
    % by equating piecewise functions at their borders)
    % Also note i,j absent - these are reserved variable in matlab. 

    xL = linspace(fit_uncollimated.xL, fit_collimated.xL, n_dists);
    xR = linspace(fit_uncollimated.xR, fit_collimated.xR, n_dists);
    a = linspace(fit_uncollimated.a, fit_collimated.a, n_dists);
    b = linspace(fit_uncollimated.b, fit_collimated.b, n_dists);
    d = linspace(fit_uncollimated.d, fit_collimated.d, n_dists);
    e = linspace(fit_uncollimated.e, fit_collimated.e, n_dists);
    g = linspace(fit_uncollimated.g, fit_collimated.g, n_dists);
    h = linspace(fit_uncollimated.h, fit_collimated.h, n_dists);
    k = linspace(fit_uncollimated.k, fit_collimated.k, n_dists);

    % Polynomial coefficients of fit i can be accessed as row i of array dists
    piecewise_values = [xL; xR; a; b; d; e; g; h; k];

    % Evaluate each distribution & width at each value of x
    for i = 1:n_dists
        dists(i, :) = normalise(piecewise_fit(piecewise_values(:, i), x));
        FWHM_dists(i) = findFWHM(dists(i, :), x);
        % TODO angle subtended by plasma plume
    end

else 
    disp("Invalid mode");
end 

% Plot surface with x = distance in x-direction from centre of cathode, 
% y = normalised plasma distribution
% z = index of distribution (where 1 = uncollimated data from Anders paper (looks like a Gaussian),
% and 10 = n_dists = collimated data from Anders paper (looks like a top hat). Expect the
% distributions to move smoothly from a gaussian to top hat shape as z
% increases).
if plotfig
    
    fig2 = figure("Name","Normalised distribution of the " + ...
         "film thickness for a cathodic arc deposition system");
    hold on

    for i=1:n_dists
        h = plot(x, dists(i, :), 'DisplayName', "FWHM = "+FWHM_dists(i)+" m");
    end

    title(fig2.Name)
    xlabel("Displacement (m)")
    ylabel("Normalised Film Thickness (no units)")
    legend
    hold off
    saveas(fig2, "images/Andersdists", "png");
    
end

% VELOCITY ----------------------------------

theta = atan(x/s_y); % Rad, angle between y-axis and 
% plasma velocity vector

p_total = m_ions * v; % kg ms-1, total momentum of molybdenum ions 
% ejected in one pulse
F_total = p_total / t_pulse; % kg ms^-2=N, total force on thruster by 
% ejected Mo ions (directed away from ejected ions, assuming that 
% initial momentum of both ions and thruster is 0)

% Evaluate force at different locations
[F_x, F_y, F] = deal(zeros(n_dists, max(size(x))));
[F_x_total, F_y_total, F_total_sum] = deal(zeros(n_dists, 1));

for i = 1:n_dists
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

disp("--------------------- Uncollimated results ---------------------")
disp("Total force in y-direction in Newtons: " + ...
    "(x-component of force is symmetrical and cancels out) ")
disp(F_y_total(1))

fprintf("Total force: %7.6f Newtons.\n " + ...
    "Total force - force in y-direction = %7.6f Newtons.\n", F_total(1), F_total(1) - F_y_total(1));

if plotfig
    fig3 = figure("Name", "Force exerted on thruster by ions " + ...
        "with angle Theta (Uncollimated)");
    plot(theta, F_y(1, :))
    hold on
    plot(theta, F_x(1, :))
    plot(theta, F(1, :))
    xlabel("Angle Theta (radians)")
    ylabel("Force (N)")
    legend("Y-component of force", "X-component of force", "Total force")
    title(fig3.Name)
    saveas(fig3, "images/ForceComponentsUncollimated", "png");
end

disp("--------------------- Collimated results ---------------------")

disp("Total force in y-direction in Newtons: " + ...
    "(x-component of force is symmetrical and cancels out) ")
disp(F_y_total(end))

fprintf("Total force: %7.6f Newtons.\n " + ...
    "Total force - force in y-direction = %7.6f Newtons.\n", F_total(end), F_total(end) - F_y_total(end));

if plotfig
    fig4 = figure("Name", "Force exerted on thruster by ions " + ...
        "with angle Theta (Collimated)");
    plot(theta, F_y(end, :))
    hold on
    plot(theta, F_x(end, :))
    plot(theta, F(end, :))
    xlabel("Angle Theta (radians)")
    ylabel("Force (N)")
    legend("Y-component of force", "X-component of force", "Total force")
    title(fig4.Name)
    saveas(fig4, "images/ForceComponentsCollimated", "png");

    fig5 = figure("Name", "Total Thrust vs Angle Subtended By Plasma Plume");
    plot(F_y_total, FWHM_dists)
    hold on
    xlabel("Angle Subtended by Plasma Plume (radians)")
    ylabel("Thrust (N)")
    title(fig5.Name)
    saveas(fig5, "images/ThrustvsTheta", "png");
end


fprintf("Percentage change between uncollimated & colllimated thrust: %7.4f percent. ", 100*(F_y_total(end)-F_y_total(1))/F_y_total(1))
fprintf("\nPercentage change between uncollimated & total thrust: %7.4f percent. ", 100*(F_total-F_y_total(1))/F_y_total(1))