% Reconstruction script for the simulated GATE projections of the pie
% phantom, comparing different quasi-Newton methods: GD-BB, L-BFGS and GNK.

%
% Date: 22-11-2021
% Author: Nathanael Six
% Affiliation: imec-VisionLab, University of Antwerp, Belgium
% contact: nathanael.six@uantwerpen.be


%% read in data
user_path = './'; % define path to .h5 file here
phant_gate = h5read([user_path, 'PiePhantomGATE.h5'], '/phantom'); % ground truth phantom
sino_pac = h5read([user_path, 'PiePhantomGATE.h5'], '/sinogram'); % simulated GATE sinogram
angles = h5read([user_path, 'PiePhantomGATE.h5'], '/angles'); % corresponding projection angles in radians
spectrum_est =  h5read([user_path, 'PiePhantomGATE.h5'], '/estimated_spectrum'); % estimated normalised spectrum. energies in MeV
mu_plex = h5read([user_path, 'PiePhantomGATE.h5'], '/plexiglass_attenuation'); % attenuation coefficients for plexiglass from NIST. unit: cm^2/g
mu_al = h5read([user_path, 'PiePhantomGATE.h5'], '/aluminium_attenuation'); % attenuation coefficients for aluminium from NIST. unit: cm^2/g
mu_fe = h5read([user_path, 'PiePhantomGATE.h5'], '/iron_attenuation'); % attenuation coefficients for iron from NIST. unit: cm^2/g
mu_h2o = h5read([user_path, 'PiePhantomGATE.h5'], '/water_attenuation'); % attenuation coefficients for water from NIST. unit: cm^2/g

%% geometric parameters 
num_angles = numel(angles); % can select a number smaller than 300 for limited data experiments
geom = struct();
geom.px = 0.375;
geom.SOD = 75; 
geom.SDD = 300;
geom.vol_size = [400 400];
geom.proj_size = [num_angles 400];
geom.magn = geom.SDD / geom.SOD;
geom.vx = geom.px / geom.magn;

geom.angles = -angles(1:num_angles);
geom.sino = sino_pac(1:num_angles, :);
geom.vol = astra_create_vol_geom(geom.vol_size);
geom.proj = astra_create_proj_geom('fanflat', geom.px/geom.vx, 400,...
    double(geom.angles), geom.SOD/geom.vx, (geom.SDD - geom.SOD)/geom.vx);

A = opTomo('cuda', geom.proj, geom.vol); % opTomo projector from the astra toolbox (requires opSpot operators)


ref_col_poly = 48; % chosen reference energy, as the index of the bin, not in kev!

iter_flag = 'lo'; % 'hi' or 'lo'


%% Spectrum
n_bins = numel(spectrum_est(:,1));
geom.energies = spectrum_est(:,1);
geom.spect = spectrum_est(:,2);

%% Materials parsing

density_water = 1.0;
mu_h2o = mu_h2o * density_water; % to 1/cm
mu_h2o = mu_h2o * 0.1; % to 1/mm
mu_h2o = mu_h2o * geom.vx; % to 1/voxel

density_plexiglass = 1.190;
mu_plex = mu_plex * density_plexiglass; % to 1/cm
mu_plex = mu_plex * 0.1; % to 1/mm
mu_plex = mu_plex * geom.vx; % to 1/voxel

density_al = 2.7;
mu_al = mu_al * density_al; % to 1/cm
mu_al = mu_al * 0.1; % to 1/mm
mu_al = mu_al * geom.vx; % to 1/voxel


density_fe = 7.874;
mu_fe = mu_fe * density_fe; % to 1/cm
mu_fe = mu_fe * 0.1; % to 1/mm
mu_fe = mu_fe * geom.vx; % to 1/voxel

%% select material basis
% mu_h2o, mu_plex, mu_al and mu_fe are available from the .h5 file

% mu = [zeros(1, n_bins); mu_h2o'; mu_fe'];
% mu = [zeros(1, n_bins); mu_plex'; mu_al'];
% mu = [zeros(1, n_bins); mu_plex'; mu_fe'];
mu = [zeros(1, n_bins); mu_plex'; mu_al'; mu_fe'];


%% give phantom the expected values at the selected reference energy
mu_phant = [zeros(1, n_bins); mu_plex'; mu_al'];

phant_gate_poly = phant_gate;
for kk = 0:2
    phant_gate_poly(phant_gate == kk) = mu_phant(kk+1, ref_col_poly);
end


%% run reconstructions
for time_iter = 0:1  % do both a timed run (without error collection) and an untimed run (with error collection)

if time_iter == 0
    time_run = 'yes';
else
    time_run = 'no';
end


%% gradient descent - BB
start_tic = tic;
options_gd = struct('is_logcorrected', true, 'gt', phant_gate_poly);

GDBB_class = AstraQuasiNewtonPoly(A, zeros(A.n, 1), geom.sino(:), geom.spect(:), mu, mu(:, ref_col_poly), options_gd);

if strcmpi(iter_flag, 'hi')
    grad_iters = 1000;
elseif strcmpi(iter_flag, 'lo')
   grad_iters = 20; 
else
    error('Unknown iter_flag.')
end
if strcmpi(time_run, 'yes')
    GDBB_class.calculate_errors = false;
    GDBB_class.verbose = false;
else
    GDBB_class.calculate_errors = true;
end

GDBB_class.grad_desc_line_search = 'bb';
GDBB_class.initialize;
GDBB_class.run_gradient_descent(grad_iters);

if strcmpi(time_run, 'yes')
end_tic_grad_desc = toc(start_tic);
fprintf("Gradient descent time: %g.\n",end_tic_grad_desc)
end



%% gradient descent tvmin - BB
start_tic = tic;
options_gd_tvmin = struct('is_logcorrected', true, 'is_regularized', true, ...
    'reg_method', 'tvmin', 'reg_lambda', 1, 'tv_beta', 1e-7 * max(mu(:, ref_col_poly)), 'gt', phant_gate_poly);
GDBB_tvmin_class = AstraQuasiNewtonPoly(A, zeros(A.n, 1), geom.sino(:), geom.spect(:), mu, mu(:, ref_col_poly), options_gd_tvmin);

if strcmpi(iter_flag, 'hi')
    grad_iters = 1000;
elseif strcmpi(iter_flag, 'lo')
   grad_iters = 20; 
else
    error('Unknown iter_flag.')
end
if strcmpi(time_run, 'yes')
    GDBB_tvmin_class.calculate_errors = false;
    GDBB_tvmin_class.verbose = false;
else
    GDBB_tvmin_class.calculate_errors = true;
end
GDBB_tvmin_class.grad_desc_line_search = 'bb';
GDBB_tvmin_class.initialize;
GDBB_tvmin_class.run_gradient_descent(grad_iters);

if strcmpi(time_run, 'yes')
end_tic_grad_desc_tvmin = toc(start_tic);
fprintf("Gradient descent tvmin time: %g.\n",end_tic_grad_desc_tvmin)
end

%% Gauss Newton Krylov method
start_tic = tic;
options_GNK = struct('is_logcorrected', true, 'gt', phant_gate_poly);
GNK_class = AstraQuasiNewtonPoly(A, zeros(A.n, 1), geom.sino(:), geom.spect(:), mu, mu(:, ref_col_poly), options_GNK);

if strcmpi(iter_flag, 'hi')
    GNK_class.iters_newton = 200;
elseif strcmpi(iter_flag, 'lo')
    GNK_class.iters_newton = 4; 
else
    error('Unknown iter_flag.')
end
if strcmpi(time_run, 'yes')
    GNK_class.calculate_errors = false;
    GNK_class.verbose = false;
else
    GNK_class.calculate_errors = true;
end

GNK_class.iters_minres = 5;
GNK_class.initialize;
GNK_class.run;

if strcmpi(time_run, 'yes')
end_tic_GNK = toc(start_tic);
fprintf("Gauss-Newton-Krylov time: %g.\n",end_tic_GNK)
end



%% Gauss Newton Krylov reg TVmin
start_tic = tic;
options_GNK_tvmin = struct('is_logcorrected', true, 'is_regularized', true, ...
    'reg_method', 'tvmin', 'reg_lambda', 1, 'tv_beta', 1e-7 * max(mu(:, ref_col_poly)), 'gt', phant_gate_poly);
GNK_tvmin_class = AstraQuasiNewtonPoly(A, zeros(A.n, 1), geom.sino(:), geom.spect(:), mu, mu(:, ref_col_poly), options_GNK_tvmin);

if strcmpi(iter_flag, 'hi')
    GNK_tvmin_class.iters_newton = 200;
elseif strcmpi(iter_flag, 'lo')
    GNK_tvmin_class.iters_newton = 4; 
else
    error('Unknown iter_flag.')
end
if strcmpi(time_run, 'yes')
    GNK_tvmin_class.calculate_errors = false;
    GNK_tvmin_class.verbose = false;
else
    GNK_tvmin_class.calculate_errors = true;
end
GNK_tvmin_class.iters_minres = 5;
GNK_tvmin_class.initialize;
GNK_tvmin_class.run;

if strcmpi(time_run, 'yes')
end_tic_GNK_reg_tvmin = toc(start_tic);
fprintf("Gauss-Newton-Krylov tvmin time: %g.\n",end_tic_GNK_reg_tvmin)
end

%% LBFGS
start_tic = tic;
if strcmpi(iter_flag, 'hi')
    iters_bfgs = 1000;
elseif strcmpi(iter_flag, 'lo')
    iters_bfgs = 20; 
else
    error('Unknown iter_flag.')
end


options_lbfgs = struct('is_logcorrected', true, 'is_regularized', false, ...
    'gt', phant_gate_poly, 'bfgs_line_search', 'backtracking', ...
    'bfgs_iters', iters_bfgs, 'bfgs_mem', 2);



LBFGS_class = AstraQuasiNewtonPoly(A, zeros(A.n, 1), ...
    geom.sino(:), geom.spect(:), mu, mu(:, ref_col_poly), options_lbfgs);

if strcmpi(time_run, 'yes')
    LBFGS_class.calculate_errors = false;
    LBFGS_class.verbose = false;    
else
    LBFGS_class.calculate_errors = true;
end

LBFGS_class.initialize;
LBFGS_class.run_lbfgs;

if strcmpi(time_run, 'yes')
end_tic_lbfgs = toc(start_tic);
fprintf("L-BFGS time: %g.\n",end_tic_lbfgs)
end



%% LBFGS TVmin
start_tic = tic;
if strcmpi(iter_flag, 'hi')
    iters_bfgs = 1000;
elseif strcmpi(iter_flag, 'lo')
    iters_bfgs = 20; 
else
    error('Unknown iter_flag.')
end

options_lbfgs_reg_tvmin = struct('is_logcorrected', true, 'is_regularized', true, ...
    'reg_method', 'tvmin', 'reg_lambda', 1, 'tv_beta', 1e-7 * max(mu(:, ref_col_poly)), ...
    'gt', phant_gate_poly, 'bfgs_line_search', 'backtracking', ...
    'bfgs_iters', iters_bfgs, 'bfgs_mem', 2);

LBFGS_tvmin_class = AstraQuasiNewtonPoly(A, zeros(A.n, 1), ...
    geom.sino(:), geom.spect(:), mu, mu(:, ref_col_poly), options_lbfgs_reg_tvmin);

if strcmpi(time_run, 'yes')
    LBFGS_tvmin_class.calculate_errors = false;
    LBFGS_tvmin_class.verbose = false;
else
    LBFGS_tvmin_class.calculate_errors = true;
end


LBFGS_tvmin_class.initialize;
LBFGS_tvmin_class.run_lbfgs;

if strcmpi(time_run, 'yes')
end_tic_lbfgs_reg_tvmin = toc(start_tic);
fprintf("L-BFGS tvmin time: %g.\n",end_tic_lbfgs_reg_tvmin)
end


end % time_iter








