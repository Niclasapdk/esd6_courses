% Assuming y is your CO₂ data (column vector)
% Time index (or feature) and intercept for regression

co2weeklymlo = import_co2_concentration("co2_weekly_mlo.txt", [50, Inf]);

y = co2weeklymlo.co2_ppm;

n = length(y);
X = [ones(n,1), (1:n)'];  % intercept + linear term

% Define function, gradient, and Hessian
f = @(beta) 0.5 * norm(X*beta - y)^2;
grad_f = @(beta) X' * (X*beta - y);
Hf = @(beta) X' * X;

%calc condition number EXERCISE 4
H = X' * X;
cond_num_kappa = cond(H);
cond_num_r = inv(cond_num_kappa);
%result of kappa is 7.8635e+06 is very bad beacuse much bigger than 1 leading to SD 
% As a high condition number means slow convergence or instability for SD.
%r is 1.2717e-07 which is very small


% Initial guess
x0 = zeros(size(X,2),1);  % two parameters (intercept and slope)
%x0 = randn(size(X,2),1);
%x0 = [mean(y); 0];

% Maximum iterations
max_iter = 100;

% Try Steepest Descent
disp('Running Steepest Descent:');
[beta_SD, fval_SD, exitflag_SD] = unconstrained_opt(f, grad_f, Hf, x0, max_iter, 0.0001, 'SD', 2);

% Try Newton-Raphson
disp('Running Newton-Raphson:');
[beta_NR, fval_NR, exitflag_NR] = unconstrained_opt(f, grad_f, Hf, x0, max_iter, 0.0001, 'NR', 2);


%%%%%%%%%%NOTES!!
%minimim points is baseline and slope ie x = [298.3255 0.0478 ]
% linear regression model y = beta0 + beta1 * t
% the slope matchtes  approx  (412.53-298.3255)/2427=0.047056


%SD fails seen by reaching max iterations and min points are not close at all and NR succeeds
%SD first NR second
% Backtracking line search Results:
% Iterations: 96
% Minimum point: x = 0.007564
% Minimum point: x = 0.232053
% Function value at minimum: f(x) = 44872357.027000
% Minimum found at x = [0.0076 0.2321 ] with f(x) = 44872357.0270
% Total iterations: 100
% Running Newton-Raphson:
% Backtracking line search Results:
% Iterations: 0
% Minimum point: x = 298.325496
% Minimum point: x = 0.047785
% Function value at minimum: f(x) = 17890598.622164
% Backtracking line search Results:
% Iterations: 2
% Minimum point: x = 298.325496
% Minimum point: x = 0.047785
% Function value at minimum: f(x) = 17890598.622164
% Convergence reached at iteration 2.
% Minimum found at x = [298.3255 0.0478 ] with f(x) = 17890598.6222
% Total iterations: 2
