%Real robot postion
xr = [1;1];
%Beacons' positions:
b1 = [2;0];
b2 = [0;2];
b3 = [-2;-2];
%Noisy measured distances:
d1 = norm(xr-b1)+0.01*randn;
d2 = norm(xr-b2)+0.01*randn;
d3 = norm(xr-b3)+0.01*randn;

F = @(x) [norm(x-b1)^2-d1^2;norm(x-b2)^2-d2^2;norm(x-b3)^2-d3^2];

Jf = @(x) 2 * [
    (x - b1)';
    (x - b2)';
    (x - b3)';
];%Write the jacobian

max_iter = 100;
%tol = 1e-8; %from mohhamed
tol = 1e-2;
stepsize_rule = 1;
x0 = [0;0];
[xmin, fmin, x, iter] = Gauss_Newton(F, Jf, x0, max_iter, tol, stepsize_rule);
error = norm(xmin-xr);
fprintf('Distance Error =  %.4f\n', error);

% Golden Section Search Results:
% Iterations: 30
% Minimum point: x = 1.026570
% Minimum point: x = 1.003373
% Function value at minimum: f(x) = 0.009201
% Golden Section Search Results:
% Iterations: 19
% Minimum point: x = 1.031366
% Minimum point: x = 1.007851
% Function value at minimum: f(x) = 0.006063
% Golden Section Search Results:
% Iterations: 6
% Minimum point: x = 1.031378
% Minimum point: x = 1.007859
% Function value at minimum: f(x) = 0.006063
% Golden Section Search Results:
% Iterations: 0
% Minimum point: x = 1.031379
% Minimum point: x = 1.007860
% Function value at minimum: f(x) = 0.006063
% Golden Section Search Results:
% Iterations: 0
% Minimum point: x = 1.031379
% Minimum point: x = 1.007860
% Function value at minimum: f(x) = 0.006063
% Golden Section Search Results:
% Iterations: 0
% Minimum point: x = 1.031379
% Minimum point: x = 1.007860
% Function value at minimum: f(x) = 0.006063
% Golden Section Search Results:
% Iterations: 0
% Minimum point: x = 1.031379
% Minimum point: x = 1.007860
% Function value at minimum: f(x) = 0.006063
% Golden Section Search Results:
% Iterations: 0
% Minimum point: x = 1.031379
% Minimum point: x = 1.007860
% Function value at minimum: f(x) = 0.006063
% Golden Section Search Results:
% Iterations: 0
% Minimum point: x = 1.031379
% Minimum point: x = 1.007860
% Function value at minimum: f(x) = 0.006063
% Convergence reached at iteration 9.
% Minimum found at x = [1.0314 1.0079 ] with f(x) = 0.0061
% Total iterations: 9
% Distance Error =  0.0323