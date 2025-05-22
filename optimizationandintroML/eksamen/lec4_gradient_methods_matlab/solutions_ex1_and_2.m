syms x1 x2
x0 = [1;1];  % Column vector
f = @(x) 1*x(1)^2+x(2)^2;  % Element-wise multiplication and exponentiation
grad_f = @(x) [2*x(1);2*x(2)];
Hf = @(x) [2 0; 0 2];
max_iter = 10;
unconstrained_opt(f,grad_f,Hf,x0,max_iter,0.01,'SD',1);



%X = readtable('co2_weekly_mlo.txt');

%%%first run alpha 100 NR
% Backtracking line search Results:
% Iterations: 0
% Minimum point: x = 0.000000
% Minimum point: x = 0.000000
% Function value at minimum: f(x) = 0.000000
% Backtracking line search Results:
% Iterations: 0
% Minimum point: x = 0.000000
% Minimum point: x = 0.000000
% Function value at minimum: f(x) = 0.000000
% Convergence reached at iteration 2.
% Minimum found at x = [0.0000 0.0000 ] with f(x) = 0.0000
% Total iterations: 2

%%%second run SD 1 aplha
% Backtracking line search Results:
% Iterations: 2
% Minimum point: x = -0.280000
% Minimum point: x = -0.280000
% Function value at minimum: f(x) = 0.156800
% Backtracking line search Results:
% Iterations: 2
% Minimum point: x = 0.078400
% Minimum point: x = 0.078400
% Function value at minimum: f(x) = 0.012293
% Backtracking line search Results:
% Iterations: 2
% Minimum point: x = -0.021952
% Minimum point: x = -0.021952
% Function value at minimum: f(x) = 0.000964
% Backtracking line search Results:
% Iterations: 2
% Minimum point: x = 0.006147
% Minimum point: x = 0.006147
% Function value at minimum: f(x) = 0.000076
% Backtracking line search Results:
% Iterations: 2
% Minimum point: x = -0.001721
% Minimum point: x = -0.001721
% Function value at minimum: f(x) = 0.000006
% Backtracking line search Results:
% Iterations: 2
% Minimum point: x = 0.000482
% Minimum point: x = 0.000482
% Function value at minimum: f(x) = 0.000000
% Convergence reached at iteration 6.
% Minimum found at x = [0.0005 0.0005 ] with f(x) = 0.0000
% Total iterations: 6