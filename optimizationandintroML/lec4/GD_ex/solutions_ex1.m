syms x1 x2
x0 = [1;1];  % Column vector
f = @(x) 100*x(1)^2+x(2)^2;  % Element-wise multiplication and exponentiation
grad_f = @(x) [2*x(1);2*x(2)];
Hf = @(x) [2 0; 0 2];
max_iter = 10;
% unconstrained_opt(f,grad_f,Hf,x0,max_iter,0.01,'NR',2);

X = readtable('co2_weekly_mlo.txt');

