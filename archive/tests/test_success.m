% Test success logic
clear;

violation = 0.0;
success = (violation <= 0);
fprintf('violation=%f, success=%d\n', violation, success);

violation = 1e-15;
success = (violation <= 0);
fprintf('violation=%e, success=%d\n', violation, success);

% Check if there's a tiny negative value
violation = -1e-15;
success = (violation <= 0);
fprintf('violation=%e, success=%d\n', violation, success);
