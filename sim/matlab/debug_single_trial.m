function debug_single_trial()
% Minimal reproduction: solve single relaxed problem and print diagnostics

prm = generate_scenario(8, 4, 4, 2, 2, 20, 30, ...
    'AreaSize', 400, 'N_req', 2, 'eps_h', 0, 'seed', 2026, 'gamma_k_dB', 0);

K = prm.K; P = prm.P; N = prm.N; M = prm.M;

W0 = cell(K,1);
for k = 1:K
    W0{k} = eye(N) * (prm.Pmax / prm.K / prm.M);
end
b0 = ones(M, P) * (prm.N_req / M);

[W, Z, mu, b, M_p, status] = solve_p3_sca_t(prm, W0, b0, 0, 0);
fprintf('Status: %s\n', status);

if contains(status, 'Solved')
    [max_viol, report] = validate_solution(prm, W, Z, mu, b, M_p, 1e-6);
    fprintf('Max violation: %.3e\n', max_viol);
    disp(report);
else
    fprintf('No solution returned.\n');
end

% Try with all-AP active (b=1) to see if per-AP power is bottleneck
b_all = ones(M, P);
[W2, Z2, mu2, b2, M_p2, status2] = solve_p3_sca_t(prm, W0, b_all, 0, 0, b_all);
fprintf('All-AP status: %s\n', status2);
if contains(status2, 'Solved')
    [max_viol2, report2] = validate_solution(prm, W2, Z2, mu2, b2, M_p2, 1e-6);
    fprintf('All-AP max violation: %.3e\n', max_viol2);
end

% Try without service count equality (b relaxed) and without power gate
% skip, not supported by solver directly

end
