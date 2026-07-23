function smoke_test_paper()
%SMOKE_TEST_PAPER  Quick sanity check for the new realistic scenario generator

fprintf('Generating realistic Cell-Free ISAC scenario...\n');
prm = generate_scenario(16, 4, 4, 3, 2, 30, 10, ...
    'AreaSize', 400, 'N_req', 3, 'eps_h', 0.05, 'seed', 2026);

fprintf('M=%d, Nt=%d, K=%d, P=%d, N_theta=%d, Pmax=%.2f (%.1f dBm)\n', ...
    prm.M, prm.Nt, prm.K, prm.P, prm.N_theta, prm.Pmax, 10*log10(prm.Pmax));

fprintf('\nRunning Algorithm 2 (proposed)...\n');
res = baseline_alg2(prm, 80, 1e-5, 1.0, 1.0, 1.3, true);

if ~contains(res.status, 'Solved')
    fprintf('Proposed failed: %s\n', res.status);
    return;
end

fprintf('\n=== Proposed result ===\n');
fprintf('Status: %s\n', res.status);
fprintf('Final power: %.4f\n', res.final_obj);
fprintf('Sum rate: %.4f bit/s/Hz\n', res.sum_rate);
fprintf('Sensing SINR (dB): '); fprintf('%.2f ', res.sens_sinr_db); fprintf('\n');
fprintf('PCRB trace: '); fprintf('%.4f ', res.pcrb); fprintf('\n');
fprintf('Max violation: %.2e\n', res.max_violation);
fprintf('Binary converged: %d\n', res.binary_converged);

fprintf('\nRunning heuristic AP-selection baseline...\n');
b_fixed = zeros(prm.M, prm.P);
for p = 1:prm.P
    dists = sqrt(sum((prm.AP_pos - prm.Target_pos(p,:)).^2, 2));
    [~, idx] = sort(dists, 'ascend');
    b_fixed(idx(1:prm.N_req), p) = 1;
end
res_heur = solve_p3_with_fixed_b(prm, b_fixed, 80, 1e-5, 1.0, 1.0, 1.3);

if contains(res_heur.status, 'Solved')
    fprintf('Heuristic final power: %.4f\n', res_heur.final_obj);
else
    fprintf('Heuristic failed: %s\n', res_heur.status);
end

end
