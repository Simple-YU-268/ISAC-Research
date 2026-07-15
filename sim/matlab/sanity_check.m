% Sanity check for MATLAB implementation of (P3)
% Requires: CVX (with SDPT3 or MOSEK)

clear; clc;

prm = default_params();

fprintf('Running Algorithm 2 (double-DC SCA) in MATLAB...\n');
res = baseline_alg2(prm, 80, 1e-5, 1.0, 1.0, 1.3, true);

if ~isfield(res, 'W')
    fprintf('Algorithm failed to produce a solution: %s\n', res.status);
    return;
end

% Evaluate and validate
[max_viol, viol] = validate_solution(prm, res.W, res.Z, res.mu, res.b, res.M_p, 1e-6);

fprintf('\n');
fprintf('Status: %s\n', res.status);
fprintf('Iterations: %d\n', res.iters);
fprintf('Final power objective: %.4f\n', res.final_obj);
fprintf('Sum rate: %.4f bit/s/Hz\n', res.sum_rate);
fprintf('Sensing SINR (dB): '); fprintf('%.2f ', res.sens_sinr_db); fprintf('\n');
fprintf('PCRB trace: '); fprintf('%.4f ', res.pcrb); fprintf('\n');
fprintf('\nMax constraint violation: %.2e\n', max_viol);

fprintf('\nDetailed violations:\n');
fields = fieldnames(viol);
for i = 1:length(fields)
    fprintf('  %s: ', fields{i});
    fprintf('%.2e ', viol.(fields{i}));
    fprintf('\n');
end
fprintf('Binary converged: %d\n', res.binary_converged);
fprintf('Final power objective: %.4f\n', res.final_obj);

fprintf('\nPer-AP power:\n');
M = prm.M; Nt = prm.N / prm.M;
E = build_E_m(M, Nt);
R = zeros(prm.N, prm.N);
for k = 1:prm.K
    R = R + res.W{k};
end
R = R + res.Z;
for m = 1:M
    fprintf('  AP %d: %.4f (limit %.4f)\n', m, real(trace(E{m} * R)), prm.Pmax);
end

fprintf('\nAP-target assignment (relaxed b):\n');
for p = 1:prm.P
    fprintf('  target %d: b = [', p);
    fprintf('%.3f ', res.b(:, p));
    fprintf(']\n');
end

