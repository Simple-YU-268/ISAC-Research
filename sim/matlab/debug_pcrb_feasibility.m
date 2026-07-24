function debug_pcrb_feasibility()
%DEBUG_PCRB_FEASIBILITY  Check which constraint blocks the warm start

prm = default_params();
K = prm.K; P = prm.P; N = prm.N; N_theta = prm.N_theta;

% Warm start matrices
W = cell(K,1);
for k = 1:K
    W{k} = eye(N) * (prm.Pmax / prm.K / prm.M);
end
Z = eye(N) * 1e-6;
R = Z;
for k = 1:K
    R = R + W{k};
end

fprintf('Warm-start power trace: %.4f\n', real(trace(R)));

% Check SINR
for k = 1:K
    hk = prm.H(:, k);
    sig = real(hk' * W{k} * hk);
    interf = 0;
    for j = setdiff(1:K, k)
        interf = interf + real(hk' * W{j} * hk);
    end
    sinr = sig / (interf + prm.sigma_c2);
    fprintf('UE %d SINR: %.4f (target %.4f)\n', k, sinr, prm.gamma_k(k));
end

% Check sensing SINR
for p = 1:P
    gp = prm.G(:, p);
    val = real(gp' * Z * gp) / prm.sigma_s2;
    fprintf('Target %d sensing SNR: %.4f (target %.4f)\n', p, val, prm.gamma_PoD(p));
end

% Check PCRB Schur eigenvalues
for p = 1:P
    Dp = prm.D(:,:,p);
    Jp = 2 * real(Dp' * R * Dp) / prm.sigma_s2;
    fprintf('Target %d Jp eigenvalues: %.4e %.4e\n', p, eig(Jp));
    M_block = prm.Gamma_track(p) / N_theta * eye(N_theta);
    Schur = [M_block, eye(N_theta); eye(N_theta), Jp];
    d = eig(Schur, 'vector');
    fprintf('Target %d Schur mineig: %.4e\n', p, min(d));
end

% Check per-AP power
M = prm.M; Nt = prm.N / prm.M;
E = build_E_m(M, Nt);
for m = 1:M
    pwr = real(trace(E{m} * R));
    fprintf('AP %d power: %.4f (limit %.4f)\n', m, pwr, prm.Pmax);
end
end
