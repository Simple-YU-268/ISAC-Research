function prm = default_params()
%DEFAULT_PARAMS  Generate a simple test scenario with feasible channels
%   Returns a struct with all fields needed by solve_p3_sca_t.

rng(42);

% Network
M = 6;          % APs
Nt = 4;         % antennas per AP
N = M * Nt;     % stacked dimension
K = 3;          % UEs
P = 3;          % targets
N_req = 2;      % APs per target
N_theta = 2;    % target parameter dimension (1 scalar ranging, 2 2D position, 3 3D)

% Target SNR at best AP = 20 dB (linear 100) so gamma=0 dB is feasible
Pmax = 0.1;     % 20 dBm in linear
sigma_c2 = 1.0;
sigma_s2 = 1.0;
noise_snr_target = 100;

% Random AP positions (square area)
AP_pos = 1000 * rand(M, 2);
UE_pos = 1000 * rand(K, 2);
Target_pos = 1000 * rand(P, 2);

H = zeros(N, K);     % stacked comm channel
G = zeros(N, P);     % stacked sensing steering matrix
D = zeros(N, N_theta, P);  % sensing derivative matrix

for k = 1:K
    for m = 1:M
        d = norm(UE_pos(k,:) - AP_pos(m,:)) + 20;
        pl = 1 / d^2;       % pathloss
        H((m-1)*Nt + 1 : m*Nt, k) = sqrt(pl/2) * (randn(Nt,1) + 1j*randn(Nt,1));
    end
end

for p = 1:P
    gp = zeros(N, 1);
    for m = 1:M
        d_vec = Target_pos(p,:) - AP_pos(m,:);
        d = norm(d_vec) + 20;
        pl = 1 / d^2;
        gp((m-1)*Nt + 1 : m*Nt) = sqrt(pl/2) * (randn(Nt,1) + 1j*randn(Nt,1));
    end
    G(:, p) = gp;

    % Sensing derivative matrix D: for N_theta=1 use G itself; for N_theta>1
    % generate N_theta orthogonal directions sharing the same spatial structure.
    Dp = zeros(N, N_theta);
    Dp(:, 1) = gp;
    for n = 2:N_theta
        % random vector with similar block structure, then orthogonalize w.r.t. previous columns
        v = zeros(N, 1);
        for m = 1:M
            d = norm(Target_pos(p,:) - AP_pos(m,:)) + 20;
            pl = 1 / d^2;
            v((m-1)*Nt + 1 : m*Nt) = sqrt(pl/2) * (randn(Nt,1) + 1j*randn(Nt,1));
        end
        for nn = 1:n-1
            v = v - (Dp(:, nn)' * v) / (Dp(:, nn)' * Dp(:, nn)) * Dp(:, nn);
        end
        Dp(:, n) = v;
    end
    D(:, :, p) = Dp;
end

% Per-UE / per-target scaling so that best-AP SNR = noise_snr_target
for k = 1:K
    block_norms = zeros(M,1);
    for m = 1:M
        hm = H((m-1)*Nt + 1 : m*Nt, k);
        block_norms(m) = real(hm' * hm);
    end
    scale = sqrt(noise_snr_target * sigma_c2 / Pmax / max(block_norms));
    H(:, k) = H(:, k) * scale;
end

for p = 1:P
    block_norms = zeros(M,1);
    for m = 1:M
        gm = G((m-1)*Nt + 1 : m*Nt, p);
        block_norms(m) = real(gm' * gm);
    end
    scale = sqrt(noise_snr_target * sigma_s2 / Pmax / max(block_norms));
    G(:, p) = G(:, p) * scale;
    D(:,:,p) = D(:,:,p) * scale;
end

prm.H = H;
prm.D = D;       % N x N_theta x P sensing derivative matrix
prm.G = G;       % kept for compatibility (sensing SINR)
prm.N_theta = N_theta;
prm.eps_h = 0.01;
prm.gamma_k = ones(K, 1);       % 0 dB
prm.gamma_PoD = ones(P, 1);     % 0 dB
prm.Gamma_track = 10 * ones(P, 1);  % PCRB trace threshold
prm.sigma_c2 = sigma_c2;
prm.sigma_s2 = sigma_s2;
prm.Pmax = Pmax;
prm.N_req = N_req;
prm.N = N;
prm.M = M;
prm.K = K;
prm.P = P;
% Default: all targets active
prm.active_targets = 1:P;
prm.use_s_procedure = true;  % robust S-Procedure for SINR (eps_h is relative to ||hk||)
prm.solver = 'mosek';          % default solver for SCA subproblems

end
