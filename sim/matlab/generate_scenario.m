function prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, varargin)
%GENERATE_SCENARIO  Cell-Free ISAC scenario with realistic channel/steering models
%
%   prm = generate_scenario(M, Nt, K, P, N_theta, Pmax_dBm, Gamma_track, ...)
%   Inputs:
%       M           - number of APs
%       Nt          - antennas per AP
%       K           - number of single-antenna UEs
%       P           - number of sensing targets
%       N_theta     - target parameter dimension (1: scalar, 2: 2D position, 3: 3D)
%       Pmax_dBm    - per-AP maximum transmit power in dBm
%       Gamma_track - PCRB trace threshold: scalar (broadcast), P-vector, or
%                     'auto' (calibrated from the physical FIM upper bound,
%                     scaled by the 'Gamma_alpha' factor, default 3)
%   Optional name-value pairs:
%       'AreaSize'      - square area side length in meters (default: 400)
%       'N_req'         - required APs per target (default: 3)
%       'fc'            - carrier frequency in Hz (default: 2.8e9)
%       'sigma_c2'      - communication noise variance (default: 1e-13 ~ -100 dBm)
%       'sigma_s2'      - sensing noise variance (default: 1e-13)
%       'eps_h'         - normalized channel uncertainty (default: 0.05)
%       'gamma_k_dB'    - communication SINR target in dB (default: 0)
%       'gamma_PoD_dB'  - sensing SINR target in dB (default: 0)
%       'RicianK_dB'    - sensing Rician K-factor in dB (default: Inf = pure LoS)
%       'seed'          - RNG seed (default: 0, uses clock if negative)
%       'fim_rcond_min' - minimum reciprocal condition number accepted by
%                          the automatic PCRB calibration (default: 1e-10)
%
%   The channel model follows the paper: UE channels use Rayleigh fading with
%   3GPP-like pathloss; target channels use LoS/Rician steering with derivative
%   matrix D_p generated from true target coordinates.

p = inputParser;
addParameter(p, 'AreaSize', 400, @isnumeric);
addParameter(p, 'N_req', 3, @isnumeric);
addParameter(p, 'fc', 2.8e9, @isnumeric);
addParameter(p, 'sigma_c2', 1.0, @isnumeric);  % normalized to 1 for solver numerical stability
addParameter(p, 'sigma_s2', 1.0, @isnumeric);
addParameter(p, 'eps_h', 0.05, @isnumeric);
addParameter(p, 'gamma_k_dB', 0, @isnumeric);  % 0 dB SINR target for feasibility headroom
addParameter(p, 'gamma_PoD_dB', 0, @isnumeric);
addParameter(p, 'RicianK_dB', Inf, @isnumeric);
addParameter(p, 'seed', 0, @isnumeric);
addParameter(p, 'noise_snr_target', 1e4, @isnumeric);  % target best-AP SNR
addParameter(p, 'Gamma_alpha', 3, @isnumeric);  % safety factor for 'auto' Gamma_track
addParameter(p, 'fim_rcond_min', 1e-10, @isnumeric);
parse(p, varargin{:});
opt = p.Results;

assert(opt.N_req <= M, 'generate_scenario:NreqTooLarge', ...
    'N_req (%d) must not exceed M (%d).', opt.N_req, M);

if ~ismember(N_theta, [1, 2])
    error('generate_scenario:UnsupportedNTheta', ...
        'Only N_theta = 1 or 2 is supported by the current 2D geometry model.');
end

if opt.seed >= 0
    rng(opt.seed);
else
    rng('shuffle');
end

% System dimensions
N = M * Nt;
Pmax = 10^((Pmax_dBm - 30) / 10);  % W

% Network geometry
AreaSize = opt.AreaSize;
if M <= 4
    AP_pos = AreaSize * rand(M, 2);  % fallback for very small networks
else
    % Structured grid-like deployment with small random jitter to avoid symmetry
    n_side = ceil(sqrt(M));
    [X, Y] = meshgrid(linspace(0, AreaSize, n_side), linspace(0, AreaSize, n_side));
    grid = [X(:), Y(:)];
    AP_pos = grid(1:M, :) + AreaSize/(2*n_side) * (2*rand(M,2) - 1);
    AP_pos = max(0, min(AreaSize, AP_pos));
end     % 2D positions of APs
UE_pos   = AreaSize * rand(K, 2);     % 2D positions of UEs
Target_pos = AreaSize * rand(P, 2);   % 2D positions of targets

% Carrier wavelength for steering vectors
c = 3e8;
lambda = c / opt.fc;
% Uniform linear array (ULA) along y-axis for each AP. Antenna spacing = lambda/2.
ant_ULA = ((0:Nt-1) - (Nt-1)/2).' * (lambda/2);

H = zeros(N, K);    % stacked communication channel
G = zeros(N, P);    % stacked sensing channel (LoS/Rician steering)
D = zeros(N, N_theta, P); % sensing derivative matrix

%% Communication channels: Rayleigh fading with 3GPP-like pathloss
% Pathloss: 32.4 + 20 log10(fc) + 30 log10(d) for d in meters, fc in GHz
PL_const = 32.4 + 20*log10(opt.fc/1e9);
for k = 1:K
    for m = 1:M
        d = max(norm(UE_pos(k,:) - AP_pos(m,:)), 1);
        pl_dB = PL_const + 30*log10(d);
        pl = 10^(-pl_dB/10);  % linear channel gain
        H((m-1)*Nt + 1 : m*Nt, k) = sqrt(pl/2) * (randn(Nt,1) + 1j*randn(Nt,1));
    end
end

%% Sensing channels: LoS/Rician steering + derivative matrix
for p = 1:P
    for m = 1:M
        ap = AP_pos(m,:);
        tgt = Target_pos(p,:);
        d_vec = tgt - ap;
        d = max(norm(d_vec), 1);
        sin_phi = d_vec(2) / d;       % elevation angle sine (for ULA along y)
        cos_phi = d_vec(1) / d;
        
        % LoS steering vector
        steers = exp(-1j * 2*pi / lambda * ant_ULA * sin_phi);
        
        % Pathloss for sensing (same model as comm, can be customized)
        pl_dB = PL_const + 30*log10(d);
        pl = 10^(-pl_dB/10);
        
        if isinf(opt.RicianK_dB)
            % Pure LoS
            gp_block = sqrt(pl) * steers;
        else
            K_lin = 10^(opt.RicianK_dB/10);
            gp_block = sqrt(pl) * (sqrt(K_lin/(K_lin+1))*steers + ...
                                   sqrt(1/(K_lin+1)) * (randn(Nt,1)+1j*randn(Nt,1))/sqrt(2));
        end
        
        G((m-1)*Nt + 1 : m*Nt, p) = gp_block;
        
        % Derivative of steering w.r.t. target coordinates
        % For N_theta = 2, parameters are [x, y]
        % For N_theta = 3, parameters are [x, y, ...] (e.g., third dim if 3D)
        for n = 1:N_theta
            if n == 1
                % d/dx: derivative of phase = -j * 2pi/lambda * y_ant * d(sin_phi)/dx
                % d(sin_phi)/dx = (y - y_AP)*(-x)/d^3 ? Direct derivative:
                d_sin_phi = -(tgt(2)-ap(2))*(tgt(1)-ap(1)) / d^3; % d sin_phi / dx
            elseif n == 2
                % d/dy
                d_sin_phi = ((tgt(1)-ap(1))^2) / d^3; % d sin_phi / dy
            else
                % 3D extension: not used for 2D scenario
                d_sin_phi = 0;
            end
            deriv_phase = -1j * 2*pi / lambda * ant_ULA * d_sin_phi;
            D_block = deriv_phase .* gp_block;  % d(steering)/d theta_n
            % Optional: include amplitude derivative? Usually phase dominates.
            D((m-1)*Nt + 1 : m*Nt, n, p) = D_block;
        end
    end
end

% Keep the physical coordinate derivatives.  Do not QR-normalize these
% columns: their relative magnitude carries the geometry-dependent FIM.

% Scale H and G so that the best AP has a target receive SNR of
% noise_snr_target (linear) at Pmax. This mirrors the scaling in default_params.
for k = 1:K
    block_norms = zeros(M,1);
    for m = 1:M
        hm = H((m-1)*Nt + 1 : m*Nt, k);
        block_norms(m) = real(hm' * hm);
    end
    scale = sqrt(opt.noise_snr_target * opt.sigma_c2 / Pmax / max(block_norms));
    H(:, k) = H(:, k) * scale;
end

for p = 1:P
    block_norms = zeros(M,1);
    for m = 1:M
        gm = G((m-1)*Nt + 1 : m*Nt, p);
        block_norms(m) = real(gm' * gm);
    end
    scale = sqrt(opt.noise_snr_target * opt.sigma_s2 / Pmax / max(block_norms));
    G(:, p) = G(:, p) * scale;
    D(:,:,p) = D(:,:,p) * scale;
end

prm.H = H;
prm.G = G;
prm.D = D;
prm.AP_pos = AP_pos;
prm.UE_pos = UE_pos;
prm.Target_pos = Target_pos;

prm.M = M;
prm.Nt = Nt;
prm.N = N;
prm.K = K;
prm.P = P;
prm.N_theta = N_theta;
prm.N_req = opt.N_req;

prm.Pmax = Pmax;
prm.sigma_c2 = opt.sigma_c2;
prm.sigma_s2 = opt.sigma_s2;
prm.eps_h = opt.eps_h;
prm.gamma_k = 10^(opt.gamma_k_dB/10) * ones(K,1);
prm.gamma_PoD = 10^(opt.gamma_PoD_dB/10) * ones(P,1);
% PCRB trace threshold: scalar (broadcast to all targets), P-vector, or 'auto'.
% 'auto' calibrates per-target thresholds from an all-AP isotropic reference
% for dedicated sensing waveforms: Gamma_p = Gamma_alpha * trace(inv(Jp_ref)).
% Each target receives an equal Pmax/P share of each AP's budget, so all target
% reference covariances together respect tr(E_m R_X) <= Pmax. This reference is
% neither a PCRB bound nor an optimum: directional covariance design can yield
% a different (and often smaller) PCRB trace.
gamma_track_auto = (ischar(Gamma_track) || isstring(Gamma_track)) && strcmpi(Gamma_track, 'auto');
if gamma_track_auto
    R_ref = zeros(N);
    for m = 1:M
        R_ref((m-1)*Nt+1:m*Nt, (m-1)*Nt+1:m*Nt) = (Pmax / (P * Nt)) * eye(Nt);
    end
    Gamma_track = zeros(P, 1);
    for p = 1:P
        Jp_ref = 2 * real(prm.D(:,:,p)' * R_ref * prm.D(:,:,p)) / opt.sigma_s2;
        Jp_ref = (Jp_ref + Jp_ref') / 2;
        rcond_ref = rcond(Jp_ref);
        if ~isfinite(rcond_ref) || rcond_ref < opt.fim_rcond_min
            error('generate_scenario:UnobservableReferenceTarget', ...
                ['Target %d has an ill-conditioned isotropic-reference FIM ' ...
                 '(rcond %.3e). The reference geometry is unobservable; use ' ...
                 'a different geometry or specify Gamma_track explicitly.'], ...
                p, rcond_ref);
        end
        Gamma_track(p) = opt.Gamma_alpha * trace(inv(Jp_ref));
    end
elseif isscalar(Gamma_track)
    Gamma_track = Gamma_track * ones(P, 1);
end
prm.Gamma_track = Gamma_track(:);
prm.gamma_track_auto = gamma_track_auto;

prm.use_s_procedure = true;
prm.enable_sensing_sinr = true;
prm.enable_pcrb = true;
% Dedicated sensing waveforms are treated as interference at UEs unless an
% explicit receiver-side cancellation assumption is enabled by the caller.
prm.sensing_waveform_cancelled_at_ue = false;
% Use CVX's bundled SDPT3 by default. Set prm.solver = 'mosek' explicitly
% only after cvx_setup reports MOSEK ready.
prm.solver = 'sdpt3';
prm.active_targets = 1:P;

prm.mosek_tol_rel_gap = 1e-8;
prm.mosek_tol_pfeas = 1e-9;
prm.seed = opt.seed;

end
