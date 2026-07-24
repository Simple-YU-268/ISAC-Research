function report = check_fixed_b_pcrb(prm, b_fixed, include_sensing_sinr)
%CHECK_FIXED_B_PCRB  PCRB feasibility check for a binary AP topology.
%   Keeps the exact Schur PCRB LMIs and both per-AP power gates, while
%   removing communication constraints. Set include_sensing_sinr=true to
%   additionally impose the sensing-SINR constraints using R as sensing
%   covariance. Thus infeasibility identifies a sensing-side bottleneck.

if nargin < 3, include_sensing_sinr = false; end

N = prm.N; M = prm.M; P = prm.P; Nt = prm.Nt;
N_theta = prm.N_theta;
assert(isequal(size(b_fixed), [M, P]), 'b_fixed must be M-by-P.');
assert(all(b_fixed(:) == 0 | b_fixed(:) == 1), 'b_fixed must be binary.');
E = build_E_m(M, Nt);

cvx_begin quiet
    if isfield(prm, 'solver') && strcmpi(prm.solver, 'mosek')
        cvx_solver mosek
    else
        cvx_solver SDPT3
    end
    variable R(N,N) hermitian
    variable M_p(N_theta,N_theta,P) hermitian
    minimize(real(trace(R)))
    subject to
        R == hermitian_semidefinite(N);
        for m = 1:M
            real(trace(E{m} * R)) <= prm.Pmax * sum(b_fixed(m,:));
            real(trace(E{m} * R)) <= prm.Pmax;
        end
        for p = 1:P
            if include_sensing_sinr
                gp = prm.G(:,p);
                real(gp' * R * gp) >= prm.gamma_PoD(p) * prm.sigma_s2;
            end
            Dp = prm.D(:,:,p);
            Jp = 2 * real(Dp' * R * Dp) / prm.sigma_s2;
            if N_theta == 1
                inv_pos(Jp) <= M_p(1,1,p);
            else
                [M_p(:,:,p), eye(N_theta); eye(N_theta), Jp] == ...
                    hermitian_semidefinite(2 * N_theta);
            end
            real(trace(M_p(:,:,p))) <= prm.Gamma_track(p);
        end
cvx_end

report.status = cvx_status;
report.b = b_fixed;
report.include_sensing_sinr = include_sensing_sinr;
if contains(cvx_status, 'Solved')
    report.R = full(R);
    report.M_p = full(M_p);
    report.power = zeros(M,1);
    report.pcrb = zeros(P,1);
    for m = 1:M
        report.power(m) = real(trace(E{m} * report.R));
    end
    for p = 1:P
        Dp = prm.D(:,:,p);
        Jp = 2 * real(Dp' * report.R * Dp) / prm.sigma_s2;
        report.pcrb(p) = trace(inv(Jp));
    end
end
end
