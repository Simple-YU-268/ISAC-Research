function report = check_fixed_b_pcrb(prm, b_fixed, include_sensing_sinr)
%CHECK_FIXED_B_PCRB  PCRB feasibility check for a binary AP topology.
%   Keeps the exact Schur PCRB LMIs, target-specific sensing covariances, the
%   AP-target Big-M constraints, and total per-AP power ceilings while removing
%   communication constraints. Thus infeasibility identifies a sensing-side
%   bottleneck under the selected sensing cluster.

if nargin < 3, include_sensing_sinr = false; end

N = prm.N; M = prm.M; P = prm.P; Nt = prm.N / prm.M;
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
    variable S_p(N,N,P) hermitian
    variable M_p(N_theta,N_theta,P) hermitian
    sense_obj = 0;
    for p = 1:P
        sense_obj = sense_obj + real(trace(S_p(:,:,p)));
    end
    minimize(sense_obj)
    subject to
        for m = 1:M
            real(trace(E{m} * sum(S_p,3))) <= prm.Pmax;
        end
        for p = 1:P
            S_p(:,:,p) == hermitian_semidefinite(N);
            for m = 1:M
                real(trace(E{m} * S_p(:,:,p))) <= prm.Pmax * b_fixed(m,p);
            end
            if include_sensing_sinr
                gp = prm.G(:,p);
                real(gp' * S_p(:,:,p) * gp) >= prm.gamma_PoD(p) * prm.sigma_s2;
            end
            Dp = prm.D(:,:,p);
            Jp = 2 * real(Dp' * S_p(:,:,p) * Dp) / prm.sigma_s2;
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
    report.S_p = full(S_p);
    report.R = sum(report.S_p, 3);
    report.M_p = full(M_p);
    report.power = zeros(M,1);
    report.pcrb = zeros(P,1);
    for m = 1:M
        report.power(m) = real(trace(E{m} * report.R));
    end
    for p = 1:P
        Dp = prm.D(:,:,p);
        Jp = 2 * real(Dp' * report.S_p(:,:,p) * Dp) / prm.sigma_s2;
        report.pcrb(p) = trace(inv(Jp));
    end
end
end
