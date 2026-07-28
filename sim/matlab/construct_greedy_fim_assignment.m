function [b, detail] = construct_greedy_fim_assignment(prm, criterion)
%CONSTRUCT_GREEDY_FIM_ASSIGNMENT  Geometry-first AP-target association.
%   Builds exactly N_req APs per target using a local-FIM proxy
%   sum_m D_{m,p}'D_{m,p}.  It is a lightweight sensor-selection heuristic;
%   fixed-b SDP re-optimization remains the physical feasibility authority.
%
%   criterion: 'doptimal' (default) maximizes log det; 'eoptimal' maximizes
%   the normalized weakest FIM eigenvalue.

if nargin < 2 || isempty(criterion), criterion = 'doptimal'; end
M = prm.M; P = prm.P; Nt = prm.N / M; N_theta = prm.N_theta;
b = zeros(M,P);
detail.selected = cell(P,1);
detail.score_trace = cell(P,1);
for p = prm.active_targets
    Dp = prm.D(:,:,p);
    local_info = cell(M,1);
    local_energy = zeros(M,1);
    for m = 1:M
        rows = (m-1)*Nt + (1:Nt);
        Dm = Dp(rows,:);
        local_info{m} = real(Dm' * Dm);
        local_energy(m) = trace(local_info{m});
    end
    ridge = max(mean(local_energy) / max(N_theta,1), 1e-12) * 1e-6;
    J = ridge * eye(N_theta);
    selected = zeros(prm.N_req,1);
    score_trace = zeros(prm.N_req,1);
    available = true(M,1);
    for q = 1:prm.N_req
        scores = -inf(M,1);
        for m = find(available).'
            J_trial = J + local_info{m};
            switch lower(criterion)
                case 'doptimal'
                    scores(m) = real(log(det(J_trial)) - log(det(J)));
                case 'eoptimal'
                    scale = max(trace(J_trial) / max(N_theta,1), 1e-12);
                    scores(m) = min(real(eig(J_trial))) / scale;
                otherwise
                    error('criterion must be doptimal or eoptimal.');
            end
        end
        [score_trace(q), m_star] = max(scores);
        selected(q) = m_star;
        available(m_star) = false;
        J = J + local_info{m_star};
    end
    b(selected,p) = 1;
    detail.selected{p} = selected;
    detail.score_trace{p} = score_trace;
end
end
