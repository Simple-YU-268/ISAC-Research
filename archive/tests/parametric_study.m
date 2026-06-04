% Parametric study: test different epsilon values
clear; rng(42);

M = 16; Nt = 4; K = 10;
Pmax = 30; 
snrReq = 3;  
sinrReq = 0; 

apPos = [];
for x = linspace(-60,60,4)
    for y = linspace(-60,60,4)
        apPos = [apPos; x, y];
    end
end

% Generate ONE scenario
targetPos = -50 + 100*rand(1,2);
userPos = -50 + 100*rand(K,2);

H = zeros(M*Nt, K);
for k = 1:K
    for m = 1:M
        d = max(norm(apPos(m,:)-userPos(k,:)), 5);
        pl = (10/d)^2.5;
        idx = (m-1)*Nt+(1:Nt);
        H(idx,k) = sqrt(pl)*(randn(Nt,1)+1j*randn(Nt,1))/sqrt(2);
    end
end

G = zeros(M*Nt,1);
for m = 1:M
    d = max(norm(apPos(m,:)-targetPos), 5);
    pl = (10/d)^2.5;
    idx = (m-1)*Nt+(1:Nt);
    G(idx) = sqrt(pl)*(randn(Nt,1)+1j*randn(Nt,1))/sqrt(2);
end

fprintf('Testing different epsilon values...\n');
fprintf('%-10s %-10s %-10s %-10s %-10s %-10s\n', 'epsilon', 'wcFactor', 'SNR(dB)', 'SINR(dB)', 'Power(W)', 'Success');

for epsilon = [0, 0.05, 0.10, 0.15, 0.20, 0.25]
    wcFactor = (1-epsilon)/(1+epsilon);
    
    % ZF
    W_zf = H * inv(H'*H);
    W = zeros(size(W_zf));
    for k = 1:K
        W(:,k) = W_zf(:,k)/norm(W_zf(:,k));
    end
    
    % Power with robustness
    gammaSinr = 10^(sinrReq/10) / wcFactor;
    sigma2 = 0.5;
    
    Pcomm = zeros(M*Nt, K);
    for k = 1:K
        hk = H(:,k);
        wk = W(:,k);
        gain = abs(hk'*wk)^2;
        pk = gammaSinr * sigma2 / gain;
        Pcomm(:,k) = sqrt(pk) * wk;
    end
    
    PcommTotal = sum(sum(abs(Pcomm).^2));
    
    % Best rho
    bestViol = inf;
    bestResult = [];
    for rho = 0.1:0.05:0.9
        PsensTotal = (1-rho)/rho * PcommTotal;
        Total = PcommTotal + PsensTotal;
        if Total > Pmax, continue; end
        
        z = G/norm(G);
        Psens = sqrt(PsensTotal)*z;
        
        snr = abs(G'*Psens)^2/sigma2;
        snrWc = snr * wcFactor;
        
        minSinr = inf;
        for k = 1:K
            hk = H(:,k);
            desired = abs(hk'*Pcomm(:,k))^2;
            interf = sigma2;
            for j = 1:K
                if j~=k, interf = interf + abs(hk'*Pcomm(:,j))^2; end
            end
            sinr = desired/interf;
            sinrWc = sinr * wcFactor;
            minSinr = min(minSinr, sinrWc);
        end
        
        vSnr = max(0, 10^(snrReq/10) - snrWc) / 10^(snrReq/10);
        vSinr = max(0, 10^(sinrReq/10) - minSinr) / 10^(sinrReq/10);
        viol = max(vSnr, vSinr);
        
        if viol < bestViol
            bestViol = viol;
            bestResult = struct('rho', rho, 'snr', 10*log10(snrWc), 'sinr', 10*log10(minSinr), ...
                'power', Total, 'viol', viol);
        end
    end
    
    if ~isempty(bestResult)
        if bestResult.viol <= 0
            status = 'YES';
        else
            status = 'NO';
        end
        fprintf('%-10.2f %-10.3f %-10.1f %-10.1f %-10.1f %-10s\n', ...
            epsilon, wcFactor, bestResult.snr, bestResult.sinr, bestResult.power, status);
    else
        fprintf('%-10.2f %-10.3f %-10s %-10s %-10s %-10s\n', ...
            epsilon, wcFactor, 'N/A', 'N/A', 'N/A', 'NO');
    end
end
