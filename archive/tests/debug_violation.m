% Debug: print actual values during solveOneScenario
clear; rng(42);

% One scenario
M = 16; Nt = 4; K = 10;
Pmax = 30; snrReqDb = 3; sinrReqDb = 0;

apPos = [];
for x = linspace(-60,60,4)
    for y = linspace(-60,60,4)
        apPos = [apPos; x, y];
    end
end

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

% ZF
W_zf = H * inv(H'*H);
W = zeros(size(W_zf));
for k = 1:K
    W(:,k) = W_zf(:,k)/norm(W_zf(:,k));
end

% Power
gamma = 1; sigma2 = 0.5;
Pcomm = zeros(M*Nt, K);
for k = 1:K
    hk = H(:,k);
    wk = W(:,k);
    gain = abs(hk'*wk)^2;
    pk = gamma * sigma2 / gain;
    Pcomm(:,k) = sqrt(pk) * wk;
end

PcommTotal = sum(sum(abs(Pcomm).^2));
fprintf('PcommTotal = %.4f W\n', PcommTotal);

% Test rho=0.5
rho = 0.5;
PsensTotal = (1-rho)/rho * PcommTotal;
TotalPower = PcommTotal + PsensTotal;
fprintf('rho=%.2f: Psens=%.4f W, Total=%.4f W\n', rho, PsensTotal, TotalPower);

z = G/norm(G);
Psens = sqrt(PsensTotal)*z;

% SNR
snr = abs(G'*Psens)^2/sigma2;
snrWc = snr * 0.85/1.15;
fprintf('SNR = %.4f (linear) = %.2f dB, worst-case = %.2f dB\n', snr, 10*log10(snr), 10*log10(snrWc));

% SINR
for k = 1:K
    hk = H(:,k);
    desired = abs(hk'*Pcomm(:,k))^2;
    interf = sigma2;
    for j = 1:K
        if j~=k, interf = interf + abs(hk'*Pcomm(:,j))^2; end
    end
    sinr = desired/interf;
    sinrWc = sinr * 0.85/1.15;
    fprintf('User %d: desired=%.4f, interf=%.4f, SINR=%.2f dB, wc=%.2f dB\n', ...
        k, desired, interf, 10*log10(sinr), 10*log10(sinrWc));
end

% Check: is there interference?
HW = H'*Pcomm;
fprintf('\nInterference check (should be diagonal):\n');
for k = 1:K
    for j = 1:K
        if k~=j && abs(HW(k,j)) > 0.001
            fprintf('  Non-zero at (%d,%d): %.4f\n', k, j, abs(HW(k,j)));
        end
    end
end
