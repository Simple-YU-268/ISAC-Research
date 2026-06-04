% Debug violation calculation
clear; rng(42);

M = 16; Nt = 4; K = 10;
Pmax = 30; snrReq = 3; sinrReq = 0; epsilon = 0;

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
gammaSinr = 10^(sinrReq/10);
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
rho = 0.1;
PsensTotal = (1-rho)/rho * PcommTotal;
Total = PcommTotal + PsensTotal;

z = G/norm(G);
Psens = sqrt(PsensTotal)*z;

snr = abs(G'*Psens)^2/sigma2;

minSinr = inf;
for k = 1:K
    hk = H(:,k);
    desired = abs(hk'*Pcomm(:,k))^2;
    interf = sigma2;
    for j = 1:K
        if j~=k, interf = interf + abs(hk'*Pcomm(:,j))^2; end
    end
    sinr = desired/interf;
    minSinr = min(minSinr, sinr);
    fprintf('User %d: desired=%.10f, interf=%.10f, sinr=%.10f, sinr_dB=%.10f\n', ...
        k, desired, interf, sinr, 10*log10(sinr));
end

fprintf('\nminSinr = %.10f\n', minSinr);
fprintf('10*log10(minSinr) = %.10f\n', 10*log10(minSinr));

sinrReqLin = 10^(sinrReq/10);
fprintf('sinrReqLin = %.10f\n', sinrReqLin);

vSinr = max(0, sinrReqLin - minSinr) / sinrReqLin;
fprintf('vSinr = max(0, %.10f - %.10f) / %.10f = %.10f\n', sinrReqLin, minSinr, sinrReqLin, vSinr);

fprintf('\nSNR = %.10f, SNR_dB = %.10f\n', snr, 10*log10(snr));
snrReqLin = 10^(snrReq/10);
vSnr = max(0, snrReqLin - snr) / snrReqLin;
fprintf('vSnr = %.10f\n', vSnr);
