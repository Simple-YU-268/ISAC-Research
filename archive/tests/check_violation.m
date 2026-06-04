% Check violation calculation
clear;

% Example: SNR=4.5dB, SINR=1.4dB, required SNR=3dB, SINR=0dB
snr_db = 4.5;
sinr_db = 1.4;
snr_req = 3;
sinr_req = 0;

% Convert to linear
snr_lin = 10^(snr_db/10);
sinr_lin = 10^(sinr_db/10);
snr_req_lin = 10^(snr_req/10);
sinr_req_lin = 10^(sinr_req/10);

fprintf('SNR: %.2f dB = %.4f linear (req %.2f dB = %.4f)\n', snr_db, snr_lin, snr_req, snr_req_lin);
fprintf('SINR: %.2f dB = %.4f linear (req %.2f dB = %.4f)\n', sinr_db, sinr_lin, sinr_req, sinr_req_lin);

% Violation (normalized)
v_snr = max(0, snr_req_lin - snr_lin) / snr_req_lin;
v_sinr = max(0, sinr_req_lin - sinr_lin) / sinr_req_lin;
viol = max(v_snr, v_sinr);

fprintf('\nViolation: SNR=%.4f, SINR=%.4f, max=%.4f\n', v_snr, v_sinr, viol);
fprintf('Should be 0 (both constraints satisfied)!\n');

% Ah! The issue: if SNR > req, v_snr = 0. Correct.
% Let me check with the actual values from Trial 1
fprintf('\n--- Trial 1 values ---\n');
snr_db = 4.5; sinr_db = 1.4;
snr_lin = 10^(snr_db/10);  % = 2.818
sinr_lin = 10^(sinr_db/10);  % = 1.380
snr_req_lin = 10^(3/10);  % = 1.995
sinr_req_lin = 10^(0/10);  % = 1.0

v_snr = max(0, snr_req_lin - snr_lin) / snr_req_lin;
v_sinr = max(0, sinr_req_lin - sinr_lin) / sinr_req_lin;
fprintf('v_snr = max(0, %.4f - %.4f) / %.4f = %.4f\n', snr_req_lin, snr_lin, snr_req_lin, v_snr);
fprintf('v_sinr = max(0, %.4f - %.4f) / %.4f = %.4f\n', sinr_req_lin, sinr_lin, sinr_req_lin, v_sinr);
