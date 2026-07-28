function plot_double_dc_convergence(result, output_base)
%PLOT_DOUBLE_DC_CONVERGENCE  Plot feasibility and stabilization diagnostics.
%   The outer loop increases DC penalties, so total power need not decrease.
%   This figure therefore separates penalty continuation from residual decay.

idx = find(isfinite(result.power));
if isempty(idx)
    error('plot_double_dc_convergence:NoData', 'No valid iterations to plot.');
end
rank_tol = 1e-5;
binary_tol = 1e-5;
power = result.power(idx);
power_change = [NaN; abs(diff(power)) ./ max(abs(power(1:end-1)), 1e-12)];

figure('Name', 'DoubleDCConvergence', 'Visible', 'off', 'Position', [100 100 850 650]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(idx, power * 1e3, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5); grid on;
xlabel('SCA iteration'); ylabel('Total power [mW]');
title('Continuation trajectory');

nexttile;
semilogy(idx, max(result.binary_distance(idx), 1e-14), 'k-d', 'LineWidth', 1.5, 'MarkerSize', 5); hold on;
yline(binary_tol, 'k--', 'Tolerance'); grid on;
xlabel('SCA iteration'); ylabel('Max binary violation');
title('Binary feasibility');

nexttile;
semilogy(idx, max(result.rank_deficiency(idx), 1e-14), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 5); hold on;
yline(rank_tol, 'r--', 'Tolerance'); grid on;
xlabel('SCA iteration'); ylabel('Max rank deficiency');
title('Rank-one recovery');

nexttile;
semilogy(idx, max(power_change, 1e-14), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5); hold on;
yyaxis right;
stairs(idx, result.eta_b(idx), 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2); ylabel('Binary penalty');
yyaxis left; grid on;
xlabel('SCA iteration'); ylabel('Relative power change');
title('Stabilization after penalty continuation');

sgtitle('Double-DC SCA diagnostics');
saveas(gcf, [char(output_base), '.png']);
close(gcf);
end
