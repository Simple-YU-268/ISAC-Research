function plot_isac_tradeoff_topology_slices(result_file, output_dir)
%PLOT_ISAC_TRADEOFF_TOPOLOGY_SLICES  Spatial slices for the trade-off surface.
%   AP marker size denotes total transmitted power.  Its colour interpolates
%   from blue (communication-dominant) to orange (dedicated-sensing-dominant).
%   Lines denote only the AP--target sensing-cluster indicator b_mp.

if nargin < 1 || isempty(result_file)
    result_file = fullfile(pwd, '..', '..', 'experiment_packages', 'v1.0', ...
        'results', 'isac_tradeoff_surface', 'surface_final.mat');
end
if nargin < 2 || isempty(output_dir)
    output_dir = fileparts(result_file);
end
loaded = load(result_file, 'result'); result = loaded.result;

[~, a_sense] = min(result.alpha); [~, g_sense] = min(result.gamma_db);
[~, a_comm] = max(result.alpha); [~, g_comm] = max(result.gamma_db);
indices = [a_comm, g_comm; a_sense, g_sense];
titles = {'Communication-stringent, sensing-loose', ...
          'Sensing-stringent, communication-loose'};

maps = cell(2,1);
for q = 1:2
    maps{q} = result.topology{indices(q,1), indices(q,2)};
    if isempty(maps{q}), error('Selected surface point is not physically feasible.'); end
end
if ~isequal(round(maps{1}.b), round(maps{2}.b))
    warning('The selected operating points do not share the same sensing topology.');
end
total_power = cellfun(@(m) m.communication_power_W(:) + sum(m.sensing_power_W,2), maps, ...
    'UniformOutput', false);
power_scale = max(cellfun(@max, total_power));
bar_scale = max(cellfun(@(v) max(v), total_power));

fig = figure('Visible', 'off', 'Position', [100 100 1350 670]);
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

ax_topology = nexttile(layout, [2 1]);
draw_map(ax_topology, maps{1}, power_scale, false, true);
title(ax_topology, {'Common sensing-cluster topology', ...
    'Identical AP-target associations at both operating points'}, 'Interpreter', 'none');
legend(ax_topology, 'Location', 'southeast');

for q = 1:2
    ax_map = nexttile(layout, q+1);
    draw_map(ax_map, maps{q}, power_scale, true, true);
    title(ax_map, sprintf('%s\nalpha=%.2g, gamma_k=%.0f dB, P=%.1f mW', ...
        titles{q}, result.alpha(indices(q,1)), result.gamma_db(indices(q,2)), ...
        1e3*result.power_W(indices(q,1),indices(q,2))), 'Interpreter', 'none');

    ax_bar = nexttile(layout, q+4);
    comm = 1e3 * maps{q}.communication_power_W(:);
    sensing = 1e3 * sum(maps{q}.sensing_power_W,2);
    bars = bar(ax_bar, 1:numel(comm), [comm, sensing], 'stacked');
    bars(1).FaceColor = [0.10 0.35 0.85];
    bars(2).FaceColor = [0.90 0.35 0.08];
    grid(ax_bar, 'on'); xlim(ax_bar, [0.4, numel(comm)+0.6]);
    ylim(ax_bar, [0, 1e3*bar_scale*1.12]);
    xticks(ax_bar, 1:numel(comm)); xticklabels(ax_bar, compose('AP%d', 1:numel(comm)));
    ylabel(ax_bar, 'Per-AP power (mW)');
    title(ax_bar, 'Power composition');
    if q == 1
        legend(ax_bar, {'Communication', 'Dedicated sensing'}, 'Location', 'northwest');
    end
end
exportgraphics(fig, fullfile(output_dir, 'fig8b_tradeoff_topology_slices.png'), 'Resolution', 300);
close(fig);
end

function draw_map(ax, map, power_scale, show_power, show_labels)
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
comm = map.communication_power_W(:);
sensing = sum(map.sensing_power_W, 2);
total = comm + sensing;
if show_power
    sensing_ratio = sensing ./ max(total, eps);
    ap_colour = (1-sensing_ratio) .* repmat([0.10 0.35 0.85], numel(total), 1) + ...
        sensing_ratio .* repmat([0.90 0.35 0.08], numel(total), 1);
    marker_size = 80 + 1050 * total / max(power_scale, eps);
else
    ap_colour = repmat([0.55 0.55 0.55], numel(total), 1);
    marker_size = 125 * ones(size(total));
end
for p = 1:size(map.b,2)
    selected = find(map.b(:,p) > 0.5);
    for ii = selected.'
        plot(ax, [map.AP_pos(ii,1), map.Target_pos(p,1)], ...
            [map.AP_pos(ii,2), map.Target_pos(p,2)], '-', ...
            'Color', [0.85 0.25 0.10 0.42], 'LineWidth', 1.1, ...
            'HandleVisibility', 'off');
    end
end
scatter(ax, map.AP_pos(:,1), map.AP_pos(:,2), marker_size, ap_colour, 'filled', ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'AP');
scatter(ax, map.UE_pos(:,1), map.UE_pos(:,2), 100, [0.05 0.60 0.25], 'd', ...
    'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'UE');
scatter(ax, map.Target_pos(:,1), map.Target_pos(:,2), 170, [0.80 0.05 0.12], 'p', ...
    'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Target');
if show_labels
    for m = 1:size(map.AP_pos,1)
        text(ax, map.AP_pos(m,1)+6, map.AP_pos(m,2)-9, sprintf('AP%d',m), ...
            'FontSize', 8, 'FontWeight', 'bold');
    end
end
all_pos = [map.AP_pos; map.UE_pos; map.Target_pos];
xlim(ax, [min(all_pos(:,1))-40, max(all_pos(:,1))+40]);
ylim(ax, [min(all_pos(:,2))-25, max(all_pos(:,2))+75]);
xlabel(ax, 'x (m)'); ylabel(ax, 'y (m)');
end
