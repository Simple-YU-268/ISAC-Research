function traces = run_convergence_trace_figure(varargin)
%RUN_CONVERGENCE_TRACE_FIGURE  Diagnostic continuation traces for Figure 2.

ip = inputParser;
addParameter(ip,'Seeds',[1 50 100],@(x) isnumeric(x) && isvector(x));
addParameter(ip,'T_max',10,@(x) isnumeric(x) && isscalar(x) && x >= 2);
addParameter(ip,'Diagnostic_continuation',false,@(x) islogical(x) && isscalar(x));
addParameter(ip,'Output_dir',fullfile(pwd,'..','..','experiment_packages','v1.0', ...
    'results','convergence_traces'),@(x) ischar(x) || isstring(x));
parse(ip,varargin{:}); opt=ip.Results; out_dir=char(opt.Output_dir);
if ~exist(out_dir,'dir'), mkdir(out_dir); end
traces = cell(numel(opt.Seeds),1);
for i = 1:numel(opt.Seeds)
    prm = generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400, ...
        'N_req',3,'eps_h',0.05,'seed',opt.Seeds(i));
    prm.solver='mosek'; prm.mosek_max_time=10; prm.recovery_mosek_max_time=10;
    prm.recovery_max_candidates=1; prm.recovery_stop_first_feasible=true;
    prm.enable_topology_early_stop=~opt.Diagnostic_continuation;
    traces{i} = baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false);
    traces{i}.seed = opt.Seeds(i);
end
save(fullfile(out_dir,'convergence_traces.mat'),'traces','opt');
plot_traces(traces, fullfile(pwd,'..','..','experiment_packages','v1.0','figures'));
end

function plot_traces(traces, figure_dir)
if ~exist(figure_dir,'dir'), mkdir(figure_dir); end
colors = lines(numel(traces)); fig=figure('Visible','off','Position',[100 100 760 820]);
layout=tiledlayout(fig,4,1,'TileSpacing','compact','Padding','compact');
labels=cellfun(@(r) sprintf('Seed %d',r.seed),traces,'UniformOutput',false);
for panel=1:4
    ax=nexttile(layout); hold(ax,'on'); grid(ax,'on');
    for i=1:numel(traces)
        r=traces{i}; t=1:r.dc_iterations;
        switch panel
            case 1, y=r.power_trace; ylabel(ax,'Total transmit power (W)');
            case 2, y=r.rank_residual_trace; ylabel(ax,'Rank residual');
            case 3, y=r.binary_distance_trace; ylabel(ax,'Binary distance');
            case 4, y=double(r.topology_changed_trace); ylabel(ax,'Top-N support changed');
        end
        plot(ax,t,y,'-o','Color',colors(i,:),'LineWidth',1.5,'MarkerFaceColor',colors(i,:));
    end
    if panel==2, set(ax,'YScale','log'); end
    if panel==3, yline(ax,.05,'--','early-stop threshold','Color',[.75 0 0]); end
    if panel==4, ylim(ax,[-.1 1.1]); yticks(ax,[0 1]); end
    if panel==1, legend(ax,labels,'Location','best'); end
end
xlabel(ax,'DC iteration');
exportgraphics(fig,fullfile(figure_dir,'fig2_dc_sca_convergence.png'),'Resolution',300);
close(fig);
end
