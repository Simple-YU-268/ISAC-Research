function result = run_isac_tradeoff_surface_mc(varargin)
%RUN_ISAC_TRADEOFF_SURFACE_MC  Statistical QoS trade-off surface.

ip=inputParser;
addParameter(ip,'Seeds',1:5,@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'Gamma_alpha_list',[1.5 2.5 3.5 5 6],@(x)isnumeric(x)&&isvector(x)&&all(x>0));
addParameter(ip,'Gamma_k_dB_list',[-3 0 3 6 9],@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'T_max',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'Mosek_max_time',10,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'Output_dir',fullfile(pwd,'..','..','experiment_packages','v1.0', ...
    'results','isac_tradeoff_surface_mc'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Resume',true,@(x)islogical(x)&&isscalar(x));
parse(ip,varargin{:}); opt=ip.Results;
seeds=opt.Seeds(:).'; alpha=opt.Gamma_alpha_list(:).'; gamma_db=opt.Gamma_k_dB_list(:).';
out_dir=char(opt.Output_dir); if ~exist(out_dir,'dir'), mkdir(out_dir); end
checkpoint_file=fullfile(out_dir,'checkpoint.mat');
power_W=NaN(numel(seeds),numel(alpha),numel(gamma_db));
runtime_s=NaN(size(power_W)); feasible=false(size(power_W));
if opt.Resume && exist(checkpoint_file,'file')
    saved=load(checkpoint_file,'power_W','runtime_s','feasible','seeds_saved','alpha_saved','gamma_saved');
    if isequal(saved.seeds_saved,seeds)&&isequal(saved.alpha_saved,alpha)&&isequal(saved.gamma_saved,gamma_db)
        power_W=saved.power_W; runtime_s=saved.runtime_s; feasible=saved.feasible;
    end
end
for s=1:numel(seeds)
    for a=1:numel(alpha)
        for g=1:numel(gamma_db)
            if ~isnan(runtime_s(s,a,g)), continue; end
            fprintf('Tradeoff seed=%d alpha=%.2g gamma=%.1f (%d/%d)\n',seeds(s),alpha(a), ...
                gamma_db(g),(s-1)*numel(alpha)*numel(gamma_db)+(a-1)*numel(gamma_db)+g, ...
                numel(seeds)*numel(alpha)*numel(gamma_db));
            prm=generate_scenario(6,2,3,2,2,20,'auto','AreaSize',400,'N_req',3, ...
                'eps_h',.05,'seed',seeds(s),'Gamma_alpha',alpha(a),'gamma_k_dB',gamma_db(g));
            prm.solver='mosek'; prm.mosek_max_time=opt.Mosek_max_time;
            prm.recovery_mosek_max_time=opt.Mosek_max_time;
            prm.recovery_max_candidates=3; prm.recovery_stop_first_feasible=false;
            timer=tic; res=baseline_alg2(prm,opt.T_max,1e-5,1,1,1,false);
            runtime_s(s,a,g)=toc(timer);
            feasible(s,a,g)=isfield(res,'is_physical_feasible')&&res.is_physical_feasible;
            if feasible(s,a,g), power_W(s,a,g)=res.final_obj; end
            seeds_saved=seeds; alpha_saved=alpha; gamma_saved=gamma_db;
            save(checkpoint_file,'power_W','runtime_s','feasible','seeds_saved','alpha_saved','gamma_saved','opt');
        end
    end
end
result.seeds=seeds; result.alpha=alpha; result.gamma_db=gamma_db;
result.power_W=power_W; result.runtime_s=runtime_s; result.feasible=feasible;
result.mean_power_W=squeeze(mean(power_W,1,'omitnan'));
result.feasibility_rate=squeeze(mean(feasible,1));
save(fullfile(out_dir,'tradeoff_mc_final.mat'),'result','opt');
% Keep figures inside the same experiment package irrespective of cwd.
plot_tradeoff_mc(result,fullfile(fileparts(fileparts(out_dir)),'figures'));
end

function plot_tradeoff_mc(result,figure_dir)
if ~exist(figure_dir,'dir'), mkdir(figure_dir); end
[G,A]=meshgrid(result.gamma_db,result.alpha);
fig=figure('Visible','off','Position',[100 100 1050 450]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile; surf(ax,G,A,1e3*result.mean_power_W,'EdgeColor',[.25 .25 .25]);
xlabel(ax,'Robust communication SINR target (dB)'); ylabel(ax,'PCRB allowance scale, \alpha');
zlabel(ax,'Mean transmit power (mW)'); title(ax,'Mean ISAC trade-off surface'); colorbar(ax); view(ax,45,30);
ax=nexttile; imagesc(ax,result.gamma_db,result.alpha,100*result.feasibility_rate);
set(ax,'YDir','normal'); xlabel(ax,'Robust communication SINR target (dB)');
ylabel(ax,'PCRB allowance scale, \alpha'); title(ax,'Physical feasibility rate (%)');
colorbar(ax); clim(ax,[0 100]);
exportgraphics(fig,fullfile(figure_dir,'fig8_statistical_tradeoff.png'),'Resolution',300);
close(fig);
end
