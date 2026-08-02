function output = run_large_scale_nreq_sharded(varargin)
%RUN_LARGE_SCALE_NREQ_SHARDED Parallel, restartable large-scale Nreq study.
%   Each seed is run in an isolated checkpoint directory so concurrent MATLAB
%   workers never write the same file.  After all seed shards complete, their
%   rows are strictly validated and merged into the standard
%   nreq_method_performance_final.mat artifact.

ip = inputParser;
addParameter(ip,'Seeds',1:50,@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'N_req_list',2:6,@(x)isnumeric(x)&&isvector(x));
addParameter(ip,'M',9,@(x)isnumeric(x)&&isscalar(x)&&x>=2);
addParameter(ip,'Nt',2,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'K',4,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'P',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'N_theta',2,@(x)isnumeric(x)&&isscalar(x)&&ismember(x,[1 2]));
addParameter(ip,'AreaSize',400,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'Pmax_dBm',20,@(x)isnumeric(x)&&isscalar(x));
addParameter(ip,'eps_h',.05,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
addParameter(ip,'Gamma_alpha',3,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'T_max',3,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(ip,'Mosek_max_time',15,@(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(ip,'N_workers',0,@(x)isnumeric(x)&&isscalar(x)&&x>=0);
addParameter(ip,'Output_dir',fullfile(pwd,'experiment_packages','v1.0', ...
    'results','large_scale_algorithm_validation','M9_K4_P3'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Resume',true,@(x)islogical(x)&&isscalar(x));
parse(ip,varargin{:}); opt=ip.Results;

seeds=unique(opt.Seeds(:).','stable'); nreq=opt.N_req_list(:).';
assert(all(nreq>=1 & nreq<=opt.M),'N_req must be between one and M.');
out_dir=char(opt.Output_dir); if ~exist(out_dir,'dir'), mkdir(out_dir); end
shard_root=fullfile(out_dir,'seed_shards'); if ~exist(shard_root,'dir'), mkdir(shard_root); end
write_progress(out_dir,sprintf('START M%d_Nt%d_K%d_P%d: %d seeds, Nreq=%s, workers=%d', ...
    opt.M,opt.Nt,opt.K,opt.P,numel(seeds),mat2str(nreq),opt.N_workers));

if opt.N_workers>0
    pool=gcp('nocreate');
    if isempty(pool), pool=parpool('local',opt.N_workers); end
    q=parallel.pool.DataQueue;
    afterEach(q,@(msg)write_progress(out_dir,char(msg)));
    parfor s=1:numel(seeds)
        run_seed_shard(seeds(s),nreq,opt,shard_root);
        send(q,sprintf('DONE seed=%d',seeds(s)));
    end
else
    for s=1:numel(seeds)
        run_seed_shard(seeds(s),nreq,opt,shard_root);
        write_progress(out_dir,sprintf('DONE seed=%d',seeds(s)));
    end
end

output=merge_shards(seeds,nreq,opt,shard_root);
save(fullfile(out_dir,'nreq_method_performance_final.mat'),'output','opt','-v7.3');
write_progress(out_dir,'COMPLETE');
end

function run_seed_shard(seed,nreq,opt,shard_root)
dir_seed=fullfile(shard_root,sprintf('seed_%05d',seed));
file=fullfile(dir_seed,'nreq_method_performance_final.mat');
if opt.Resume && is_complete_seed_file(file,seed,nreq), return; end
run_nreq_method_performance_mc('Seeds',seed,'N_req_list',nreq, ...
    'M',opt.M,'Nt',opt.Nt,'K',opt.K,'P',opt.P,'N_theta',opt.N_theta, ...
    'AreaSize',opt.AreaSize,'Pmax_dBm',opt.Pmax_dBm,'eps_h',opt.eps_h, ...
    'Gamma_alpha',opt.Gamma_alpha,'T_max',opt.T_max, ...
    'Mosek_max_time',opt.Mosek_max_time,'Output_dir',dir_seed,'Resume',opt.Resume);
end

function yes=is_complete_seed_file(file,seed,nreq)
yes=false;
if ~exist(file,'file'), return; end
try
    raw=load(file,'output'); o=raw.output;
    yes=isequal(o.seeds,seed) && isequal(o.nreq_list,nreq) && ...
        size(o.records,1)==numel(nreq) && size(o.records,2)==1 && ...
        all([o.records.seed]==seed) && all([o.records.N_req]==nreq);
catch
    yes=false;
end
end

function output=merge_shards(seeds,nreq,opt,shard_root)
labels=[]; records=[];
for s=1:numel(seeds)
    file=fullfile(shard_root,sprintf('seed_%05d',seeds(s)), ...
        'nreq_method_performance_final.mat');
    assert(exist(file,'file')==2,'Missing seed shard %d.',seeds(s));
    raw=load(file,'output'); one=raw.output;
    assert(isequal(one.seeds,seeds(s)) && isequal(one.nreq_list,nreq), ...
        'Shard metadata mismatch for seed %d.',seeds(s));
    if isempty(labels), labels=one.labels; records=repmat(one.records(1,1),numel(nreq),numel(seeds));
    else, assert(isequal(one.labels,labels),'Method labels differ across seed shards.'); end
    records(:,s)=one.records;
end
for q=1:numel(nreq)
    assert(isequal([records(q,:).seed],seeds),'Merged seed ordering mismatch.');
    assert(all([records(q,:).N_req]==nreq(q)),'Merged Nreq metadata mismatch.');
    methods=[records(q,:).methods];
    assert(all(string({methods.label})==repmat(labels,1,numel(seeds))), ...
        'Merged method ordering mismatch.');
end
output=struct('records',records,'seeds',seeds,'nreq_list',nreq, ...
    'labels',labels,'configuration',struct('M',opt.M,'Nt',opt.Nt, ...
    'K',opt.K,'P',opt.P,'N_theta',opt.N_theta,'AreaSize',opt.AreaSize, ...
    'Pmax_dBm',opt.Pmax_dBm,'eps_h',opt.eps_h,'Gamma_alpha',opt.Gamma_alpha));
end

function write_progress(out_dir,message)
line=sprintf('[%s] %s\n',datestr(now,'yyyy-mm-dd HH:MM:SS'),message);
fprintf('%s',line); fid=fopen(fullfile(out_dir,'progress.log'),'a');
if fid>=0, fprintf(fid,'%s',line); fclose(fid); end
end
