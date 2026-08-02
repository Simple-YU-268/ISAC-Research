function output = merge_nreq_method_performance_shards(varargin)
%MERGE_NREQ_METHOD_PERFORMANCE_SHARDS  Assemble disjoint common-seed rows.
%   This merger deliberately selects only complete Nreq rows from separately
%   checkpointed shards.  It rejects missing/duplicate seeds and mismatched
%   method labels before writing the final 30-seed artifact.

ip = inputParser;
addParameter(ip,'Pilot_file',fullfile(pwd,'experiment_packages','v1.0','results', ...
    'nreq_method_performance_pilot10','nreq_method_performance_final.mat'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Rows23_file',fullfile(pwd,'experiment_packages','v1.0','results', ...
    'nreq_method_performance_seeds11to30','checkpoint.mat'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Row4_file',fullfile(pwd,'experiment_packages','v1.0','results', ...
    'nreq_method_performance_seeds11to30_nreq4','nreq_method_performance_final.mat'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Rows56_file',fullfile(pwd,'experiment_packages','v1.0','results', ...
    'nreq_method_performance_seeds11to30_nreq5to6','nreq_method_performance_final.mat'),@(x)ischar(x)||isstring(x));
addParameter(ip,'Output_dir',fullfile(pwd,'experiment_packages','v1.0','results', ...
    'nreq_method_performance_30seeds'),@(x)ischar(x)||isstring(x));
parse(ip,varargin{:}); opt=ip.Results;

[R1,n1,s1,labels]=load_shard(char(opt.Pilot_file));
[R23,n23,s23,labels23]=load_shard(char(opt.Rows23_file));
[R4,n4,s4,labels4]=load_shard(char(opt.Row4_file));
[R56,n56,s56,labels56]=load_shard(char(opt.Rows56_file));
assert(isequal(labels,labels23) && isequal(labels,labels4) && isequal(labels,labels56), ...
    'Method labels differ across shards.');
assert(isequal(s1,1:10) && isequal(s23,11:30) && isequal(s4,11:30) && isequal(s56,11:30), ...
    'Unexpected seed ranges in method-comparison shards.');
assert(isequal(n1,2:6) && isequal(n23,2:6) && isequal(n4,4) && isequal(n56,[5 6]), ...
    'Unexpected Nreq rows in method-comparison shards.');

records = repmat(R1(1,1),5,30);
for q=1:5
    records(q,1:10)=R1(q,:);
end
% Only rows 2 and 3 of the original 11:30 checkpoint are complete/useful.
records(1,11:30)=R23(1,:);
records(2,11:30)=R23(2,:);
records(3,11:30)=R4(1,:);
records(4,11:30)=R56(1,:);
records(5,11:30)=R56(2,:);

for q=1:5
    assert(all(~isnan([records(q,:).seed])),'Merged row has missing records.');
    assert(isequal([records(q,:).seed],1:30),'Merged row seed order is not 1:30.');
    assert(all([records(q,:).N_req]==q+1),'Merged row has incorrect Nreq metadata.');
    methods=[records(q,:).methods];
    assert(numel(methods)==120 && all(string({methods.label})==repmat(labels,1,30)), ...
        'Merged method ordering or labels are inconsistent.');
end
output.records=records; output.seeds=1:30; output.nreq_list=2:6; output.labels=labels;
out_dir=char(opt.Output_dir); if ~exist(out_dir,'dir'), mkdir(out_dir); end
save(fullfile(out_dir,'nreq_method_performance_final.mat'),'output','opt','-v7.3');
end

function [records,nreq,seeds,labels]=load_shard(file)
L=load(file);
if isfield(L,'output')
    records=L.output.records; nreq=L.output.nreq_list; seeds=L.output.seeds; labels=L.output.labels;
else
    records=L.records; nreq=L.nreq_saved; seeds=L.seeds_saved; labels=L.labels_saved;
end
end
