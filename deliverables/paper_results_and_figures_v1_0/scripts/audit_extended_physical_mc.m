function audit = audit_extended_physical_mc(results_file)
%AUDIT_EXTENDED_PHYSICAL_MC Verify final extended campaign invariants.
%   Physical feasibility is certified during solving with a 1e-5 absolute
%   residual tolerance.  Gamma_track is scenario-scaled, so the separately
%   recomputed, descriptive PCRB ratio can exceed one by a small relative
%   amount when Gamma_track is small.  The tolerance below is only a reporting
%   tolerance for that ratio; it does not replace validate_solution.

if nargin<1
    results_file=fullfile(pwd,'experiment_packages','v1.0','results', ...
        'extended_physical_mc','extended_physical_mc_final.mat');
end
raw=load(results_file,'campaign'); c=raw.campaign;
assert(~isempty(c.configurations) && numel(c.configurations)<=22, ...
    'Unexpected number of physical configurations.');
assert(~isempty(c.seeds) && all(diff(c.seeds)==1), ...
    'Seeds must form an ordered common-seed range.');
assert(isequal(c.methods,["Proposed","FIM-greedy","Nearest-AP"]), ...
    'Unexpected method set.');

audit=struct('total_trials',0,'feasible_by_method',zeros(1,3), ...
    'error_trials',0,'max_pcrb_ratio',NaN,'participation_violations',0, ...
    'pcrb_ratio_reporting_tolerance',5e-4);
for q=1:numel(c.configurations)
    cfg=c.configurations(q); records=c.records{q};
    assert(numel(records)==numel(c.seeds),'Configuration has missing seed records.');
    assert(isequal([records.seed],c.seeds),'Seed order mismatch.');
    for i=1:numel(records)
        audit.total_trials=audit.total_trials+1;
        if strlength(records(i).error_id)>0, audit.error_trials=audit.error_trials+1; end
        for j=1:3
            out=records(i).methods(j);
            if out.feasible
                audit.feasible_by_method(j)=audit.feasible_by_method(j)+1;
                assert(isfinite(out.power_W)&&out.power_W>0,'Invalid feasible power.');
                assert(out.nonzero_pairs==cfg.P*cfg.N_req, ...
                    'Dedicated sensing participation invariant fails.');
                assert(out.mean_pcrb_ratio<=1+audit.pcrb_ratio_reporting_tolerance, ...
                    'PCRB numerical audit tolerance fails.');
                audit.max_pcrb_ratio=max(audit.max_pcrb_ratio,out.mean_pcrb_ratio);
            end
        end
    end
end
fprintf('EXTENDED CAMPAIGN AUDIT PASSED: %d scenarios, feasible [P/F/N]=%s, errors=%d.\n', ...
    audit.total_trials,mat2str(audit.feasible_by_method),audit.error_trials);
end
