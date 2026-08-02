function audit = audit_paper_results(results_root)
%AUDIT_PAPER_RESULTS  Raw-data integrity and figure-statistic audit.

if nargin < 1
    results_root=fullfile(pwd,'..','..','experiment_packages','v1.0','results');
end
audit = struct(); audit.pass = true; audit.findings = strings(0,1);

% Figure 3 source: common 50-seed cluster-size records.
paths={fullfile(results_root,'nreq_sweep','nreq2','pilot_final.mat'), ...
    fullfile(results_root,'main_config_mc_100seeds','pilot_final.mat'), ...
    fullfile(results_root,'nreq_sweep','nreq4','pilot_final.mat'), ...
    fullfile(results_root,'nreq_sweep','nreq5','pilot_final.mat'), ...
    fullfile(results_root,'nreq_sweep','nreq6','pilot_final.mat')};
counts=zeros(1,5); feasible=zeros(1,5); med_power=NaN(1,5);
for q=1:5
    data=load(paths{q},'records'); r=data.records; if q==2, r=r(1:50); end
    counts(q)=numel(r); f=[r.proposed_feasible]; feasible(q)=nnz(f);
    p=[r.proposed_power_W]; med_power(q)=median(p(f),'omitnan');
end
audit.fig3.counts=counts; audit.fig3.feasible=feasible; audit.fig3.median_power_W=med_power;
assert(all(counts==50),'Figure 3 must use 50 common records per Nreq.');
assert(all(feasible==50),'Figure 3 has unexpected infeasible proposed records.');
assert(all(isfinite(med_power) & med_power>0),'Figure 3 contains invalid power values.');

% Figure 4 raw QoS records.
qos=load(fullfile(results_root,'nreq_qos_sweep','nreq_qos_final.mat'),'output'); r=qos.output.records;
assert(isequal(size(r),[5 50]),'Figure 4 QoS sweep must be 5 by 50 records.');
qos_pcrb=[]; qos_comm=[]; qos_sense=[];
for q=1:5
    assert(all([r(q,:).feasible]),'QoS sweep has an infeasible record.');
    m=[r(q,:).metrics]; qos_pcrb=[qos_pcrb, arrayfun(@(x)max(x.pcrb_ratio),m)]; %#ok<AGROW>
    qos_comm=[qos_comm, arrayfun(@(x)min(x.nominal_sinr_margin_dB),m)]; %#ok<AGROW>
    qos_sense=[qos_sense, arrayfun(@(x)min(x.sensing_sinr_margin_dB),m)]; %#ok<AGROW>
end
audit.fig4.max_pcrb_ratio=max(qos_pcrb); audit.fig4.min_comm_margin_dB=min(qos_comm);
audit.fig4.min_sensing_margin_dB=min(qos_sense);
% Recomputing trace(inv(J)) from a floating-point SDP solution can differ
% slightly from its Schur auxiliary certificate.  Use a documented
% numerical-audit tolerance while retaining the raw maximum in the report.
audit.fig4.pcrb_numeric_tolerance=5e-5;
assert(audit.fig4.max_pcrb_ratio<=1+audit.fig4.pcrb_numeric_tolerance, ...
    'PCRB ratio exceeds numerical-audit tolerance.');
assert(audit.fig4.min_comm_margin_dB>=-1e-5,'Nominal communication margin is negative.');
assert(audit.fig4.min_sensing_margin_dB>=-1e-5,'Sensing margin is negative.');

% Figure 5 source.
main=load(fullfile(results_root,'main_config_mc_100seeds','pilot_final.mat'),'records'); r=main.records;
g=[r.power_penalty_pct]; audit.fig5.n=numel(g); audit.fig5.nan_count=nnz(~isfinite(g));
assert(audit.fig5.n==100 && audit.fig5.nan_count==0,'Figure 5 CDF has missing observations.');

% Figure 6 source and summary agreement.
ab=load(fullfile(results_root,'recovery_ablation_30seeds','ablation_final.mat'),'records','summary');
rates=arrayfun(@(m)mean(arrayfun(@(x)x.methods(m).feasible,ab.records)),1:3);
audit.fig6.raw_feasibility_rate=rates; audit.fig6.summary_feasibility_rate=ab.summary.feasibility_rate;
assert(max(abs(rates-ab.summary.feasibility_rate))<1e-12,'Figure 6 summary disagrees with raw records.');

% Figure 7 source sizes.
dim_paths={fullfile(results_root,'dimension_sweep','m4','pilot_final.mat'), ...
    fullfile(results_root,'main_config_mc_100seeds','pilot_final.mat'), ...
    fullfile(results_root,'dimension_sweep','m8','pilot_final.mat')};
dim_counts=zeros(1,3);
for q=1:3
    d=load(dim_paths{q},'records'); rr=d.records; if q==2, rr=rr(1:30); end
    dim_counts(q)=numel(rr);
end
audit.fig7.counts=dim_counts; assert(all(dim_counts==30),'Figure 7 requires 30 records per dimension.');

% Figures 8 and 9 source completeness/domain checks.
trade=load(fullfile(results_root,'isac_tradeoff_surface_mc','tradeoff_mc_final.mat'),'result');
audit.fig8.feasibility_min=min(trade.result.feasibility_rate,[],'all');
assert(isequal(size(trade.result.power_W),[5 5 5]),'Figure 8 has wrong raw-data shape.');
assert(all(isfinite(trade.result.mean_power_W),'all'),'Figure 8 has missing mean powers.');
rob=load(fullfile(results_root,'csi_robustness','csi_robustness_final.mat'),'output');
rr=rob.output.records; assert(isequal(size(rr),[4 30]),'Figure 9 has wrong raw-data shape.');
for q=1:4
    for i=1:30
        assert(rr(q,i).robust.feasible && rr(q,i).nominal.feasible,'Figure 9 has infeasible design.');
        assert(rr(q,i).robust.outage_probability>=0 && rr(q,i).robust.outage_probability<=1, ...
            'Robust outage outside [0,1].');
        assert(rr(q,i).nominal.outage_probability>=0 && rr(q,i).nominal.outage_probability<=1, ...
            'Nominal outage outside [0,1].');
    end
end
audit.fig9.robust_outage_median=arrayfun(@(q)median(arrayfun(@(x)x.robust.outage_probability,rr(q,:))),1:4);
audit.fig9.nominal_outage_median=arrayfun(@(q)median(arrayfun(@(x)x.nominal.outage_probability,rr(q,:))),1:4);

fprintf('RAW-DATA AUDIT PASSED\\n');
fprintf('Fig3: 50 records each; Fig4: max PCRB ratio %.8f, min margins %.4g/%.4g dB\\n', ...
    audit.fig4.max_pcrb_ratio,audit.fig4.min_comm_margin_dB,audit.fig4.min_sensing_margin_dB);
fprintf('Fig5: %d CDF observations; Fig6 rates match; Fig7: 30 records each\\n',audit.fig5.n);
fprintf('Fig8 min feasibility %.1f%%; Fig9 robust/nominal outage medians at eps=.08: %.3f/%.3f\\n', ...
    100*audit.fig8.feasibility_min,audit.fig9.robust_outage_median(end),audit.fig9.nominal_outage_median(end));
end
