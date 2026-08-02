function audit = audit_current_model_method_comparison(results_file)
%AUDIT_CURRENT_MODEL_METHOD_COMPARISON Validate final 30-seed method results.
%   This audit targets the current dedicated-sensing participation model only.
%   It deliberately does not use legacy 50/100-seed artifacts.

if nargin < 1
    results_file = fullfile(pwd, '..', '..', 'experiment_packages', 'v1.0', ...
        'results', 'nreq_method_performance_30seeds', ...
        'nreq_method_performance_final.mat');
end

raw = load(results_file, 'output');
assert(isfield(raw, 'output'), 'Final artifact must contain the output struct.');
records = raw.output.records;
methods = string(raw.output.labels);
nreq_values = raw.output.nreq_list;
seeds = raw.output.seeds;

assert(isequal(size(records), [numel(nreq_values), numel(seeds)]), ...
    'Record dimensions do not match Nreq values and common seeds.');
assert(numel(methods) == 4, 'Expected proposed, FIM, nearest, and random methods.');
assert(numel(seeds) == 30, 'The final method comparison must use 30 seeds.');
assert(isequal(nreq_values(:).', 2:6), 'Unexpected Nreq sweep.');

audit = struct();
audit.methods = methods;
audit.nreq_values = nreq_values;
audit.seeds = seeds;
audit.feasible_count = zeros(numel(nreq_values), numel(methods));
audit.mean_power_W = NaN(numel(nreq_values), numel(methods));
audit.max_pcrb_ratio = NaN(numel(nreq_values), numel(methods));

for q = 1:numel(nreq_values)
    for i = 1:numel(seeds)
        rec = records(q, i);
        assert(rec.seed == seeds(i), 'Common-seed ordering mismatch.');
        assert(numel(rec.methods) == numel(methods), 'Method count mismatch.');
        for j = 1:numel(methods)
            out = rec.methods(j);
            if out.feasible
                assert(isfinite(out.power_W) && out.power_W > 0, ...
                    'A feasible method has invalid power.');
                met = out.metrics;
                assert(all(isfinite(met.pcrb_ratio)), 'Feasible method has invalid PCRB ratio.');
                % Re-evaluating trace(inv(J)) from a floating-point SDP
                % covariance may differ slightly from the Schur certificate.
                % This is an audit tolerance, not a relaxation of the model.
                assert(max(met.pcrb_ratio) <= 1 + 1e-4, ...
                    'PCRB constraint fails numerical audit.');
                assert(met.num_nonzero_sensing_pairs == 2 * nreq_values(q), ...
                    'Positive sensing-floor participation invariant fails.');
            end
        end
    end
    for j = 1:numel(methods)
        out = arrayfun(@(x) x.methods(j), records(q, :));
        f = [out.feasible];
        audit.feasible_count(q, j) = nnz(f);
        if any(f)
            audit.mean_power_W(q, j) = mean([out(f).power_W]);
            audit.max_pcrb_ratio(q, j) = max(arrayfun(@(x) max(x.metrics.pcrb_ratio), out(f)));
        end
    end
end

assert(all(audit.feasible_count(:,1) == numel(seeds)), ...
    'Proposed method is not feasible on every final common-seed trial.');
fprintf('CURRENT-MODEL METHOD AUDIT PASSED: %d records, %d methods.\n', ...
    numel(records), numel(methods));
end
