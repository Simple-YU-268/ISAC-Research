# Final evidence summary

The final common-seed method comparison contains 150 scenarios
(`Nreq=2:6`, 30 seeds each) and four physical association/recovery methods.
At `Nreq=3`, mean transmit power is 30.83 mW for the proposed method,
30.93 mW for FIM-greedy association, and 39.29 mW for nearest-AP association.
Random association is feasible in only 10 of 30 trials and has a conditional
mean power of 115.82 mW.

The proposed method is feasible in all 30 trials for every tested `Nreq`.
Its binary-DC mechanism reduces the median binary residual from `4.268e-1`
without the binary penalty to `6.229e-5`.  The robust CSI study reports zero
sampled outage for the robust design at uncertainty radii 0.02, 0.05, and 0.08,
whereas the nominal design has mean outages of 89.0%, 89.6%, and 91.9%.

Read `README.md` for the model and reproduction commands.  All conditional
metrics in the figures are explicitly conditioned on physical feasibility.
