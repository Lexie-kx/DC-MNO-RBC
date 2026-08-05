# M10-1a Path-B RMSCap Final Verdict

## Experiment category

Lightweight Path-B residual stabilization control.

M10-1a StateParam-RMSCap is promoted to pre-candidate status.
It is not the final formal model.

## Frozen design

- Frozen audited M6 backbone
- Path A: horizontal buoyancy anomaly -> u_y
- Path B: -u·grad(b) -> buoyancy
- Bounded dynamic gates
- Sample-wise Path-B RMS cap
- Cap source: split-specific train-only ground-truth support
- State conditioner receives raw Path-B
- RMS cap applies only to Path-B residual injection
- H4 free-autoregressive training
- 20 epochs
- Seeds: 42, 123, 2026
- Test horizons: 1, 4, 8, 16

## Locked caps

- unseen-Pr: 2.180076360160806
- unseen-Ra: 1.5787383958464702

## Three-seed StateParam-RMSCap results

### unseen-Pr global Rel-L2 (%)

- h1:  7.002441 ± 0.009875
- h4: 13.504509 ± 0.024652
- h8: 19.440531 ± 0.097916
- h16: 32.273084 ± 0.285686

M6 h16: 32.224166
Difference at h16: +0.048918 percentage points.

### unseen-Ra global Rel-L2 (%)

- h1:  13.821817 ± 0.001065
- h4:  25.975340 ± 0.267263
- h8:  34.839147 ± 0.531580
- h16: 49.045831 ± 0.874719

M6 h16: 44.866413
Difference at h16: +4.179418 percentage points.

## Final conclusions

1. Path-B train-support RMS safeguarding suppresses the catastrophic
   long-horizon degradation of Parameter-only coupling in held-out Ra
   across all three random seeds.

2. StateParam-RMSCap produces finite and concentrated held-out-Ra
   long-horizon errors across all three seeds.

3. StateParam-RMSCap outperforms Param-RMSCap at h16 in all three seeds.

4. On held-out Pr, StateParam-RMSCap preserves the original StateParam
   behavior and remains approximately tied with M6 at h16.

5. On held-out Ra, StateParam-RMSCap remains worse than M6 at medium and
   long horizons.

6. M10-1a validates a stabilization mechanism but does not establish
   long-horizon predictive superiority over M6.

7. No claim of improved physical consistency is made at this stage.

## Status

- Stabilization mechanism: PASS
- Three-seed reliability: PASS
- Predictive superiority over M6: NOT PASS
- Current model status: PRE-CANDIDATE
- M10-1a status: FROZEN
