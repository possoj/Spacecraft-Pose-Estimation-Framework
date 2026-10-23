# Fixed-Setting Evaluation: Temporal Filtering on D-SPEED

## Scope and experimental protocol

This evaluation accompanies the temporal-filtering illustration in the research article. It provides the experimental protocol, mathematical definitions, fixed filter settings, detailed numerical results, and one composite figure for every video sequence. The purpose is to illustrate how D-SPEED supports the evaluation of temporal post-processing of image-based pose estimates.

**These are descriptive results on the videos used for parameter calibration.** Parameters were selected by visual inspection of a discrete sweep and then confirmed by the author. They remain fixed across sequences. These results are not a held-out or cross-validated estimate of performance on unseen trajectories. They do not establish universal superiority of a particular filter.

The experiment uses **Mobile-URSONet+**, checkpoint `mursop_fp32_dspeed`, trained on D-SPEED still images, without retraining. All methods receive the same cached network predictions. There are {{N_SEQUENCES}} video sequences, {{N_FAMILIES}} trajectory families and {{N_FRAMES}} frames, sampled at 25 Hz. Each complete sequence contains 1,500 frames; the plotted timestamps run from 0 to 59.96 s. Suffixes `_eb` and `_st` designate Earth-background and star-background variants, respectively. Variants of the same motion belong to the same family; CATR is a singleton family.

Inference was performed once, frame by frame, using the FP32 model in `pose_brevitas:latest`. The inference run used the CPU because the execution environment could not access a working NVIDIA driver. Filtering, scoring and figure generation reuse this cache and require no GPU. The cache records frame identifiers, ground-truth poses, raw predicted poses and both predicted probability mass functions (PMFs).

Every filter is causal. Its state is reset at the start of every sequence; all frames, including initialization, are included in the metrics. The two pose components are filtered independently. Ground truth is used for calibration, evaluation and plotting, never in a filter update. Predictions are not shifted in time before scoring.

## Filter definitions and confirmed settings

Let $i$ denote the zero-based frame index, $P_i$ a normalized network PMF for either attitude or position, and $\bar{P}_i$ the filtered PMF. As in the article, the two pose components are processed separately.

- **Raw:** the original decoded network pose, without temporal filtering.
- **Causal moving average:** $\bar{P}_i=L_i^{-1}\sum_{j=0}^{L_i-1}P_{i-j}$, where $L_i=\min(W,i+1)$. The initial window uses the available samples.
- **EMA:** $\bar{P}_0=P_0$ and $\bar{P}_i=\beta P_i+(1-\beta)\bar{P}_{i-1}$. Smaller $\beta$ increases smoothing.
- **Adaptive:** $\bar{P}_0=P_0$, $d_i=\lVert P_i-\bar{P}_{i-1}\rVert_2$, $w_i=\exp(-\alpha d_i)$, and

$$
\bar{P}_i=
\frac{n w_i P_i+(1-w_i)\bar{P}_{i-1}}
     {n w_i+1-w_i}.
$$

For the adaptive filter, smaller $n$ increases smoothing when $\alpha$ is fixed. Its effective gain is $g_i=nw_i/(nw_i+1-w_i)$, which varies with the discrepancy between the current and previous PMFs. It is therefore distinct from a fixed-gain EMA, including when $n=1$.

{{PARAMETERS}}

PMFs are normalized after each update. Position is decoded as the probability-weighted mean of its bin centers. Attitude is decoded as the unit eigenvector corresponding to the largest eigenvalue of $\sum_b\bar{P}_i(b)q_bq_b^{\mathsf{T}}$, where $b$ indexes histogram bins and $q_b$ is the associated unit quaternion. The frame index $i$ and bin index $b$ are distinct. This quaternion average is invariant to the signs of the bin quaternions. Existing filtering and quaternion utilities are reused.

The PMFs are classification outputs. Their widths depend in part on the soft-label hyperparameters used during training; they are **not treated as calibrated estimates of network uncertainty**.

### Parameter selection

The adaptive sensitivity parameters were kept fixed at their previously established values. The sweep increased smoothing by decreasing adaptive $n$ and EMA $\beta$ from 1.0 to 0.1 in steps of 0.1, and by increasing the moving-average window from 1 to 25 frames. Each candidate was compared with the preceding candidate using pose trajectories and framewise errors, with particular attention to rapid variations. Attitude and position were inspected separately.

For the adaptive filter, the confirmed settings precede a visible deterioration in tracking: for example, decreasing attitude $n$ from 0.8 to 0.7 causes a pronounced degradation on TITR_eb, while decreasing position $n$ from 0.5 to 0.4 degrades CATR. These observations support the previously used adaptive settings.

For EMA and moving average, degradation is gradual and there is no equally sharp loss-of-tracking boundary. The confirmed values represent a visual compromise between residual fluctuations and tracking response. The moving-average windows lie in shallow error plateaus; nearby windows should not be interpreted as meaningfully inferior or the selected windows as unique optima. No universal lag threshold or tracking-error threshold was imposed. This manual protocol is reported as such, rather than as an automatic optimization or a cross-validation procedure.

## Mathematical definitions of the metrics

A sequence contains $N$ frames indexed by $i=0,\ldots,N-1$, with sampling interval $\Delta\tau=0.04\,\mathrm{s}$. Following the article, $q_i=q_{BC,i}$ is the body-to-camera orientation and $t_i=t_{BC,i}$ is the target position in the camera frame; hats denote predictions. Here $t_i$ is a position vector, not a time index.

Quaternions are unit, scalar-first Hamilton quaternions. The symbol $\odot$ denotes the Hamilton product, $q^*$ denotes conjugation, and $q\cdot\hat{q}$ is the Euclidean inner product of quaternion coordinates. Define the unit-conversion factors $c_q=180/\pi$ and $c_t=1$.

### Framewise pose errors

As in the article, the shortest attitude error is first defined in radians:

$$
e_q^{(i)}=2\arccos\!\left(\operatorname{clip}
\left(\left|q_i\cdot\hat{q}_i\right|,0,1\right)\right).
$$

The position error is the Euclidean distance in meters:

$$
e_t^{(i)}=\lVert t_i-\hat{t}_i\rVert_2.
$$

The implementation uses the equivalent rotation-vector expression for numerical stability. The absolute inner product makes the attitude error invariant to replacing a quaternion by its negative.

The exported attitude errors are in degrees. To distinguish these reporting units from the article's radian definition, let $\varepsilon_i^q=c_q e_q^{(i)}$ and $\varepsilon_i^t=c_t e_t^{(i)}$. For either component $a\in\{q,t\}$, the reported statistics are

$$
\mu_a=\frac{1}{N}\sum_{i=0}^{N-1}\varepsilon_i^a,
\qquad
\sigma_a=\sqrt{\frac{1}{N}\sum_{i=0}^{N-1}
\left(\varepsilon_i^a-\mu_a\right)^2},
$$

$$
\operatorname{RMSE}_a=
\sqrt{\frac{1}{N}\sum_{i=0}^{N-1}\left(\varepsilon_i^a\right)^2}.
$$

**SD denotes $\sigma_a$, the population standard deviation of scalar framewise error magnitudes** (`ddof=0`). It is neither the standard deviation across sequences nor an uncertainty interval for the mean. Median and 95th percentile use the same error sequence and NumPy's default linear percentile interpolation. Their units, and those of mean, SD and RMSE, are degrees for attitude and meters for position.

### Residual jitter

Jitter measures changes in the **pose residual**, rather than changes in the predicted trajectory. Exactly tracked motion therefore has zero residual jitter.

The position residual and attitude error quaternion follow the article:

$$
r_i^t=\hat{t}_i-t_i,
\qquad
\delta q_i=q_i^*\odot\hat{q}_i.
$$

The attitude residual is the shortest-angle rotation vector

$$
r_i^q=\operatorname{rotvec}(\delta q_i)\in\mathbb{R}^3.
$$

For a unit residual quaternion $\delta q_i=[s_i,v_i]$ whose sign is chosen so that $s_i\geq0$, this conversion is

$$
\operatorname{rotvec}([s_i,v_i])=
\begin{cases}
2\operatorname{atan2}(\lVert v_i\rVert_2,s_i)
\dfrac{v_i}{\lVert v_i\rVert_2}, & \lVert v_i\rVert_2>0,\\
0, & \lVert v_i\rVert_2=0.
\end{cases}
$$

Its norm is the attitude error in radians, at most $\pi$. For either component $a\in\{q,t\}$, residual jitter is

$$
J_a=\frac{c_a}{\Delta\tau}
\sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}
\lVert r_i^a-r_{i-1}^a\rVert_2^2}.
$$

Thus $J_q$ is in deg/s and $J_t$ is in m/s. These are RMS residual variation rates, not error standard deviations. A constant position bias has zero residual jitter. Attitude residual vectors are expressed in the instantaneous ground-truth local frame, and their principal rotation-vector representation can be discontinuous near an attitude error of 180 degrees. The statistic describes the implemented residual fluctuations; it is not a calibrated sensor-noise estimate or an intrinsic covariant derivative on the rotation manifold. Small jitter alone does not guarantee accurate tracking: excessive smoothing can also reduce jitter.

### Relative motion error and diagnostic lag

For an interval of $k=25$ frames ($k\Delta\tau=1\,\mathrm{s}$), define the true and predicted relative rotations

$$
\Delta q_i^{(k)}=q_i^*\odot q_{i+k},
\qquad
\Delta\hat{q}_i^{(k)}=\hat{q}_i^*\odot\hat{q}_{i+k}.
$$

These temporal increments use $\Delta$ to distinguish them from the framewise error quaternion $\delta q_i$. Relative motion errors, in the exported units, are

$$
e_{q,\mathrm{rel}}^{(i,k)}=
2c_q\arccos\!\left(\operatorname{clip}
\left(\left|\Delta q_i^{(k)}\cdot\Delta\hat{q}_i^{(k)}\right|,0,1\right)\right),
$$

$$
e_{t,\mathrm{rel}}^{(i,k)}=
\lVert(\hat{t}_{i+k}-\hat{t}_i)-(t_{i+k}-t_i)\rVert_2.
$$

For either component $a\in\{q,t\}$, `rpe25` is

$$
\operatorname{RPE}_{a,k}=
\sqrt{\frac{1}{N-k}\sum_{i=0}^{N-k-1}
\left(e_{a,\mathrm{rel}}^{(i,k)}\right)^2},
\qquad k=25.
$$

The units are degrees for attitude and meters for position. The value is undefined when $N\leq k$.

`lag_frames` is a bounded diagnostic proxy. For $\ell\in\{0,\ldots,10\}$, the implementation compares ground truth at frames $i=0,\ldots,N-11$ with predictions at $i+\ell$ and returns the smallest $\ell$ minimizing mean squared pose error. Positive lag indicates a delayed prediction. Its search interval is 0-0.4 s; it cannot measure anticipation or distinguish delays beyond that bound. Lag is reported only when $N>35$ and the ground-truth motion from its initial pose exceeds 3 degrees or 0.3 m for the corresponding component; it is otherwise undefined. These diagnostic cutoffs do not enter parameter selection. No time compensation is applied to any other metric. Mean lag values average defined values only, first within families and then across eligible families, and need not be integers.

### Mapping to exported metric names

The code and CSV schemas retain `ori` and `pos` for attitude and position; the mathematical notation uses $q$ and $t$ as in the article.

| CSV field | Mathematical quantity | Units |
|---|---|---|
| `mean` | $\mu_a$ | deg or m |
| `median` | Median of $\varepsilon_i^a$ | deg or m |
| `p95` | 95th percentile of $\varepsilon_i^a$ | deg or m |
| `std` | $\sigma_a$ | deg or m |
| `rmse` | $\operatorname{RMSE}_a$ | deg or m |
| `jitter` | $J_a$ | deg/s or m/s |
| `rpe25` | $\operatorname{RPE}_{a,25}$ | deg or m |
| `lag_frames` | Diagnostic lag $\ell$ | frames |

## Aggregation and confidence intervals

Let $M_s$ be a scalar metric for sequence $s$, and let $\mathcal S_f$ be the sequences in trajectory family $f$. The primary aggregate gives equal weight to each family:

$$
M_f=\frac{1}{|\mathcal S_f|}\sum_{s\in\mathcal S_f}M_s,
\qquad
M_{\mathrm{family}}=\frac{1}{F}\sum_{f=1}^{F}M_f.
$$

Consequently, the paired backgrounds of a motion are not counted as two independent motion families. All reported aggregate medians, percentiles, SDs, RMSEs and jitter values are **averages of the corresponding sequence statistics**, first within families and then across families. They are not computed from a pooled frame distribution. In particular, an aggregate SD is the average within-sequence SD, not the dispersion across families. The secondary sequence-weighted table uses $M_{\mathrm{sequence}}=S^{-1}\sum_s M_s$ and provides a complementary view; the article's Table VI uses the family-weighted aggregate. The singleton CATR receives more weight in the family-weighted table.

[bootstrap_ci.csv](bootstrap_ci.csv) supplies descriptive 95% percentile bootstrap intervals for the family-weighted mean of each metric except lag. There are 2,000 resamples of the {{N_FAMILIES}} complete families, sampled with replacement using NumPy `default_rng(0)`. The same sampled family indices are used across methods and metrics. Parameters are held fixed during resampling. These intervals describe variation under family resampling conditional on the selected settings; they do not remove the optimism from using the same videos for calibration and reporting.

## Results

### Primary results: equal weight per trajectory family

SD and jitter follow the definitions above. Values are rounded only for presentation; CSV files retain numerical precision.

{{FAMILY_RESULTS}}

### Additional family-weighted statistics

Median, 95th percentile, RMSE and RPE25 use deg for attitude and m for position. Lag is in frames.

{{EXTRA_RESULTS}}

### Secondary results: equal weight per video

{{SEQUENCE_RESULTS}}

### Cases where smoothing increases mean pose error

The following sequences have a higher mean error than the raw predictions at the confirmed settings. This list uses a strict numerical comparison without a practical-significance threshold; small changes should be judged using the numerical tables and curves. Signed differences are available in [changes_vs_raw.csv](changes_vs_raw.csv).

{{REGRESSIONS}}

Lower jitter does not imply lower pose error in every scenario. These comparisons illustrate the trade-off between smoothing and response to motion. Each filter should be assessed using both errors and trajectories, rather than by a single aggregate number.

## Results by trajectory family

{{FAMILIES}}

## Per-sequence results and composite figures

Each figure contains eight synchronized panels. The left column shows yaw, pitch, roll, and geodesic attitude error. The right column shows x, y, z, and Euclidean position error. Green dashed curves are ground truth; grey curves are raw predictions; orange, blue and purple curves are moving average, EMA and adaptive filtering, respectively. The error panels contain the four prediction methods.

Euler angles use the existing `quat2euler` ZYX convention (yaw, pitch, roll). Ground-truth angles are unwrapped over time, and estimated angles are shifted by integer multiples of 360 degrees to the nearest ground-truth branch **for display only**. This does not change the predictions or any score. Euler coordinates remain ambiguous near gimbal lock and may show branch changes; attitude errors are always computed from quaternions. Each PNG and SVG is the same single composite figure, supplied in raster and vector formats.

{{SEQUENCES}}

## Files and reproducibility

- [selected_parameters.json](selected_parameters.json): the confirmed settings used for this document.
- [per_sequence.csv](per_sequence.csv): all metrics for every sequence, method and pose component.
- [per_family.csv](per_family.csv): the within-family averages.
- [summary.csv](summary.csv): family-weighted and sequence-weighted summaries.
- [bootstrap_ci.csv](bootstrap_ci.csv): family-resampling intervals with the settings held fixed.
- [changes_vs_raw.csv](changes_vs_raw.csv): signed changes in mean error, SD, jitter and RMSE.
- [protocol.json](protocol.json): frame counts, settings, cache identity and generation metadata.
- `figures/`: one composite figure per video, in PNG and SVG formats.
- `tracks/`: frame identifiers, ground truth and the four decoded pose trajectories per video in NumPy NPZ format; keys are `frame`, `true_ori`, `true_pos`, and `{raw,moving_average,ema,adaptive}_{ori,pos}`.
- `legacy_check.csv`, when historical thesis workbooks are available: agreement checks for raw predictions and matching adaptive settings (0.1 deg / 0.01 m tolerance).
- [SHA256SUMS](SHA256SUMS): checksums of the generated package files, excluding the checksum file itself.

The accompanying repository implements this export in `src/temporal/evaluation.py`, using the existing filters in `src/temporal/benchmark_filters.py`, `moving_average.py` and `pdf_compare.py`, the quaternion utilities in `src/spe/utils.py`, and the same scoring function as the benchmark.

From the repository root, after producing the inference cache:

```bash
python temporal_benchmark.py --stage evaluate \
  --model models/mursop_fp32_dspeed \
  --dataset ../datasets/dspeed/video \
  --cache-dir experiments/temporal_benchmark_ieee_taes/cache \
  --selected-parameters experiments/temporal_benchmark_ieee_taes/sweep/manual_selection.json \
  --output-dir experiments/temporal_benchmark_ieee_taes/evaluation
```

Use the existing `pose_brevitas:latest` container with the project and dataset mounted at the same relative locations. To reproduce a distributed evaluation package, `--selected-parameters` can instead point to its `selected_parameters.json`. Generation refuses to overwrite a nonempty output directory; select a new directory for another run. Neither network retraining nor inference is repeated by this command. Weights, video images and PMF caches are not included in the evaluation package.

Model parameters SHA256: `{{MODEL_HASH}}`.

Model configuration SHA256: `{{CONFIG_HASH}}`.

The cache manifest also records SHA256 hashes of the sequence label files. The behavioral checks in `tests/test_temporal_benchmark.py` cover causality, state reset, probability normalization, quaternion sign invariance, residual metrics and export consistency.
