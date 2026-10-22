# Area-adjusted accuracy assessment

Map: landcover_map; reference: sample_points (field REF_CLASS).

Stratified estimators of Olofsson et al. (2013, 2014). 1431 reference samples, 5 classes, total mapped area 69,120 ha. Intervals are 95% confidence intervals (estimate +/- 1.96 SE).

## Summary

| Measure | Value |
|---|---:|
| Overall accuracy (area-weighted) | 0.825 +/- 0.019 |
| Overall accuracy (unweighted sample, as in Compute Confusion Matrix) | 0.829 |
| Kappa (sample counts, for comparison only) | 0.763 |

## Error matrix of sample counts

| Map \ Reference | Water | Forest | Grass/shrub | Cropland | Developed | Total | W_i |
|---|---:|---:|---:|---:|---:|---:|---:|
| Water | 100 | 0 | 0 | 0 | 0 | 100 | 0.0457 |
| Forest | 0 | 383 | 34 | 0 | 0 | 417 | 0.3050 |
| Grass/shrub | 0 | 52 | 454 | 12 | 12 | 530 | 0.3878 |
| Cropland | 0 | 14 | 100 | 165 | 5 | 284 | 0.2074 |
| Developed | 0 | 0 | 12 | 3 | 85 | 100 | 0.0542 |
| Total | 100 | 449 | 600 | 180 | 102 | 1431 | 1.0000 |

## Error matrix of estimated area proportions

| Map \ Reference | Water | Forest | Grass/shrub | Cropland | Developed | Total |
|---|---:|---:|---:|---:|---:|---:|
| Water | 0.0457 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0457 |
| Forest | 0.0000 | 0.2801 | 0.0249 | 0.0000 | 0.0000 | 0.3050 |
| Grass/shrub | 0.0000 | 0.0381 | 0.3322 | 0.0088 | 0.0088 | 0.3878 |
| Cropland | 0.0000 | 0.0102 | 0.0730 | 0.1205 | 0.0037 | 0.2074 |
| Developed | 0.0000 | 0.0000 | 0.0065 | 0.0016 | 0.0461 | 0.0542 |
| Total | 0.0457 | 0.3284 | 0.4366 | 0.1309 | 0.0585 | 1.0000 |

## Accuracy by class

| Class | n (map stratum) | User's accuracy +/- 95% CI | Producer's accuracy +/- 95% CI |
|---|---:|---:|---:|
| Water | 100 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Forest | 417 | 0.918 +/- 0.026 | 0.853 +/- 0.029 |
| Grass/shrub | 530 | 0.857 +/- 0.030 | 0.761 +/- 0.026 |
| Cropland | 284 | 0.581 +/- 0.057 | 0.920 +/- 0.038 |
| Developed | 100 | 0.850 +/- 0.070 | 0.787 +/- 0.080 |

## Area estimates

| Class | Mapped area (ha) | Estimated area (ha) | 95% CI (ha) | Estimate - mapped |
|---|---:|---:|---:|---:|
| Water | 3,157 | 3,157 | +/- 0.00 | 0.00 |
| Forest | 21,078 | 22,696 | +/- 949 | 1,618 |
| Grass/shrub | 26,807 | 30,178 | +/- 1,281 | 3,371 |
| Cropland | 14,332 | 9,046 | +/- 900 | -5,286 |
| Developed | 3,745 | 4,043 | +/- 483 | 297 |

## Warnings

- Map class Water (1): all 100 samples agree with the reference. Read this as 'no errors observed', not 'no uncertainty'. The Wald interval for its user's accuracy (and its share of the area intervals) has zero width; the Wilson 95% interval is 0.963 to 1.000.

## Method

- Rows of the error matrix are map classes (strata); p_ij = W_i n_ij / n_i.
- User's accuracy U_i = p_ii / p_i.; producer's accuracy P_j = p_jj / p_.j; overall accuracy O = sum p_jj.
- Area of class k = total area x p_.k, with SE(p_.k) = sqrt(sum_i (W_i p_ik - p_ik^2) / (n_i. - 1)).
- Variances follow Olofsson et al. (2013, 2014); producer's accuracy uses the variance of the stratified ratio estimator.
- User's accuracy intervals: Wald (U +/- z SE, as in Olofsson et al. 2014) by default. The Wilson score interval (Wilson 1927) is available because the Wald interval has zero width when U is 0 or 1 and tends to under-cover when U is near 0 or 1 with small n.
- A user's accuracy of exactly 0 or 1 gives a zero-width interval; the true uncertainty is not zero, so treat those intervals with caution.

## References

- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22, 209-212.
- Olofsson, P., Foody, G. M., Stehman, S. V., & Woodcock, C. E. (2013). Making better use of accuracy data in land change studies: Estimating accuracy and area and quantifying uncertainty using stratified estimation. Remote Sensing of Environment, 129, 122-131.
- Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014). Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment, 148, 42-57.
