# Strictly monotone OPBGB calibration

For a base score `p in (0,1)`, define

    logit z(p) = lambda_1 log(p) - lambda_0 log(1-p),

where `lambda_0, lambda_1 > 0`. The symmetric model imposes
`lambda_0 = lambda_1 = lambda`.

The calibrated probability is

    logit C(p) = theta + a log z(p) - b log(1-z(p)),

with `a,b > 0`. Its derivative is

    C'(p) = C(p)(1-C(p)) [a(1-z(p)) + b z(p)]
            [lambda_1/p + lambda_0/(1-p)] > 0.

Thus the mapping is strictly increasing, preserves score ordering, and leaves
rank-based discrimination unchanged apart from numerical ties. Setting
`lambda_0=lambda_1=1` recovers beta calibration exactly. Quadratic penalties on
`log(lambda_0)` and `log(lambda_1)` shrink the model toward this nested beta
submodel.
