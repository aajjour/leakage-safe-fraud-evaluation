#!/usr/bin/env python3
"""Strictly monotone symmetric and two-sided OPBGB post-hoc calibrators.

The calibrator is
    z(p) = expit(lambda_1 log(p) - lambda_0 log(1-p))
    C(p) = expit(theta + a log z(p) - b log(1-z(p)))
with a,b,lambda_0,lambda_1 > 0.  The symmetric model sets
lambda_0=lambda_1=lambda.  Beta calibration is recovered at lambda_0=lambda_1=1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import log_loss

EPS = 1e-10
LOG_BOUNDS = (-2.0, 2.0)


def clip01(x):
    return np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, x)


def _transform_logs(p, lam1, lam0):
    """Return stable inner-transform quantities without clipping z itself."""
    p = clip01(p)
    q = lam1 * np.log(p) - lam0 * np.log1p(-p)
    log_z = -_softplus(-q)
    log1m_z = -_softplus(q)
    z = expit(q)
    return z, q, log_z, log1m_z


@dataclass
class FitDiagnostics:
    success: bool
    status: int
    message: str
    n_iter: int
    objective: float
    gradient_norm: float
    boundary_hit: bool


class MonotoneOPBGBCalibrator:
    """Penalized strictly monotone OPBGB calibrator.

    Parameters
    ----------
    two_sided:
        If False, lambda_0=lambda_1. If True, both are estimated.
    penalty:
        Quadratic penalty on log power parameters, shrinking toward beta
        calibration (lambda_0=lambda_1=1).
    outer_penalty:
        Small quadratic penalty on log a and log b for numerical stability.
    max_iter:
        Maximum L-BFGS-B iterations.
    """
    def __init__(self, two_sided=False, penalty=1e-3, outer_penalty=1e-6,
                 max_iter=1200, tol=1e-9):
        self.two_sided = bool(two_sided)
        self.penalty = float(penalty)
        self.outer_penalty = float(outer_penalty)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    @property
    def name(self):
        return "monotone_opbgb_two_sided" if self.two_sided else "monotone_opbgb_symmetric"

    def _unpack(self, x):
        theta = float(x[0]); a = float(np.exp(x[1])); b = float(np.exp(x[2]))
        if self.two_sided:
            lam1 = float(np.exp(x[3])); lam0 = float(np.exp(x[4]))
        else:
            lam1 = lam0 = float(np.exp(x[3]))
        return theta, a, b, lam1, lam0

    def _objective_gradient(self, x, p, y):
        theta, a, b, lam1, lam0 = self._unpack(x)
        p = clip01(p); y = np.asarray(y, dtype=float)
        lp = np.log(p); l1p = np.log1p(-p)
        z, q, lz, l1z = _transform_logs(p, lam1, lam0)
        eta = theta + a * lz - b * l1z
        mu = expit(eta)
        n = max(len(y), 1)
        nll = np.mean(np.logaddexp(0.0, eta) - y * eta)
        pen = self.outer_penalty * (x[1] ** 2 + x[2] ** 2)
        if self.two_sided:
            pen += self.penalty * (x[3] ** 2 + x[4] ** 2)
        else:
            pen += self.penalty * x[3] ** 2
        r = (mu - y) / n
        g = np.zeros_like(x, dtype=float)
        g[0] = np.sum(r)
        g[1] = np.sum(r * a * lz) + 2.0 * self.outer_penalty * x[1]
        g[2] = np.sum(r * (-b * l1z)) + 2.0 * self.outer_penalty * x[2]
        deta_dq = a * (1.0 - z) + b * z
        if self.two_sided:
            g[3] = np.sum(r * deta_dq * lam1 * lp) + 2.0 * self.penalty * x[3]
            g[4] = np.sum(r * deta_dq * (-lam0 * l1p)) + 2.0 * self.penalty * x[4]
        else:
            g[3] = np.sum(r * deta_dq * lam1 * (lp - l1p)) + 2.0 * self.penalty * x[3]
        return float(nll + pen), g

    def fit(self, p, y, x0=None):
        p = clip01(p); y = np.asarray(y, dtype=int)
        if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or len(p) == 0:
            raise ValueError("p and y must be nonempty one-dimensional arrays of equal length")
        if np.unique(y).size != 2:
            raise ValueError("Both outcome classes are required")
        dim = 5 if self.two_sided else 4
        if x0 is None:
            prevalence = np.clip(y.mean(), 1e-5, 1 - 1e-5)
            x0 = np.zeros(dim, dtype=float)
            x0[0] = np.log(prevalence / (1.0 - prevalence))
        bounds = [(None, None), LOG_BOUNDS, LOG_BOUNDS] + [LOG_BOUNDS] * (dim - 3)
        result = minimize(
            fun=lambda x: self._objective_gradient(x, p, y), x0=np.asarray(x0, float),
            method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": 1e-7, "maxls": 50},
        )
        self.raw_params_ = result.x.copy()
        self.theta_, self.a_, self.b_, self.lambda1_, self.lambda0_ = self._unpack(result.x)
        self.nll_ = float(np.mean(-y * np.log(self.predict(p)) - (1-y) * np.log1p(-self.predict(p))))
        logpars = result.x[1:]
        boundary = bool(np.any(np.isclose(logpars, LOG_BOUNDS[0], atol=1e-4)) or
                        np.any(np.isclose(logpars, LOG_BOUNDS[1], atol=1e-4)))
        self.diagnostics_ = FitDiagnostics(
            success=bool(result.success), status=int(result.status), message=str(result.message),
            n_iter=int(result.nit), objective=float(result.fun),
            gradient_norm=float(np.linalg.norm(result.jac)), boundary_hit=boundary,
        )
        if not result.success:
            raise RuntimeError(f"{self.name} optimization failed: {result.message}")
        return self

    def decision_function(self, p):
        p = clip01(p)
        _, _, log_z, log1m_z = _transform_logs(p, self.lambda1_, self.lambda0_)
        return self.theta_ + self.a_ * log_z - self.b_ * log1m_z

    def predict(self, p):
        # Evaluate fully in log space.  Do not clip the inner transform, because
        # early clipping can create artificial probability plateaus.
        eta = self.decision_function(p)
        return np.clip(expit(eta), np.finfo(float).tiny, 1.0 - np.finfo(float).eps)

    def derivative(self, p):
        p = clip01(p)
        z, _, _, _ = _transform_logs(p, self.lambda1_, self.lambda0_)
        c = self.predict(p)
        dqdp = self.lambda1_ / p + self.lambda0_ / (1.0 - p)
        detadq = self.a_ * (1.0 - z) + self.b_ * z
        return c * (1.0 - c) * detadq * dqdp

    def parameters(self):
        d = self.diagnostics_
        return {
            "method": self.name, "theta": self.theta_, "a": self.a_, "b": self.b_,
            "lambda1": self.lambda1_, "lambda0": self.lambda0_,
            "penalty": self.penalty, "outer_penalty": self.outer_penalty,
            "success": d.success, "status": d.status, "message": d.message,
            "n_iter": d.n_iter, "objective": d.objective,
            "gradient_norm": d.gradient_norm, "boundary_hit": d.boundary_hit,
        }


class SelectedMonotoneOPBGB:
    """Select penalty on a held-out chronological calibration-selection window.

    Candidate penalties are fitted on calibration-fit data and selected by log loss
    on calibration-selection data.  The selected penalty is then refitted on the
    combined calibration-fit and calibration-selection blocks.  The future test
    block is never used during fitting or selection.
    """
    def __init__(self, two_sided=False, penalties=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
                 outer_penalty=1e-6, max_iter=1200):
        self.two_sided = bool(two_sided)
        self.penalties = tuple(float(x) for x in penalties)
        self.outer_penalty = float(outer_penalty)
        self.max_iter = int(max_iter)

    def fit(self, p_fit, y_fit, p_select, y_select):
        records = []
        for penalty in self.penalties:
            try:
                model = MonotoneOPBGBCalibrator(
                    two_sided=self.two_sided, penalty=penalty,
                    outer_penalty=self.outer_penalty, max_iter=self.max_iter,
                ).fit(p_fit, y_fit)
                pred = model.predict(p_select)
                records.append((log_loss(y_select, pred, labels=[0, 1]), penalty, model))
            except Exception:
                records.append((np.inf, penalty, None))
        best = min(records, key=lambda x: (x[0], x[1]))
        if not np.isfinite(best[0]):
            raise RuntimeError("All monotone OPBGB penalty candidates failed")
        self.selection_logloss_ = float(best[0]); self.selected_penalty_ = float(best[1])
        self.selection_records_ = [
            {"penalty": float(pen), "selection_logloss": float(loss), "success": mdl is not None}
            for loss, pen, mdl in records
        ]
        p_all = np.concatenate([clip01(p_fit), clip01(p_select)])
        y_all = np.concatenate([np.asarray(y_fit, int), np.asarray(y_select, int)])
        self.model_ = MonotoneOPBGBCalibrator(
            two_sided=self.two_sided, penalty=self.selected_penalty_,
            outer_penalty=self.outer_penalty, max_iter=self.max_iter,
        ).fit(p_all, y_all)
        return self

    def predict(self, p): return self.model_.predict(p)
    def derivative(self, p): return self.model_.derivative(p)
    def parameters(self):
        out = self.model_.parameters()
        out.update({"selected_penalty": self.selected_penalty_,
                    "selection_logloss": self.selection_logloss_})
        return out
