import numpy as np
from scipy.optimize._numdiff import approx_derivative
from sklearn.metrics import log_loss
from monotone_opbgb import MonotoneOPBGBCalibrator, SelectedMonotoneOPBGB, clip01


def synthetic(seed=42, n=3000):
    rng=np.random.default_rng(seed)
    p=clip01(rng.beta(1.3,5.0,n))
    true=clip01(1/(1+np.exp(-(-0.4+1.5*np.log(p)-0.8*np.log1p(-p)))))
    y=rng.binomial(1,true)
    return p,y


def test_analytic_gradient_matches_finite_difference():
    p,y=synthetic(n=700)
    m=MonotoneOPBGBCalibrator(two_sided=True,penalty=1e-3)
    x=np.array([-1.0,.1,-.2,.25,-.15])
    f,g=m._objective_gradient(x,p,y)
    gn=approx_derivative(lambda q: np.array([m._objective_gradient(q,p,y)[0]]),x).ravel()
    assert np.max(np.abs(g-gn)) < 2e-5


def test_strict_monotonicity_and_rank_preservation_symmetric():
    p,y=synthetic()
    m=MonotoneOPBGBCalibrator(two_sided=False).fit(p,y)
    grid=np.linspace(1e-6,1-1e-6,10000)
    pred=m.predict(grid)
    assert np.all(np.diff(pred)>0)
    assert np.min(m.derivative(grid))>0
    order=np.argsort(p)
    assert np.all(np.diff(m.predict(p[order]))>=0)


def test_strict_monotonicity_two_sided():
    p,y=synthetic(seed=9)
    m=MonotoneOPBGBCalibrator(two_sided=True).fit(p,y)
    grid=np.linspace(1e-6,1-1e-6,10000)
    assert np.all(np.diff(m.predict(grid))>0)
    assert np.min(m.derivative(grid))>0


def test_beta_submodel_at_unit_lambdas():
    p=np.linspace(.001,.999,1000)
    m=MonotoneOPBGBCalibrator(two_sided=True)
    m.theta_=-.2; m.a_=1.3; m.b_=.7; m.lambda1_=1.; m.lambda0_=1.
    expected=1/(1+np.exp(-(-.2+1.3*np.log(p)-.7*np.log1p(-p))))
    assert np.max(np.abs(m.predict(p)-expected))<1e-10


def test_selected_fit_is_finite_and_improves_miscalibrated_scores():
    p,y=synthetic(seed=11,n=5000)
    sf=slice(0,2500); ss=slice(2500,3800); st=slice(3800,None)
    m=SelectedMonotoneOPBGB(two_sided=True,penalties=(0,1e-3,1e-2)).fit(p[sf],y[sf],p[ss],y[ss])
    pred=m.predict(p[st])
    assert np.isfinite(pred).all()
    assert log_loss(y[st],pred)<log_loss(y[st],p[st])

def test_extreme_probability_grid_remains_ordered_without_inner_clipping():
    p=np.geomspace(1e-10,1e-2,4000)
    m=MonotoneOPBGBCalibrator(two_sided=True)
    m.theta_=-1.0; m.a_=2.0; m.b_=1.5; m.lambda1_=3.0; m.lambda0_=0.5
    pred=m.predict(p)
    assert np.all(np.diff(pred)>0)
