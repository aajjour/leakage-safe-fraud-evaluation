#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd

def exact_mean_dist(x):
    x=np.asarray(x,float); n=len(x)
    idx=np.array(list(itertools.product(range(n),repeat=n)),dtype=np.int16)
    return x[idx].mean(axis=1)

def interval(x): return np.quantile(exact_mean_dist(x),[.025,.975])

def summaries(dataset,df,group_cols,metrics,unit='temporal fold'):
    rows=[]
    for keys,g in df.groupby(group_cols):
        if not isinstance(keys,tuple): keys=(keys,)
        kd=dict(zip(group_cols,keys))
        for metric in metrics:
            x=g[metric].to_numpy(float); lo,hi=interval(x)
            row={**kd,'metric':metric,'n_units':len(x),'unit':unit,'mean':x.mean(),'sd':x.std(ddof=1) if len(x)>1 else np.nan,'median':np.median(x),'q1':np.quantile(x,.25),'q3':np.quantile(x,.75),'min':x.min(),'max':x.max(),'resampling_interval_low':lo,'resampling_interval_high':hi,'resamples_enumerated':len(x)**len(x),'interval_interpretation':'finite empirical fold-resampling sensitivity interval; not population-level coverage'}
            if dataset is not None: row={'dataset':dataset,**row}
            rows.append(row)
    return rows

def signflip_p(d):
    d=np.asarray(d,float); obs=abs(d.mean())
    vals=[abs((d*np.asarray(s)).mean()) for s in itertools.product([-1.,1.],repeat=len(d))]
    return float(np.mean(np.asarray(vals)>=obs-1e-15))

def paired_row(comparison,metric,d,notes):
    d=np.asarray(d,float); lo,hi=interval(d); sd=d.std(ddof=1)
    neg='negative favors' in notes.lower()
    return {'comparison':comparison,'metric':metric,'n_units':len(d),'mean_difference':d.mean(),'median_difference':np.median(d),'resampling_interval_low':lo,'resampling_interval_high':hi,'favorable_units':int(np.sum(d<0 if neg else d>0)),'total_units':len(d),'exact_signflip_p':signflip_p(d),'standardized_paired_effect_dz':d.mean()/sd if sd>0 else np.nan,'resamples_enumerated':len(d)**len(d),'notes':notes}

def main():
    here=Path(__file__).resolve()
    default_root=here.parents[2]
    ap=argparse.ArgumentParser()
    ap.add_argument('--supplement-root',type=Path,default=default_root)
    ap.add_argument('--output-dir',type=Path,default=None)
    a=ap.parse_args(); base=a.supplement_root.resolve(); out=(a.output_dir or here.parent).resolve(); out.mkdir(parents=True,exist_ok=True)
    sp=pd.read_csv(base/'results/sparkov/decisive_metrics.csv')
    sp=sp[(sp.policy=='capped_weighted')&(sp.delay_days==14)&(sp.calibration=='beta')].groupby(['fold','model'],as_index=False).mean(numeric_only=True)
    ie=pd.read_csv(base/'results/ieee_temporal/metrics.csv'); ie=ie[ie.calibration=='platt'].copy()
    metrics=['pr_auc','roc_auc','brier','logloss','precision_0.005','recall_0.005','value_capture_0.005','value_capture_0.01']
    rows=summaries('Sparkov',sp,['model'],metrics)+summaries('IEEE-CIS temporal',ie,['model'],metrics)
    pd.DataFrame(rows).to_csv(out/'primary_fold_uncertainty.csv',index=False)
    mp=[]
    for ds,df in [('Sparkov',sp),('IEEE-CIS',ie)]:
        for model,g in df.groupby('model'):
            x=g.pr_auc.to_numpy(float); lo,hi=interval(x)
            mp.append({'dataset':ds,'model':model,'mean':x.mean(),'ci_low':lo,'ci_high':hi,'sd':x.std(ddof=1),'range_low':x.min(),'range_high':x.max(),'n_folds':len(x),'interval_type':'finite fold-resampling sensitivity interval','resamples_enumerated':len(x)**len(x)})
    pd.DataFrame(mp).to_csv(out/'manuscript_pr_auc_uncertainty.csv',index=False)
    cal=pd.read_csv(base/'code/opbgb_monotone_v2/results_full/calibration_metrics.csv')
    pd.DataFrame(summaries(None,cal,['model','calibration'],['brier','logloss','ece','ici','cal_intercept','cal_slope','oe_ratio'])).to_csv(out/'calibration_fold_uncertainty.csv',index=False)
    paired=[]; sp_p=sp.pivot(index='fold',columns='model',values='pr_auc'); ie_p=ie.pivot(index='fold',columns='model',values='pr_auc')
    paired += [paired_row('LightGBM - Logistic','PR-AUC',(sp_p.lightgbm-sp_p.logistic).values,'Sparkov primary setting; repeated seeds averaged within fold.'),paired_row('XGBoost - Logistic','PR-AUC',(sp_p.xgboost-sp_p.logistic).values,'Sparkov primary setting; repeated seeds averaged within fold.'),paired_row('XGBoost - LightGBM','PR-AUC',(sp_p.xgboost-sp_p.lightgbm).values,'Sparkov primary setting; repeated seeds averaged within fold.'),paired_row('XGBoost - LightGBM','PR-AUC',(ie_p.xgboost-ie_p.lightgbm).values,'IEEE-CIS temporal folds.'),paired_row('LightGBM - Logistic','PR-AUC',(ie_p.lightgbm-ie_p.logistic).values,'IEEE-CIS temporal folds.'),paired_row('XGBoost - Logistic','PR-AUC',(ie_p.xgboost-ie_p.logistic).values,'IEEE-CIS temporal folds.')]
    def cd(model,a,b,metric):
        p=cal[(cal.model==model)&(cal.calibration.isin([a,b]))].pivot(index='fold',columns='calibration',values=metric); return (p[a]-p[b]).values
    for metric in ['brier','logloss']:
        paired += [paired_row('Logistic BGB - Beta',metric,cd('logistic','bgb','beta',metric),'IEEE-CIS logistic calibration; negative favors BGB.'),paired_row('Logistic two-sided monotone OPBGB - Beta',metric,cd('logistic','opbgb_two_sided_monotone','beta',metric),'IEEE-CIS logistic calibration; negative favors monotone OPBGB.'),paired_row('Logistic two-sided monotone OPBGB - BGB',metric,cd('logistic','opbgb_two_sided_monotone','bgb',metric),'IEEE-CIS logistic calibration; negative favors monotone OPBGB.'),paired_row('LightGBM symmetric monotone OPBGB - Beta',metric,cd('lightgbm','opbgb_symmetric_monotone','beta',metric),'IEEE-CIS LightGBM calibration; negative favors monotone OPBGB.'),paired_row('XGBoost two-sided monotone OPBGB - Beta',metric,cd('xgboost','opbgb_two_sided_monotone','beta',metric),'IEEE-CIS XGBoost calibration; negative favors monotone OPBGB.')]
    pd.DataFrame(paired).to_csv(out/'paired_fold_effects.csv',index=False)
    ra=pd.read_csv(base/'results/ieee_random/random_metrics.csv'); ra=ra[ra.calibration=='platt'].copy(); rt=[]
    for model in ['logistic','lightgbm','xgboost']:
        for metric in ['pr_auc','brier','logloss','value_capture_0.01']:
            r=ra[ra.model==model][metric].to_numpy(float); t=ie[ie.model==model][metric].to_numpy(float); rd=exact_mean_dist(r); td=exact_mean_dist(t); diff=(rd[:,None]-td[None,:]).ravel(); rel=(100*(rd[:,None]-td[None,:])/td[None,:]).ravel(); lo,hi=np.quantile(diff,[.025,.975]); rlo,rhi=np.quantile(rel,[.025,.975])
            rt.append({'comparison':f'Random - temporal ({model})','metric':metric,'n_random':len(r),'n_temporal':len(t),'mean_difference':r.mean()-t.mean(),'resampling_interval_low':lo,'resampling_interval_high':hi,'relative_difference_percent':100*(r.mean()-t.mean())/t.mean(),'relative_interval_low':rlo,'relative_interval_high':rhi,'resample_combinations_enumerated':len(rd)*len(td),'dependence_status':'random replicates overlap in transactions and are not independent experimental units','inferential_status':'descriptive sensitivity analysis only; no independent-sample p-value or Hedges g'})
    pd.DataFrame(rt).to_csv(out/'random_temporal_uncertainty.csv',index=False)
    print(f'Wrote five exact fold-resampling outputs to {out}')
if __name__=='__main__': main()
