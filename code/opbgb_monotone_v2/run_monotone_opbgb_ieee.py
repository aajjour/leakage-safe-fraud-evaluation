#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.utils.multiclass import type_of_target
from joblib import Parallel, delayed
from monotone_opbgb import SelectedMonotoneOPBGB

warnings.filterwarnings('ignore')
EPS=1e-8
SEED=20260717

def clip01(x): return np.clip(np.asarray(x,dtype=float),EPS,1-EPS)
def safe_logit(x): return logit(clip01(x))

def ece(y,p,bins=15):
    y=np.asarray(y,int); p=clip01(p); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        m=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
        if m.any(): out += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)

def ici(y,p,grid_n=200):
    # Lightweight smooth calibration error using isotonic regression as a stable estimate.
    y=np.asarray(y,int); p=clip01(p)
    iso=IsotonicRegression(out_of_bounds='clip').fit(p,y)
    return float(np.mean(np.abs(iso.predict(p)-p)))

def calibration_intercept_slope(y,p):
    z=safe_logit(p).reshape(-1,1)
    lr=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000)
    lr.fit(z,y)
    return float(lr.intercept_[0]), float(lr.coef_[0,0])

def oe_ratio(y,p): return float(np.sum(y)/max(np.sum(clip01(p)),EPS))

def top_budget(y,p,amount,budget):
    n=len(y); k=max(1,int(np.ceil(budget*n))); idx=np.argsort(-p)[:k]
    tp=int(np.sum(y[idx])); total=int(np.sum(y)); fraud_amt=float(np.sum(amount[y==1])); cap=float(np.sum(amount[idx][y[idx]==1]))
    return dict(recall=tp/max(total,1),precision=tp/k,value_capture=cap/max(fraud_amt,EPS))

class PlattCal:
    def fit(self,s,y):
        self.m=LogisticRegression(C=1e6,max_iter=2000).fit(safe_logit(s).reshape(-1,1),y); self.success_=True; return self
    def predict(self,s): return self.m.predict_proba(safe_logit(s).reshape(-1,1))[:,1]
    def params(self): return {'intercept':float(self.m.intercept_[0]),'coef':float(self.m.coef_[0,0]),'success':True}

class BetaCal:
    def _x(self,s): s=clip01(s); return np.c_[np.log(s),np.log1p(-s)]
    def fit(self,s,y): self.m=LogisticRegression(C=1e6,max_iter=2000).fit(self._x(s),y); self.success_=True; return self
    def predict(self,s): return self.m.predict_proba(self._x(s))[:,1]
    def params(self): return {'intercept':float(self.m.intercept_[0]),'a':float(self.m.coef_[0,0]),'b_raw':float(self.m.coef_[0,1]),'success':True}

class IsoCal:
    def fit(self,s,y): self.m=IsotonicRegression(out_of_bounds='clip').fit(s,y); self.success_=True; return self
    def predict(self,s): return clip01(self.m.predict(s))
    def params(self): return {'n_thresholds':int(len(self.m.X_thresholds_)),'success':True}

def odds_power_transform(s,ll=1.0,rr=None):
    s=clip01(s); rr=ll if rr is None else rr
    la=ll*np.log(s); lb=rr*np.log1p(-s); m=np.maximum(la,lb)
    a=np.exp(la-m); b=np.exp(lb-m)
    return clip01(a/(a+b))

class OPTransformLogistic:
    def __init__(self,grid,two_sided=False,n_jobs=1): self.grid=list(grid); self.two_sided=two_sided; self.n_jobs=n_jobs
    def _features(self,s,ll,rr):
        z=odds_power_transform(s,ll,rr)
        return np.c_[np.log(z),np.log1p(-z),safe_logit(z)]
    def _fit_one(self,ll,rr,sf,yf,ss,ys):
        m=LogisticRegression(C=1e6,max_iter=2000)
        try:
            m.fit(self._features(sf,ll,rr),yf)
            ps=m.predict_proba(self._features(ss,ll,rr))[:,1]
            return (log_loss(ys,clip01(ps),labels=[0,1]),-average_precision_score(ys,ps),ll,rr,m,True)
        except Exception:
            return (np.inf,np.inf,ll,rr,None,False)
    def fit(self,s_fit,y_fit,s_select,y_select):
        if self.two_sided:
            cand=[(ll,rr) for ll in self.grid for rr in self.grid if abs(np.log(ll)-np.log(rr))<=np.log(10)]
        else: cand=[(x,x) for x in self.grid]
        res=Parallel(n_jobs=self.n_jobs,prefer='threads')(delayed(self._fit_one)(ll,rr,s_fit,y_fit,s_select,y_select) for ll,rr in cand)
        best=min(res,key=lambda x:(x[0],x[1]))
        self.selection_logloss,self.neg_ap,self.lam_left,self.lam_right,self.m,self.success_=best
        if not self.success_: raise RuntimeError('All OPBGB candidates failed')
        return self
    def predict(self,s): return self.m.predict_proba(self._features(s,self.lam_left,self.lam_right))[:,1]
    def params(self): return {'lambda_left':float(self.lam_left),'lambda_right':float(self.lam_right),'selection_logloss':float(self.selection_logloss),'success':bool(self.success_)}

def monotonicity_audit(raw, calibrated):
    raw=np.asarray(raw,float); calibrated=np.asarray(calibrated,float)
    order=np.argsort(raw, kind="mergesort")
    dif=np.diff(calibrated[order])
    violations=int(np.sum(dif < -1e-12))
    ties=int(np.sum(np.abs(dif) <= 1e-12))
    from scipy.stats import spearmanr
    rho=float(spearmanr(raw,calibrated).statistic)
    return {"rank_violations":violations,"calibrated_ties":ties,"spearman_raw_calibrated":rho}

def load_join(data_dir,debug=False):
    tx_path=Path(data_dir)/'train_transaction.csv'; id_path=Path(data_dir)/'train_identity.csv'
    if debug:
        tx_head=pd.read_csv(tx_path,nrows=0).columns.tolist()
        core=['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain']
        patterned=[c for c in tx_head if c.startswith(('C','D','V'))][:45]
        tx_cols=[c for c in dict.fromkeys(core+patterned) if c in tx_head]
        id_head=pd.read_csv(id_path,nrows=0).columns.tolist()
        id_cols=['TransactionID']+[c for c in id_head if c!='TransactionID'][:18]
        tx=pd.read_csv(tx_path,usecols=tx_cols)
        ident=pd.read_csv(id_path,usecols=id_cols)
    else:
        tx=pd.read_csv(tx_path); ident=pd.read_csv(id_path)
    df=tx.merge(ident,on='TransactionID',how='left',validate='one_to_one').sort_values('TransactionDT').reset_index(drop=True)
    if debug: df=df.iloc[-20000:].reset_index(drop=True)
    return df

def choose_features(df,debug=False):
    drop={'isFraud','TransactionID'}
    cols=[c for c in df.columns if c not in drop]
    if debug:
        # Deterministic compact feature set: core fields + lowest missingness numeric predictors.
        core=[c for c in ['TransactionDT','TransactionAmt','ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain'] if c in cols]
        miss=df[cols].isna().mean().sort_values()
        extra=[c for c in miss.index if c not in core][:15]
        cols=list(dict.fromkeys(core+extra))
    return cols

def encode_fold(train,others,cols):
    Xtr=train[cols].copy(); Xos=[o[cols].copy() for o in others]
    for c in cols:
        if not pd.api.types.is_numeric_dtype(Xtr[c]):
            vals=pd.Series(Xtr[c].astype('string').fillna('__NA__').unique())
            mp={v:i for i,v in enumerate(vals)}
            Xtr[c]=Xtr[c].astype('string').fillna('__NA__').map(mp).fillna(-1)
            for X in Xos: X[c]=X[c].astype('string').fillna('__NA__').map(mp).fillna(-1)
    for c in cols:
        Xtr[c]=pd.to_numeric(Xtr[c],errors='coerce')
        for X in Xos: X[c]=pd.to_numeric(X[c],errors='coerce')
    imp=SimpleImputer(strategy='median')
    Xt=imp.fit_transform(Xtr).astype(np.float32)
    return [Xt]+[imp.transform(X).astype(np.float32) for X in Xos]

def make_model(name,threads,seed):
    if name=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=180,learning_rate=.05,num_leaves=31,subsample=.85,colsample_bytree=.8,reg_lambda=1.0,class_weight='balanced',n_jobs=threads,random_state=seed,verbosity=-1)
    if name=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=350,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.8,reg_lambda=1.0,eval_metric='logloss',tree_method='hist',n_jobs=threads,random_state=seed,scale_pos_weight=1.0)
    return LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=seed)

def folds_for(n,debug):
    if debug:
        # Compact but fully nonempty official-data debug sequence.
        sizes=(8000,3000,3000,3000,3000)
        assert sum(sizes)<=n, (n,sizes)
        a=0; out=[]
        bounds=np.cumsum((0,)+sizes)
        out.append(tuple((int(bounds[i]),int(bounds[i+1])) for i in range(5)))
        return out
    # Four rolling folds, each with 40k validation, 20k cal-fit, 20k cal-select, 50k future test.
    test_starts=[n-200000,n-150000,n-100000,n-50000]
    out=[]
    for ts in test_starts:
        cs=ts-20000; cfs=cs-20000; vs=cfs-40000
        tr_end=vs
        out.append(((0,tr_end),(tr_end, cfs),(cfs,cs),(cs,ts),(ts,min(ts+50000,n))))
    return out

def assess(y,p,amount):
    ci,cs=calibration_intercept_slope(y,p)
    row={'pr_auc':average_precision_score(y,p),'roc_auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'logloss':log_loss(y,p,labels=[0,1]),'ece':ece(y,p),'ici':ici(y,p),'cal_intercept':ci,'cal_slope':cs,'oe_ratio':oe_ratio(y,p),'pred_mean':float(np.mean(p))}
    for b in (.005,.01):
        m=top_budget(y,p,amount,b); suffix='005' if b==.005 else '01'
        row.update({f'recall_{suffix}':m['recall'],f'precision_{suffix}':m['precision'],f'value_capture_{suffix}':m['value_capture']})
    return row

def run(a):
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=load_join(a.data_dir,a.mode=='debug'); cols=choose_features(df,a.mode=='debug'); folds=folds_for(len(df),a.mode=='debug')
    models=['logistic'] if a.mode=='debug' else ['logistic','lightgbm','xgboost']
    grid=[.7,1.0,1.4] if a.mode=='debug' else [.20,.25,.35,.50,.70,1.00,1.40,2.00,2.85,4.00,5.50]
    from vendor.bgbcal import BGBCalibrator
    metrics=[]; params=[]; splits=[]
    for fi,b in enumerate(folds,1):
        parts=[df.iloc[s:e].copy() for s,e in b]
        names=['train','validation','cal_fit','cal_select','test']
        sizes={nm:len(x) for nm,x in zip(names,parts)}
        if min(sizes.values())<=0: raise RuntimeError(f'Empty temporal window fold={fi}: {sizes}')
        tr,va,cf,cs,te=parts
        if a.max_train and len(tr)>a.max_train:
            fraud=tr[tr.isFraud==1]; legit=tr[tr.isFraud==0].sample(n=max(a.max_train-len(fraud),1),random_state=SEED+fi)
            tr=pd.concat([fraud,legit]).sort_values('TransactionDT')
        Xtr,Xv,Xcf,Xcs,Xt=encode_fold(tr,[va,cf,cs,te],cols)
        ytr=tr.isFraud.to_numpy(int); ycf=cf.isFraud.to_numpy(int); ycs=cs.isFraud.to_numpy(int); yt=te.isFraud.to_numpy(int); amt=te.TransactionAmt.to_numpy(float)
        if len(np.unique(ycf))<2 or len(np.unique(ycs))<2 or len(np.unique(yt))<2: raise RuntimeError(f'Fold {fi} lacks both classes')
        splits.append({'fold':fi,**sizes,'train_used':len(tr),'event_train':float(ytr.mean()),'event_cal_fit':float(ycf.mean()),'event_cal_select':float(ycs.mean()),'event_test':float(yt.mean()),'n_features':len(cols)})
        for mn in models:
            t0=time.time(); m=make_model(mn,a.threads,SEED+fi); m.fit(Xtr,ytr); fit_time=time.time()-t0
            pcf=clip01(m.predict_proba(Xcf)[:,1]); pcs=clip01(m.predict_proba(Xcs)[:,1]); pt=clip01(m.predict_proba(Xt)[:,1])
            methods={'raw':(pt,{'success':True})}
            pl=PlattCal().fit(pcf,ycf); methods['platt']=(pl.predict(pt),pl.params())
            be=BetaCal().fit(pcf,ycf); methods['beta']=(be.predict(pt),be.params())
            iso=IsoCal().fit(pcf,ycf); methods['isotonic']=(iso.predict(pt),iso.params())
            bg=BGBCalibrator(method='bgb',shrink=a.bgb_shrink,max_iter=700).fit(pcf,ycf)
            methods['bgb']=(bg.predict_proba(pt)[:,1],{'success':True,'theta':float(bg.params_['theta']),'alpha':float(bg.params_['alpha']),'beta_inner':float(bg.params_['beta']),'a':float(bg.params_['a']),'b':float(bg.params_['b']),'nll_fit':float(bg.nll_),'aic':float(bg.aic_),'bic':float(bg.bic_)})
            ops=OPTransformLogistic(grid,two_sided=False,n_jobs=a.threads).fit(pcf,ycf,pcs,ycs)
            methods['opbgb_symmetric']=(ops.predict(pt),ops.params())
            opt=OPTransformLogistic(grid,two_sided=True,n_jobs=a.threads).fit(pcf,ycf,pcs,ycs)
            methods['opbgb_two_sided_unconstrained']=(opt.predict(pt),opt.params())
            penalties=(1e-3,1e-2) if a.mode=='debug' else (0.0,1e-4,1e-3,1e-2,1e-1)
            mos=SelectedMonotoneOPBGB(two_sided=False,penalties=penalties,max_iter=1200).fit(pcf,ycf,pcs,ycs)
            methods['opbgb_symmetric_monotone']=(mos.predict(pt),mos.parameters())
            mot=SelectedMonotoneOPBGB(two_sided=True,penalties=penalties,max_iter=1200).fit(pcf,ycf,pcs,ycs)
            methods['opbgb_two_sided_monotone']=(mot.predict(pt),mot.parameters())
            # selection based only on calibration-select log loss; constrained models are
            # refitted on fit+selection after the penalty is selected.
            candidates={
                'beta':be.predict(pcs),'bgb':bg.predict_proba(pcs)[:,1],
                'opbgb_symmetric_monotone':mos.predict(pcs),
                'opbgb_two_sided_monotone':mot.predict(pcs)}
            selected=min(candidates,key=lambda k:log_loss(ycs,clip01(candidates[k]),labels=[0,1]))
            methods['nested_selected']=methods[selected]
            for cal,(pred,par) in methods.items():
                audit=monotonicity_audit(pt,clip01(pred))
                row={'fold':fi,'model':mn,'calibration':cal,'selected_method':selected if cal=='nested_selected' else '', 'n_test':len(yt),'prevalence':float(yt.mean()),'fit_time_model':fit_time,**assess(yt,clip01(pred),amt),**audit}
                metrics.append(row); params.append({'fold':fi,'model':mn,'calibration':cal,**par,**audit})
            print(f'done fold={fi} model={mn} selected={selected}',flush=True)
    md=pd.DataFrame(metrics); pd.DataFrame(params).to_csv(out/'calibrator_parameters.csv',index=False); pd.DataFrame(splits).to_csv(out/'split_audit.csv',index=False); md.to_csv(out/'calibration_metrics.csv',index=False)
    summary=md.groupby(['model','calibration'],as_index=False).agg(pr_auc_mean=('pr_auc','mean'),pr_auc_sd=('pr_auc','std'),brier_mean=('brier','mean'),logloss_mean=('logloss','mean'),ece_mean=('ece','mean'),ici_mean=('ici','mean'),cal_intercept_mean=('cal_intercept','mean'),cal_slope_mean=('cal_slope','mean'),oe_ratio_mean=('oe_ratio','mean'),rank_violations_total=('rank_violations','sum'),spearman_min=('spearman_raw_calibrated','min'))
    summary.to_csv(out/'calibration_summary.csv',index=False)
    manifest={'mode':a.mode,'rows':len(df),'features':len(cols),'folds':len(folds),'models':models,'grid':grid,'threads':a.threads,'bgb_shrink':a.bgb_shrink,'metric_rows':len(md)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['debug','full'],default='debug'); p.add_argument('--data-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--threads',type=int,default=6); p.add_argument('--max-train',type=int,default=300000); p.add_argument('--bgb-shrink',type=float,default=1.0); run(p.parse_args())
