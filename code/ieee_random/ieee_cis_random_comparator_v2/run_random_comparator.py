#!/usr/bin/env python3
import os, json, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
BASE_SEED=20260716

def metrics(y,p,amt):
    out={'n_test':len(y),'prevalence':float(np.mean(y)),'pr_auc':average_precision_score(y,p),'roc_auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'logloss':log_loss(y,np.clip(p,1e-8,1-1e-8))}
    total=float(np.sum(amt[y==1]))
    for r in (0.005,0.01,0.02):
        k=max(1,int(np.ceil(len(y)*r))); ix=np.argsort(-p)[:k]
        out[f'precision_{r}']=float(np.mean(y[ix])); out[f'recall_{r}']=float(np.sum(y[ix])/max(1,np.sum(y)))
        out[f'value_capture_{r}']=float(np.sum(amt[ix][y[ix]==1])/total) if total>0 else np.nan
    return out

def platt_fit(pc,yc):
    z=np.clip(pc,1e-8,1-1e-8); x=np.log(z/(1-z)).reshape(-1,1)
    return LogisticRegression(C=1e6,max_iter=500,solver='lbfgs').fit(x,yc)
def platt_apply(m,p):
    z=np.clip(p,1e-8,1-1e-8); return m.predict_proba(np.log(z/(1-z)).reshape(-1,1))[:,1]

def load_data(data_dir, mode):
    tr=Path(data_dir)/'train_transaction.csv'; ident=Path(data_dir)/'train_identity.csv'
    if not tr.exists() or not ident.exists(): raise FileNotFoundError(f'Missing {tr} or {ident}')
    base=['TransactionID','isFraud','TransactionDT','TransactionAmt','ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain']
    base += [f'C{i}' for i in range(1,15)] + [f'D{i}' for i in range(1,16)] + [f'M{i}' for i in range(1,10)] + [f'V{i}' for i in range(1,340)]
    idc=['TransactionID','DeviceType','DeviceInfo']+[f'id_{i:02d}' for i in range(1,39)]
    tx=pd.read_csv(tr,usecols=lambda c:c in set(base),low_memory=False)
    idf=pd.read_csv(ident,usecols=lambda c:c in set(idc),low_memory=False)
    df=tx.merge(idf,on='TransactionID',how='left')
    if mode=='debug':
        # Preserve the original fraud prevalence in debug mode. The earlier
        # all-fraud + sampled-legitimate shortcut created an invalid enriched
        # test distribution and is deliberately avoided here.
        debug_n=min(80000, len(df))
        df,_=train_test_split(
            df,
            train_size=debug_n,
            stratify=df.isFraud,
            random_state=BASE_SEED,
        )
        df=df.copy()
    return df

def prepare(train, others, features):
    cats=[c for c in features if not is_numeric_dtype(train[c].dtype)]
    nums=[c for c in features if c not in cats]
    for c in cats:
        vals=train[c].astype('string').fillna('__NA__'); mp={v:i for i,v in enumerate(pd.Index(vals.unique()))}
        train[c]=vals.map(mp).fillna(-1).astype('int32')
        for x in others: x[c]=x[c].astype('string').fillna('__NA__').map(mp).fillna(-1).astype('int32')
    for c in nums:
        train[c]=pd.to_numeric(train[c],errors='coerce')
        for x in others: x[c]=pd.to_numeric(x[c],errors='coerce')
    med=train[nums].replace([np.inf,-np.inf],np.nan).median(numeric_only=True)
    train[nums]=train[nums].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
    for x in others: x[nums]=x[nums].replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
    bad=[c for c in features if not is_numeric_dtype(train[c].dtype)]
    if bad: raise TypeError(f'Non-numeric features remain: {bad[:20]}')
    return train, others

def model_factory(name,threads,seed):
    if name=='logistic': return make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,class_weight='balanced',solver='liblinear',random_state=seed))
    if name=='lightgbm': return LGBMClassifier(n_estimators=450,learning_rate=.04,num_leaves=63,subsample=.9,colsample_bytree=.8,class_weight='balanced',n_jobs=threads,random_state=seed,verbosity=-1)
    if name=='xgboost': return XGBClassifier(n_estimators=450,max_depth=7,learning_rate=.04,subsample=.9,colsample_bytree=.8,eval_metric='logloss',n_jobs=threads,random_state=seed,tree_method='hist')
    raise ValueError(name)

def stratified_parts(df, seed, max_train, debug=False):
    # Matched sizes: 300k train, 40k validation, 40k calibration, 50k test in full mode.
    sizes=(10000,3000,3000,5000) if debug else (max_train,40000,40000,50000)
    total=sum(sizes)
    if total>len(df): raise ValueError(f'Requested {total} rows from {len(df)}')
    pool,_=train_test_split(df,train_size=total,stratify=df.isFraud,random_state=seed)
    tr,rest=train_test_split(pool,train_size=sizes[0],stratify=pool.isFraud,random_state=seed+1)
    va,rest=train_test_split(rest,train_size=sizes[1],stratify=rest.isFraud,random_state=seed+2)
    ca,te=train_test_split(rest,train_size=sizes[2],test_size=sizes[3],stratify=rest.isFraud,random_state=seed+3)
    return [x.copy() for x in (tr,va,ca,te)]

def run(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=load_data(args.data_dir,args.mode)
    features=[c for c in df.columns if c not in ('TransactionID','isFraud','TransactionDT')]
    reps=1 if args.mode=='debug' else 4
    models=['logistic','lightgbm'] if args.mode=='debug' else ['logistic','lightgbm','xgboost']
    rows=[]
    for rep in range(1,reps+1):
        seed=BASE_SEED+rep*100
        tr,va,ca,te=stratified_parts(df,seed,args.max_train,args.mode=='debug')
        tr,[va,ca,te]=prepare(tr,[va,ca,te],features)
        Xtr,ytr=tr[features],tr.isFraud.values; Xv,yv=va[features],va.isFraud.values; Xc,yc=ca[features],ca.isFraud.values; Xt,yt=te[features],te.isFraud.values; amt=te.TransactionAmt.values
        for name in models:
            m=model_factory(name,args.threads,seed)
            if name=='lightgbm': m.fit(Xtr,ytr,eval_set=[(Xv,yv)],callbacks=[])
            else: m.fit(Xtr,ytr)
            pc=m.predict_proba(Xc)[:,1]; pt=m.predict_proba(Xt)[:,1]
            r=metrics(yt,pt,amt); r.update({'replicate':rep,'seed':seed,'model':name,'calibration':'raw','train_n':len(tr),'val_n':len(va),'cal_n':len(ca)}); rows.append(r)
            pm=platt_fit(pc,yc); pp=platt_apply(pm,pt)
            r=metrics(yt,pp,amt); r.update({'replicate':rep,'seed':seed,'model':name,'calibration':'platt','train_n':len(tr),'val_n':len(va),'cal_n':len(ca)}); rows.append(r)
            print(f'done replicate={rep} model={name}',flush=True)
    res=pd.DataFrame(rows); res.to_csv(out/'random_metrics.csv',index=False)
    summary=res.groupby(['model','calibration']).agg(pr_auc_mean=('pr_auc','mean'),pr_auc_sd=('pr_auc','std'),roc_auc_mean=('roc_auc','mean'),brier_mean=('brier','mean'),logloss_mean=('logloss','mean'),recall_005=('recall_0.005','mean'),precision_005=('precision_0.005','mean'),value_capture_005=('value_capture_0.005','mean'),recall_01=('recall_0.01','mean'),precision_01=('precision_0.01','mean'),value_capture_01=('value_capture_0.01','mean')).reset_index()
    summary.to_csv(out/'random_summary.csv',index=False)
    if args.temporal_metrics and Path(args.temporal_metrics).exists():
        tm=pd.read_csv(args.temporal_metrics)
        t=tm.groupby(['model','calibration']).agg(temporal_pr_auc=('pr_auc','mean'),temporal_roc_auc=('roc_auc','mean'),temporal_brier=('brier','mean'),temporal_logloss=('logloss','mean')).reset_index()
        cmp=summary.merge(t,on=['model','calibration'],how='left')
        cmp['pr_auc_optimism']=cmp.pr_auc_mean-cmp.temporal_pr_auc
        cmp['roc_auc_optimism']=cmp.roc_auc_mean-cmp.temporal_roc_auc
        cmp['brier_difference']=cmp.brier_mean-cmp.temporal_brier
        cmp['logloss_difference']=cmp.logloss_mean-cmp.temporal_logloss
        cmp.to_csv(out/'random_vs_temporal.csv',index=False)
    json.dump({'mode':args.mode,'rows_loaded':len(df),'frauds_loaded':int(df.isFraud.sum()),'loaded_prevalence':float(df.isFraud.mean()),'features':len(features),'replicates':reps,'models':models,'threads':args.threads,'max_train':args.max_train},open(out/'manifest.json','w'),indent=2)
    print(summary.to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['debug','full'],default='debug'); ap.add_argument('--data-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--threads',type=int,default=6); ap.add_argument('--max-train',type=int,default=300000); ap.add_argument('--temporal-metrics',default=''); run(ap.parse_args())
