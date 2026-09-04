import os, json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.special import logit
from scipy.stats import wilcoxon

ROOT=Path(os.environ.get('FRAUD_PROJECT_ROOT',Path.home()/'fraud_temporal_project')).expanduser()
DATA=Path(os.environ.get('SPARKOV_DATA_DIR',ROOT/'data/raw/sparkov')).expanduser()
OUT=Path(os.environ.get('FRAUD_RUN_DIR',ROOT/'analysis/sparkov_decisive')).expanduser(); OUT.mkdir(parents=True,exist_ok=True)
RES=OUT/'results'; RES.mkdir(exist_ok=True)
MODE=os.environ.get('RUN_MODE','debug')
THREADS=int(os.environ.get('THREADS','4'))
MAX_TRAIN=int(os.environ.get('MAX_TRAIN','120000' if MODE=='debug' else '350000'))
SEEDS=[20260716] if MODE=='debug' else [20260716,20260717]
DELAYS=[14] if MODE=='debug' else [0,7,14,30]
MODELS=os.environ.get('MODELS','logistic,lightgbm' if MODE=='debug' else 'logistic,lightgbm,xgboost').split(',')
POLICIES=os.environ.get('POLICIES','capped_weighted' if MODE=='debug' else 'capped_weighted,sliding180_weighted,sliding180_unweighted').split(',')

usecols=['trans_date_trans_time','cc_num','merchant','category','amt','gender','state','lat','long','city_pop','dob','merch_lat','merch_long','is_fraud']
print('Loading source files...')
df=pd.concat([pd.read_csv(DATA/f,usecols=usecols,parse_dates=['trans_date_trans_time','dob']) for f in ['fraudTrain.csv','fraudTest.csv']],ignore_index=True)
df=df.sort_values('trans_date_trans_time').reset_index(drop=True)
DEBUG_MAX_ROWS=int(os.environ.get('DEBUG_MAX_ROWS','350000'))
if MODE=='debug' and DEBUG_MAX_ROWS>0 and len(df)>DEBUG_MAX_ROWS:
    df=df.tail(DEBUG_MAX_ROWS).reset_index(drop=True)
df['row_id']=np.arange(len(df))
df['hour']=df.trans_date_trans_time.dt.hour.astype('int8'); df['dow']=df.trans_date_trans_time.dt.dayofweek.astype('int8'); df['month']=df.trans_date_trans_time.dt.month.astype('int8')
df['age']=((df.trans_date_trans_time-df.dob).dt.days/365.25).clip(18,100).astype('float32')
lat1=np.radians(df.lat.values); lon1=np.radians(df.long.values); lat2=np.radians(df.merch_lat.values); lon2=np.radians(df.merch_long.values)
a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
df['distance_km']=(6371*2*np.arcsin(np.sqrt(a))).astype('float32'); df['log_amt']=np.log1p(df.amt).astype('float32'); df['log_city_pop']=np.log1p(df.city_pop).astype('float32')

print('Building point-in-time card features...')
def card_features(g):
    t=(g.trans_date_trans_time.astype('int64')//10**9).to_numpy(); amt=g.amt.to_numpy(float); n=len(g); idx=np.arange(n); cs=np.r_[0.,np.cumsum(amt)]; out={}
    for name,sec in [('1h',3600),('24h',86400),('7d',604800)]:
        left=np.searchsorted(t,t-sec,'left'); out[f'card_n_{name}']=(idx-left).astype('int32'); out[f'card_amt_{name}']=(cs[idx]-cs[left]).astype('float32')
    prev=np.r_[np.nan,t[:-1]]; out['secs_since_prev_card']=np.where(np.isnan(prev),np.nan,t-prev).astype('float32')
    out['card_prior_mean_amt']=np.divide(cs[idx],idx,out=np.full(n,np.nan),where=idx>0).astype('float32')
    return pd.DataFrame(out,index=g.index)
cf=df.groupby('cc_num',sort=False,group_keys=False).apply(card_features,include_groups=False)
for c in cf: df[c]=cf[c]
df['amt_over_card_mean']=(df.amt/(df.card_prior_mean_amt+1)).astype('float32')

print('Building merchant features for delay scenarios...')
for delay in DELAYS:
    cnt=np.zeros(len(df),np.int32); fr=np.zeros(len(df),np.int32)
    for _,idxs in df.groupby('merchant',sort=False).groups.items():
        ix=np.asarray(list(idxs),dtype=np.int64); t=(df.loc[ix,'trans_date_trans_time'].astype('int64')//10**9).to_numpy(); y=df.loc[ix,'is_fraud'].to_numpy(np.int32)
        j=np.searchsorted(t,t-delay*86400,side='left'); cs=np.r_[0,np.cumsum(y)]; cnt[ix]=j; fr[ix]=cs[j]
    df[f'merchant_n_d{delay}']=cnt; df[f'merchant_rate_d{delay}']=((fr+1)/(cnt+200)).astype('float32')

base_num=['log_amt','age','hour','dow','month','log_city_pop','distance_km','card_n_1h','card_n_24h','card_n_7d','card_amt_1h','card_amt_24h','card_amt_7d','secs_since_prev_card','card_prior_mean_amt','amt_over_card_mean']
cat=['category','gender','state']
for c in base_num:
    df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan); df[c]=df[c].fillna(df[c].median())

folds=[]
for test_month in pd.period_range('2020-07','2020-12',freq='M'):
    test_start=test_month.start_time; test_end=(test_month+1).start_time; cal_start=(test_month-1).start_time; val_start=(test_month-2).start_time
    folds.append((str(test_month),val_start,cal_start,test_start,test_end))
if MODE=='debug': folds=folds[-1:]

def training_frame(val_start,policy,seed):
    hist=df[df.trans_date_trans_time<val_start]
    if policy.startswith('sliding180'): hist=hist[hist.trans_date_trans_time>=val_start-pd.Timedelta(days=180)]
    if policy=='capped_weighted' and len(hist)>MAX_TRAIN:
        pos=hist[hist.is_fraud==1]; neg=hist[hist.is_fraud==0].sample(MAX_TRAIN-len(pos),random_state=seed); hist=pd.concat([pos,neg]).sort_values('trans_date_trans_time')
    return hist

def calibrator(p,y,kind):
    p=np.clip(p,1e-7,1-1e-7)
    if kind=='platt': X=logit(p).reshape(-1,1)
    else: X=np.c_[np.log(p),-np.log(1-p)]
    m=LogisticRegression(C=1e6,max_iter=1000).fit(X,y)
    return lambda q:m.predict_proba(logit(np.clip(q,1e-7,1-1e-7)).reshape(-1,1) if kind=='platt' else np.c_[np.log(np.clip(q,1e-7,1-1e-7)),-np.log(1-np.clip(q,1e-7,1-1e-7))])[:,1]

def cal_stats(y,p):
    p=np.clip(p,1e-7,1-1e-7); X=logit(p).reshape(-1,1)
    try:
        m=LogisticRegression(C=1e6,max_iter=1000).fit(X,y); return float(m.intercept_[0]),float(m.coef_[0,0])
    except Exception:return np.nan,np.nan

def met(y,p,amt):
    p=np.clip(p,1e-12,1-1e-12); out={'pr_auc':average_precision_score(y,p),'roc_auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'logloss':log_loss(y,p,labels=[0,1])}
    ci,cs=cal_stats(y,p); out['cal_intercept']=ci; out['cal_slope']=cs
    fraud_amt=amt[y==1].sum()
    for frac in [0.001,0.005,0.01,0.02]:
        k=max(1,int(np.ceil(frac*len(y)))); ix=np.argpartition(-p,k-1)[:k]; tp=y[ix].sum()
        out[f'precision_{frac}']=tp/k; out[f'recall_{frac}']=tp/max(1,y.sum()); out[f'value_capture_{frac}']=amt[ix][y[ix]==1].sum()/max(1e-9,fraud_amt)
    return out

rows=[]; preds=[]
for fold,val_start,cal_start,test_start,test_end in folds:
  val=df[(df.trans_date_trans_time>=val_start)&(df.trans_date_trans_time<cal_start)]
  cal=df[(df.trans_date_trans_time>=cal_start)&(df.trans_date_trans_time<test_start)]
  test=df[(df.trans_date_trans_time>=test_start)&(df.trans_date_trans_time<test_end)]
  for delay in DELAYS:
    num=base_num+[f'merchant_n_d{delay}',f'merchant_rate_d{delay}']
    for policy in POLICIES:
      for seed in SEEDS:
        train=training_frame(val_start,policy,seed)
        enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
        Xtr=np.c_[train[num].to_numpy(np.float32),enc.fit_transform(train[cat]).astype(np.float32)]
        Xv=np.c_[val[num].to_numpy(np.float32),enc.transform(val[cat]).astype(np.float32)]
        Xc=np.c_[cal[num].to_numpy(np.float32),enc.transform(cal[cat]).astype(np.float32)]
        Xt=np.c_[test[num].to_numpy(np.float32),enc.transform(test[cat]).astype(np.float32)]
        ytr=train.is_fraud.to_numpy(); yc=cal.is_fraud.to_numpy(); yt=test.is_fraud.to_numpy(); amt=test.amt.to_numpy()
        weighted=policy!='sliding180_unweighted'; ratio=(len(ytr)-ytr.sum())/max(1,ytr.sum())
        model_map={
          'logistic':Pipeline([('s',StandardScaler()),('m',LogisticRegression(max_iter=500,class_weight='balanced' if weighted else None))]),
          'lightgbm':LGBMClassifier(n_estimators=350,learning_rate=.06,num_leaves=31,subsample=.85,colsample_bytree=.85,class_weight='balanced' if weighted else None,random_state=seed,n_jobs=THREADS,verbosity=-1),
          'xgboost':XGBClassifier(n_estimators=300,learning_rate=.07,max_depth=7,subsample=.85,colsample_bytree=.85,min_child_weight=3,eval_metric='aucpr',tree_method='hist',n_jobs=THREADS,random_state=seed,scale_pos_weight=ratio if weighted else 1.0)
        }
        for name in MODELS:
          m=model_map[name]; st=time.time(); m.fit(Xtr,ytr); fitsec=time.time()-st
          pc=m.predict_proba(Xc)[:,1]; pt=m.predict_proba(Xt)[:,1]
          for ck in ['raw','platt','beta']:
            pp=pt if ck=='raw' else calibrator(pc,yc,ck)(pt)
            z=met(yt,pp,amt); z.update({'fold':fold,'delay_days':delay,'policy':policy,'seed':seed,'model':name,'calibration':ck,'n_train':len(train),'train_prev':ytr.mean(),'n_test':len(test),'test_prev':yt.mean(),'fit_seconds':fitsec}); rows.append(z)
          preds.append(pd.DataFrame({'fold':fold,'delay_days':delay,'policy':policy,'seed':seed,'model':name,'row_id':test.row_id,'y':yt,'amount':amt,'p_raw':pt}))
        print('done',fold,delay,policy,seed)

r=pd.DataFrame(rows); r.to_csv(RES/'decisive_metrics.csv',index=False)
pd.concat(preds,ignore_index=True).to_csv(RES/'decisive_predictions.csv.gz',index=False,compression='gzip')
summary=r.groupby(['model','policy','delay_days','calibration']).agg(pr_auc=('pr_auc','mean'),pr_auc_sd=('pr_auc','std'),roc_auc=('roc_auc','mean'),brier=('brier','mean'),logloss=('logloss','mean'),cal_intercept=('cal_intercept','mean'),cal_slope=('cal_slope','mean'),recall_005=('recall_0.005','mean'),precision_005=('precision_0.005','mean'),value_capture_005=('value_capture_0.005','mean'),recall_01=('recall_0.01','mean'),value_capture_01=('value_capture_0.01','mean')).reset_index(); summary.to_csv(RES/'decisive_summary.csv',index=False)
# paired tests on raw PR-AUC across fold/seed for best delay=14, capped policy
stat=[]; q=r[(r.delay_days==14)&(r.policy=='capped_weighted')&(r.calibration=='raw')]
for a,b in [('xgboost','lightgbm'),('lightgbm','logistic'),('xgboost','logistic')]:
    aa=q[q.model==a].sort_values(['fold','seed']); bb=q[q.model==b].sort_values(['fold','seed']); d=aa.pr_auc.to_numpy()-bb.pr_auc.to_numpy()
    if len(d):
      try: W,p=wilcoxon(d)
      except: W,p=np.nan,np.nan
      stat.append({'model_a':a,'model_b':b,'n_pairs':len(d),'mean_difference':d.mean(),'median_difference':np.median(d),'wilcoxon_W':W,'p_value':p})
pd.DataFrame(stat).to_csv(RES/'paired_tests.csv',index=False)
with open(RES/'run_manifest.json','w') as f: json.dump({'mode':MODE,'folds':[x[0] for x in folds],'seeds':SEEDS,'delays':DELAYS,'policies':POLICIES,'models':MODELS,'max_train':MAX_TRAIN,'debug_max_rows':DEBUG_MAX_ROWS if MODE=='debug' else None},f,indent=2)
print(summary.sort_values('pr_auc',ascending=False).head(20).to_string(index=False))
