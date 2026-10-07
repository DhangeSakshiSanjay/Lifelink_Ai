import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    import shap
except Exception:
    shap = None

FEATURES = ['age_gap','hla_match','antibodies','medical_history','waiting_days','urgency_score','organ_life_pressure']
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
BEST = os.path.join(MODEL_DIR, 'best.joblib')
METRICS = os.path.join(MODEL_DIR, 'metrics.joblib')

ORGANS = {'Kidney':24, 'Liver':12, 'Heart':6, 'Lung':6, 'Pancreas':12}
URGENCY = {'Routine':1, 'Normal':2, 'Urgent':3, 'Critical':4}

def synth(n=3200, seed=42):
    rng = np.random.default_rng(seed)
    age_d = rng.integers(18, 70, n); age_r = rng.integers(18, 76, n)
    hla = rng.uniform(35,100,n); ab = rng.uniform(0,100,n); mh = rng.uniform(0,100,n)
    wait = rng.integers(1,1000,n); urgency = rng.integers(1,5,n); life = rng.choice([6,12,24], n)
    age_gap = np.abs(age_d-age_r)
    latent = (.035*hla - .022*ab - .012*mh - .25*age_gap + .0018*wait + .7*urgency + .015*(24-life) + rng.normal(0,2.5,n))
    y = (latent > 1.4).astype(int)
    X = pd.DataFrame({'age_gap':age_gap,'hla_match':hla,'antibodies':ab,'medical_history':mh,'waiting_days':wait,'urgency_score':urgency,'organ_life_pressure':life})
    return X,y

def train():
    X,y=synth(); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    candidates={
      'Random Forest':RandomForestClassifier(n_estimators=220,max_depth=10,min_samples_leaf=2,random_state=42),
      'Gradient Boosting':GradientBoostingClassifier(n_estimators=160,max_depth=3,learning_rate=.05,random_state=42)
    }
    if XGBClassifier:
        candidates['XGBoost']=XGBClassifier(n_estimators=220,max_depth=5,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric='logloss',random_state=42)
    if LGBMClassifier:
        candidates['LightGBM']=LGBMClassifier(n_estimators=220,num_leaves=31,learning_rate=.05,verbose=-1,random_state=42)
    best=None; metrics=[]
    for name,m in candidates.items():
        m.fit(Xtr,ytr); p=m.predict(Xte); proba=m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else p
        row={'model':name,'accuracy':round(accuracy_score(yte,p),4),'precision':round(precision_score(yte,p,zero_division=0),4),'recall':round(recall_score(yte,p,zero_division=0),4),'f1':round(f1_score(yte,p,zero_division=0),4),'roc_auc':round(roc_auc_score(yte,proba),4)}
        metrics.append(row)
        if best is None or row['f1'] > best[0]: best=(row['f1'],name,m)
    joblib.dump(best[2],BEST); joblib.dump({'metrics':metrics,'best_model':best[1]},METRICS)
    return metrics,best[1]

def load():
    if not os.path.exists(BEST): train()
    return joblib.load(BEST)

def metrics():
    if not os.path.exists(METRICS): train()
    return joblib.load(METRICS)

def feature_row(d,r):
    urgency = URGENCY.get(getattr(r,'urgency','Normal'),2)
    life = ORGANS.get(getattr(r,'organ','Kidney'),24)
    return pd.DataFrame([{
      'age_gap':abs((d.age or 0)-(r.age or 0)), 'hla_match':d.hla_match or 0,
      'antibodies':d.antibodies or 0, 'medical_history':d.medical_history or 0,
      'waiting_days':r.waiting_days or 0, 'urgency_score':urgency, 'organ_life_pressure':life
    }])

def compatibility_gate(d, r):
    """Hierarchical academic compatibility gate used before ML ranking.

    The gate is intentionally transparent and configurable. It is NOT a clinical
    transplant eligibility rule. Each stage returns a pass/fail reason so the UI
    can explain why a candidate entered or failed the AI ranking stage.
    """
    stages = []
    organ_ok = getattr(d, 'organ', None) == getattr(r, 'organ', None)
    stages.append({'stage': 'Organ type', 'passed': organ_ok,
                   'reason': 'Donor and recipient requested the same organ.' if organ_ok else 'Organ types are different.'})
    donor_available = str(getattr(d, 'status', '')).upper() == 'AVAILABLE'
    stages.append({'stage': 'Donor availability', 'passed': donor_available,
                   'reason': 'Donor is marked Available.' if donor_available else 'Donor is not currently Available.'})
    recipient_waiting = str(getattr(r, 'status', '')).upper() in {'REQUEST_ACTIVE','MATCHING','WAITING'}
    stages.append({'stage': 'Recipient request', 'passed': recipient_waiting,
                   'reason': 'Recipient request is Waiting.' if recipient_waiting else 'Recipient request is not in Waiting state.'})
    db = getattr(d, 'blood_group', '') or ''
    rb = getattr(r, 'blood_group', '') or ''
    blood_ok = db == rb or db.startswith('O-') or rb == 'AB+'
    stages.append({'stage': 'Academic ABO gate', 'passed': blood_ok,
                   'reason': 'ABO compatibility rule passed.' if blood_ok else 'Academic ABO compatibility rule failed.'})
    passed = all(x['passed'] for x in stages)
    return {'passed': passed, 'stages': stages,
            'reason': 'All hard compatibility gates passed.' if passed else next(x['reason'] for x in stages if not x['passed'])}


def eligibility(d,r):
    return compatibility_gate(d, r)['passed']

def kidney_predict(d,r):
    """Predict kidney-pair compatibility using the organ-specific trained model."""
    try:
        from .organ_models import get_bundle
        bundle=get_bundle('kidney')
        model=joblib.load(MODEL_DIR / 'kidney_best.joblib')
        abo_compatible = int(d.blood_group == r.blood_group or d.blood_group.startswith('O') or r.blood_group.startswith('AB'))
        hla_mismatch = int(round(max(0,min(10, (100-float(d.hla_match or 0))/10))))
        x=pd.DataFrame([{'donor_age':d.age,'recipient_age':r.age,'donor_blood':d.blood_group.rstrip('+-'),'recipient_blood':r.blood_group.rstrip('+-'),'hla_mismatches':hla_mismatch,'hla_antibody_risk':int(float(d.antibodies or 0)>50),'age_gap':abs(d.age-r.age),'abo_compatible':abo_compatible}])
        # The trained pipeline ignores unknown columns, but requires its learned feature set.
        x=x[[c for c in bundle.get('features',list(x.columns)) if c in x.columns]]
        return float(model.predict_proba(x)[0,1]) if hasattr(model,'predict_proba') else float(model.predict(x)[0])
    except Exception:
        return None

def score(d,r):
    m=load(); x=feature_row(d,r)
    prob=float(m.predict_proba(x)[0,1]) if hasattr(m,'predict_proba') else float(m.predict(x)[0])
    if getattr(r,'organ','') == 'Kidney':
        kp=kidney_predict(d,r)
        if kp is not None: prob=kp
    reasons=[]
    vals=x.iloc[0]
    if vals.hla_match >= 80: reasons.append(('HLA compatibility', vals.hla_match, 'positive'))
    if vals.antibodies <= 25: reasons.append(('Low antibody indicator', 100-vals.antibodies, 'positive'))
    if vals.age_gap <= 10: reasons.append(('Low age gap', 100-vals.age_gap*5, 'positive'))
    if vals.waiting_days >= 180: reasons.append(('Waiting-time priority', min(vals.waiting_days/10,100), 'positive'))
    if vals.urgency_score >= 3: reasons.append(('Clinical urgency priority', vals.urgency_score*25, 'positive'))
    if not reasons: reasons.append(('Learned compatibility pattern', prob*100, 'positive'))
    shap_values = explain(x,m)
    return prob*100, reasons, shap_values

def explain(x,m):
    vals=[]
    if shap is not None:
        try:
            ex=shap.TreeExplainer(m); sv=ex.shap_values(x)
            if isinstance(sv,list): sv=sv[-1][0]
            else: sv=np.asarray(sv).reshape(-1)
            vals=[{'feature':f,'value':round(float(x.iloc[0][f]),2),'impact':round(float(v),4)} for f,v in zip(FEATURES,sv)]
            return sorted(vals,key=lambda z:abs(z['impact']),reverse=True)
        except Exception:
            pass
    # deterministic fallback explanation for environments without SHAP
    weights={'age_gap':-0.08,'hla_match':0.045,'antibodies':-0.035,'medical_history':-0.02,'waiting_days':0.001,'urgency_score':0.7,'organ_life_pressure':-0.01}
    for f in FEATURES: vals.append({'feature':f,'value':round(float(x.iloc[0][f]),2),'impact':round(weights[f]*float(x.iloc[0][f]),4)})
    return sorted(vals,key=lambda z:abs(z['impact']),reverse=True)
