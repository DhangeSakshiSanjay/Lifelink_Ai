"""Research-grade decision-support utilities for LifeLink-AI.

These functions intentionally remain academic/demo decision-support logic. They do
not provide clinical eligibility or transplant recommendations.
"""
import json, math, time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix
from .ml.engine import load, feature_row, eligibility, compatibility_gate, score, synth
from .transport import find_fastest_route, check_transport_feasibility

URGENCY_SCORE = {'Normal': 25, 'Urgent': 65, 'Critical': 95}
ORGAN_QUALITY_DEFAULT = 80


def compatibility_components(d, r):
    blood = 100.0 if d.blood_group == r.blood_group else (75.0 if d.blood_group.startswith('O') or r.blood_group.startswith('AB') else 0.0)
    hla = max(0.0, min(100.0, float(d.hla_match or 0)))
    antibody = max(0.0, min(100.0, 100.0 - float(d.antibodies or 0)))
    medical = max(0.0, min(100.0, 100.0 - float(d.medical_history or 0)))
    age_gap = abs(int(d.age or 0) - int(r.age or 0))
    age = max(0.0, 100.0 - age_gap * 5.0)
    urgency = URGENCY_SCORE.get(r.urgency, 25)
    waiting = min(100.0, float(r.waiting_days or 0) / 5.0)
    return {'blood_compatibility': blood, 'hla_compatibility': hla,
            'antibody_safety': antibody, 'medical_condition': medical,
            'age_compatibility': age, 'urgency': urgency, 'waiting_priority': waiting}


def transport_component(d, r):
    try:
        route = find_fastest_route(d.location, r.location)
        f = check_transport_feasibility(d.organ, route['travel_hours'], d.harvest_hours)
        max_m = max(1, f['maximum_minutes'])
        remaining = max(0.0, f['remaining_minutes'])
        feasibility = min(100.0, max(0.0, remaining / max_m * 100.0))
        return feasibility, route, f
    except Exception:
        return 50.0, None, {'feasible': True, 'estimated_minutes': None, 'remaining_minutes': None, 'maximum_minutes': None}


def enhanced_match(d, r):
    """Return a transparent multi-objective ranking built on the existing ML model."""
    gate = compatibility_gate(d, r)
    if not gate['passed']:
        return None
    ml_score, reasons, shap_values = score(d, r)
    c = compatibility_components(d, r)
    transport, route, feasibility = transport_component(d, r)
    weights = {
        'ml_compatibility': .25, 'medical_condition': .10,
        'hla_compatibility': .20, 'blood_compatibility': .15,
        'urgency': .10, 'waiting_priority': .08,
        'age_compatibility': .04, 'transport_feasibility': .08
    }
    values = {
        'ml_compatibility': ml_score, 'medical_condition': c['medical_condition'],
        'hla_compatibility': c['hla_compatibility'], 'blood_compatibility': c['blood_compatibility'],
        'urgency': c['urgency'], 'waiting_priority': c['waiting_priority'],
        'age_compatibility': c['age_compatibility'], 'transport_feasibility': transport
    }
    final = sum(values[k] * weights[k] for k in weights)
    # Confidence combines model probability and agreement with deterministic factors.
    agreement = 100.0 - min(100.0, abs(ml_score - (c['hla_compatibility'] * .55 + c['blood_compatibility'] * .25 + c['medical_condition'] * .20)))
    confidence = max(50.0, min(99.0, .7 * ml_score + .3 * agreement))
    risk = 'LOW' if final >= 80 else 'MEDIUM' if final >= 60 else 'HIGH'
    priority = min(100.0, .65 * c['urgency'] + .25 * c['waiting_priority'] + .10 * transport)
    breakdown = {k: round(values[k], 2) for k in values}
    breakdown['weights'] = weights
    explanation = [
        f"Final multi-factor score is {final:.1f}%.",
        f"ML compatibility contributes {ml_score:.1f}% with {confidence:.1f}% estimated model agreement confidence.",
        f"HLA compatibility is {c['hla_compatibility']:.1f}% and blood compatibility is {c['blood_compatibility']:.1f}%.",
        f"Recipient urgency is {r.urgency} and waiting-time priority is {c['waiting_priority']:.1f}%.",
        f"Transport feasibility contribution is {transport:.1f}%.",
    ]
    return {
        'score': round(final, 2), 'ml_score': round(ml_score, 2),
        'gate': gate,
        'model_source': 'Organ-specific kidney model when available; otherwise general matching benchmark',
        'confidence': round(confidence, 2), 'priority': round(priority, 2),
        'risk': risk, 'breakdown': breakdown, 'reasons': reasons,
        'shap': shap_values, 'route': route, 'feasibility': feasibility,
        'explanation': explanation
    }


def model_cross_validation():
    """5-fold evaluation of the same academic feature generator used by the baseline."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.base import clone
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = None
    X, y = synth(n=3200, seed=42)
    candidates = {
        'Random Forest': RandomForestClassifier(n_estimators=160, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=42)
    }
    if XGBClassifier:
        candidates['XGBoost'] = XGBClassifier(n_estimators=160, max_depth=5, learning_rate=.05, eval_metric='logloss', random_state=42)
    if LGBMClassifier:
        candidates['LightGBM'] = LGBMClassifier(n_estimators=160, num_leaves=31, learning_rate=.05, verbose=-1, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows=[]
    for name, model in candidates.items():
        started=time.perf_counter()
        out=cross_validate(model,X,y,cv=cv,scoring=['accuracy','precision','recall','f1','roc_auc'],n_jobs=1)
        rows.append({'model':name,'accuracy':round(out['test_accuracy'].mean(),4),'precision':round(out['test_precision'].mean(),4),'recall':round(out['test_recall'].mean(),4),'f1':round(out['test_f1'].mean(),4),'roc_auc':round(out['test_roc_auc'].mean(),4),'std_f1':round(out['test_f1'].std(),4),'training_seconds':round(time.perf_counter()-started,2)})
    return rows


def dataset_quality():
    X,y=synth(n=3200,seed=42)
    df=X.copy(); df['target']=y
    missing=int(df.isna().sum().sum()); duplicates=int(df.duplicated().sum())
    outliers={}
    for c in X.columns:
        q1,q3=X[c].quantile(.25),X[c].quantile(.75); iqr=q3-q1
        outliers[c]=int(((X[c]<q1-1.5*iqr)|(X[c]>q3+1.5*iqr)).sum())
    return {'rows':len(df),'features':len(X.columns),'missing':missing,'duplicates':duplicates,'outliers':outliers,'class_balance':{'negative':int((y==0).sum()),'positive':int((y==1).sum())},'source':'Academic synthetic benchmark generated transparently from configurable factors'}


def fairness_audit():
    X,y=synth(n=3200,seed=42); model=load(); pred=model.predict(X)
    age=X['age_gap']; groups={'Low gap (0-10)':age<=10,'Medium gap (11-25)':(age>10)&(age<=25),'High gap (>25)':age>25}
    rows=[]
    for name,mask in groups.items():
        yy=y[mask]; pp=np.asarray(pred)[mask]
        cm=confusion_matrix(yy,pp,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
        rows.append({'group':name,'records':int(mask.sum()),'positive_prediction_rate':round(float(pp.mean()),3),'false_positive_rate':round(fp/max(1,fp+tn),3),'false_negative_rate':round(fn/max(1,fn+tp),3)})
    return rows


def what_if(d,r,changes):
    class Obj: pass
    dd=Obj(); rr=Obj(); dd.__dict__.update(d.__dict__); rr.__dict__.update(r.__dict__)
    for k,v in changes.items():
        if hasattr(dd,k): setattr(dd,k,v)
        elif hasattr(rr,k): setattr(rr,k,v)
    return enhanced_match(dd,rr)
