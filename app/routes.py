import json, heapq, csv, io, secrets
from datetime import datetime
from functools import wraps
from flask import Blueprint,render_template,request,redirect,url_for,flash,jsonify,abort
from flask_login import login_user,logout_user,login_required,current_user
from werkzeug.security import generate_password_hash,check_password_hash
from .models import db,User,Donor,Recipient,Match,Transport,TransportEvent,AuditLog,UserProfile,SecurityEvent,Notification,HospitalCapability,ModelExperiment,WhatIfRun,DatasetArtifact,MatchRequest
from .ml.engine import score,train,metrics,eligibility,compatibility_gate,load,feature_row,explain,FEATURES
from .transport import find_fastest_route, calculate_eta, check_transport_feasibility, TRAFFIC_FACTORS, ORGAN_MAX_MINUTES, compare_transport_modes, route_scenario, estimate_transport_mode, calculate_transport_risk
from .ai_lab import enhanced_match, model_cross_validation, dataset_quality, fairness_audit, what_if
bp=Blueprint('main',__name__)

@bp.before_app_request
def csrf_protect():
    if 'csrf_token' not in request.cookies:
        pass
    from flask import session
    if 'csrf_token' not in session: session['csrf_token']=secrets.token_urlsafe(32)
    if request.method=='POST' and request.endpoint not in {'main.login','main.register'}:
        token=(request.get_json(silent=True) or {}).get('_csrf') if request.is_json else request.form.get('_csrf')
        if not token or not secrets.compare_digest(token,session['csrf_token']): abort(400, description='Invalid or missing security token. Refresh the page and try again.')

@bp.route('/security/csrf-token')
def csrf_token():
    from flask import session, jsonify
    if 'csrf_token' not in session: session['csrf_token']=secrets.token_urlsafe(32)
    return jsonify({'token':session['csrf_token']})

BLOOD=['O-','O+','A-','A+','B-','B+','AB-','AB+']; ORGANS=['Kidney','Liver','Heart','Lung','Pancreas']; ROLES=['Admin','Doctor','Hospital','Donor','Recipient']
WORKFLOW_STATES=['REGISTERED','MATCHING','MATCH_FOUND','DONOR_REQUEST_SENT','DONOR_ACCEPTED','DOCTOR_PENDING','DOCTOR_APPROVED','HOSPITAL_PENDING','HOSPITAL_APPROVED','ADMIN_PENDING','ADMIN_VERIFIED','FINAL_APPROVED','ROUTE_CALCULATED','READY_FOR_TRANSPORT','IN_TRANSIT','ARRIVED','COMPLETED','DOCTOR_REJECTED','HOSPITAL_REJECTED','ADMIN_REJECTED','TRANSPORT_CANCELLED']
TRANSPORT_STATES=['PENDING','ROUTE_CALCULATED','READY_FOR_TRANSPORT','IN_TRANSIT','ARRIVED','COMPLETED','CANCELLED']

PERMISSIONS={'Admin':{'dashboard','data','matching','approvals','transport','analytics','audit','manage','security','reports','ai_lab','cases','simulation','notifications'},'Doctor':{'dashboard','matching','approvals','analytics','reports','ai_lab','cases','notifications'},'Hospital':{'dashboard','approvals','transport','analytics','reports','ai_lab','cases','notifications'},'Donor':{'dashboard','reports','cases','notifications'},'Recipient':{'dashboard','matching','reports','cases','notifications'}}
def allowed(page): return current_user.role in PERMISSIONS and page in PERMISSIONS[current_user.role]
def require_page(page):
    def deco(fn):
        @wraps(fn)
        def wrapped(*a,**k):
            if not allowed(page): abort(403)
            return fn(*a,**k)
        return wrapped
    return deco

def require_verified(fn):
    @wraps(fn)
    def wrapped_verified(*a,**k):
        if current_user.role != 'Admin':
            p=get_profile()
            if not p or p.verification_status != 'Verified': abort(403)
        return fn(*a,**k)
    return wrapped_verified

def transition_match(m, new_state, allowed_from):
    if m.workflow_state not in allowed_from:
        raise ValueError(f"Invalid workflow transition: {m.workflow_state} -> {new_state}")
    m.workflow_state=new_state
    return m

def log(action,entity=None,eid=None,details=''):
    db.session.add(AuditLog(user_id=current_user.id if current_user.is_authenticated else None,action=action,entity=entity,entity_id=eid,details=details)); db.session.commit()

def notify(user_id,title,message,category='INFO',reference_type=None,reference_id=None):
    if user_id:
        db.session.add(Notification(user_id=user_id,title=title,message=message,category=category,reference_type=reference_type,reference_id=reference_id))

def notify_roles(roles,title,message,category='INFO',exclude_user_id=None):
    for u in User.query.filter(User.role.in_(roles)).all():
        if exclude_user_id and u.id==exclude_user_id: continue
        notify(u.id,title,message,category)

def notify_donor_owner(donor, title, message, category='WORKFLOW', reference_type='Donor', reference_id=None):
    owner=UserProfile.query.filter_by(donor_id=donor.id).first()
    if owner: notify(owner.user_id,title,message,category,reference_type,reference_id or donor.id)


def notify_recipient_owner(recipient, title, message, category='WORKFLOW', reference_type='Recipient', reference_id=None):
    owner=UserProfile.query.filter_by(recipient_id=recipient.id).first()
    if owner: notify(owner.user_id,title,message,category,reference_type,reference_id or recipient.id)


def donor_owner_id(donor_id):
    p=UserProfile.query.filter_by(donor_id=donor_id).first(); return p.user_id if p else None


def recipient_owner_id(recipient_id):
    p=UserProfile.query.filter_by(recipient_id=recipient_id).first(); return p.user_id if p else None


def can_finalize_match(req):
    return all(v == 'ACCEPTED' for v in [req.recipient_status,req.donor_status,req.doctor_status,req.hospital_status,req.admin_status])


def candidate_matches(recipient):
    """Compute and persist ranked AI candidates only from fully authorized donors."""
    rows=[]
    if recipient.status not in ('REQUEST_ACTIVE','MATCHING','WAITING'):
        return rows
    donors=Donor.query.filter_by(organ=recipient.organ,status='AVAILABLE').all()
    for d in donors:
        result=enhanced_match(d,recipient)
        if not result: continue
        m=Match.query.filter_by(donor_id=d.id,recipient_id=recipient.id).order_by(Match.created_at.desc()).first()
        if m and m.final_status in ('FINAL_APPROVED','CLOSED','REJECTED'): continue
        if not m:
            m=Match(donor_id=d.id,recipient_id=recipient.id,score=result['score'],ml_score=result['ml_score'],confidence=result['confidence'],priority_score=result['priority'],risk_level=result['risk'],match_breakdown=json.dumps(result['breakdown']),model=metrics()['best_model'],explanation=json.dumps(result['explanation']),shap_data=json.dumps(result['shap']),status='Candidate',workflow_state='MATCHING',final_status='CANDIDATE')
            db.session.add(m)
        else:
            m.score=result['score']; m.ml_score=result['ml_score']; m.confidence=result['confidence']; m.priority_score=result['priority']; m.risk_level=result['risk']; m.match_breakdown=json.dumps(result['breakdown']); m.explanation=json.dumps(result['explanation']); m.shap_data=json.dumps(result['shap']); m.workflow_state='MATCHING'; m.status='Candidate'
        rows.append((d,result,m))
    rows.sort(key=lambda x:(x[1]['priority'],x[1]['score']),reverse=True)
    recipient.status='MATCHING' if rows else 'REQUEST_ACTIVE'
    if not rows:
        notify_recipient_owner(recipient,'No compatible donor available',f'No compatible {recipient.organ} donor is currently available. Your request remains active and will be checked again when a new donor becomes available.','WAITING','Recipient',recipient.id)
    db.session.commit()
    return rows


def check_waiting_recipient_requests(new_donor_id):
    donor=Donor.query.get(new_donor_id)
    if not donor or donor.status!='AVAILABLE': return 0
    notified=0
    active=Recipient.query.filter(Recipient.status.in_(['REQUEST_ACTIVE','MATCHING','WAITING']),Recipient.organ==donor.organ).all()
    for r in active:
        result=enhanced_match(donor,r)
        if not result: continue
        m=Match.query.filter_by(donor_id=donor.id,recipient_id=r.id).first()
        if not m:
            m=Match(donor_id=donor.id,recipient_id=r.id,score=result['score'],ml_score=result['ml_score'],confidence=result['confidence'],priority_score=result['priority'],risk_level=result['risk'],match_breakdown=json.dumps(result['breakdown']),model=metrics()['best_model'],explanation=json.dumps(result['explanation']),shap_data=json.dumps(result['shap']),status='Candidate',workflow_state='MATCHING',final_status='CANDIDATE')
            db.session.add(m)
        r.status='MATCHING'
        db.session.flush()
        notify_recipient_owner(r,'New compatible donor available',f'A new compatible {donor.organ} donor is available. Please review your ranked matches.', 'MATCH', 'Recipient', r.id)
        notified+=1
    db.session.commit()
    return notified


def ensure_recipient_active(r):
    return r and r.status in ('REQUEST_ACTIVE','MATCHING','WAITING') and r.doctor_approval=='ACCEPTED'


def case_timeline(m):
    events=[('CASE CREATED',m.created_at,'System'),('AI MATCH FOUND',m.created_at,'AI Engine')]
    if m.doctor_decided_at: events.append((f'DOCTOR {m.doctor_decision}',m.doctor_decided_at,m.doctor.name if m.doctor else 'Doctor'))
    if m.hospital_decided_at: events.append((f'HOSPITAL {m.hospital_decision}',m.hospital_decided_at,m.hospital_user.name if m.hospital_user else 'Hospital'))
    if m.admin_verified_at: events.append(('FINAL ADMIN AUTHORIZATION',m.admin_verified_at,m.admin_user.name if m.admin_user else 'Admin'))
    t=Transport.query.filter_by(match_id=m.id).order_by(Transport.created_at.asc()).all()
    for x in t:
        events.append(('ROUTE CALCULATED',x.created_at,'Transport Optimizer'))
        for e in TransportEvent.query.filter_by(transport_id=x.id).order_by(TransportEvent.created_at.asc()).all(): events.append((e.status,e.created_at,e.user.name if e.user else 'Transport'))
    return sorted(events,key=lambda x:x[1] or datetime.min)


def admin_seed():
    if not User.query.first():
        users=[('Government Security Admin','admin@lifelink.ai','admin123','Admin','Central Coordination'),('Dr. Meera Joshi','doctor@lifelink.ai','doctor123','Doctor','Sanjivani Medical Center'),('Hospital Coordinator','hospital@lifelink.ai','hospital123','Hospital','Sanjivani Medical Center'),('Donor Demo','donor@lifelink.ai','donor123','Donor','Sanjivani Medical Center'),('Recipient Demo','recipient@lifelink.ai','recipient123','Recipient','Sanjivani Medical Center')]
        for n,e,p,r,h in users: db.session.add(User(name=n,email=e,password=generate_password_hash(p),role=r,hospital=h))
        db.session.commit()
    else:
        demo=[('Donor Demo','donor@lifelink.ai','donor123','Donor','Sanjivani Medical Center'),('Recipient Demo','recipient@lifelink.ai','recipient123','Recipient','Sanjivani Medical Center')]
        for n,e,p,r,h in demo:
            if not User.query.filter_by(email=e).first(): db.session.add(User(name=n,email=e,password=generate_password_hash(p),role=r,hospital=h))
        db.session.commit()
    # Seed security profiles for existing accounts.
    for u in User.query.all():
        prof=UserProfile.query.filter_by(user_id=u.id).first()
        if not prof:
            prof=UserProfile(user_id=u.id, verification_status='Verified', account_status='Active')
            db.session.add(prof)
        # Built-in demo accounts are trusted demo identities. Existing registered
        # users remain Pending until an Admin explicitly verifies them.
        if u.email in {
            'admin@lifelink.ai','doctor@lifelink.ai','hospital@lifelink.ai',
            'donor@lifelink.ai','recipient@lifelink.ai'
        }:
            prof.verification_status='Verified'
            prof.account_status='Active'
        if u.role=='Donor' and not prof.donor_id:
            d=Donor.query.filter_by(name=u.name).first()
            if d: prof.donor_id=d.id
        if u.role=='Recipient' and not prof.recipient_id:
            r=Recipient.query.filter_by(name=u.name).first()
            if r: prof.recipient_id=r.id
    db.session.commit()
    # Seed configurable hospital capabilities for demo hospital accounts.
    for hu in User.query.filter_by(role='Hospital').all():
        if HospitalCapability.query.filter_by(hospital_user_id=hu.id).count()==0:
            for organ in ORGANS:
                db.session.add(HospitalCapability(hospital_user_id=hu.id,organ=organ,transplant_capable=(organ!='Lung'),emergency_transport=True,storage_capable=True,max_daily_cases=10))
    db.session.commit()
    if Donor.query.count()==0:
        rows=[('Donor A','Kidney','O+',32,91,12,15,'Pune',5),('Donor B','Kidney','A+',45,78,28,24,'Nashik',7),('Donor C','Liver','B+',29,88,18,10,'Ahmednagar',4),('Donor D','Heart','O-',41,84,10,20,'Pune',2),('Donor E','Kidney','O+',38,96,8,12,'Mumbai',3),('Donor F','Liver','AB+',51,82,20,18,'Nashik',6),('Donor G','Kidney','A+',36,94,10,14,'Pune',4)]
        for x in rows: db.session.add(Donor(name=x[0],organ=x[1],blood_group=x[2],age=x[3],hla_match=x[4],antibodies=x[5],medical_history=x[6],location=x[7],harvest_hours=x[8],status='AVAILABLE',doctor_approval='ACCEPTED',hospital_approval='ACCEPTED',admin_approval='ACCEPTED',consent='ACCEPTED',available_at=datetime.utcnow()))
        recs=[('Recipient 1','Kidney','O+',35,80,22,20,240,'Sanjivani Medical Center','Urgent'),('Recipient 2','Kidney','A+',50,72,35,30,90,'Nashik Civil Hospital','Normal'),('Recipient 3','Liver','B+',31,75,30,25,180,'Ahmednagar Hospital','Critical'),('Recipient 4','Heart','O+',44,81,18,17,120,'Pune Heart Institute','Urgent')]
        for x in recs: db.session.add(Recipient(name=x[0],organ=x[1],blood_group=x[2],age=x[3],hla_match=x[4],antibodies=x[5],medical_history=x[6],waiting_days=x[7],location=x[8],urgency=x[9],status='REQUEST_ACTIVE',doctor_approval='ACCEPTED',consent='ACCEPTED',activated_at=datetime.utcnow()))
        db.session.commit()
    # Normalize legacy demo registry records so the upgraded workflow starts with valid demo data.
    for d in Donor.query.all():
        if d.name.startswith('Donor '):
            d.status='AVAILABLE'; d.doctor_approval='ACCEPTED'; d.hospital_approval='ACCEPTED'; d.admin_approval='ACCEPTED'; d.consent='ACCEPTED'; d.available_at=d.available_at or datetime.utcnow()
    for r in Recipient.query.all():
        if r.name.startswith('Recipient '):
            r.status='REQUEST_ACTIVE'; r.doctor_approval='ACCEPTED'; r.consent='ACCEPTED'; r.activated_at=r.activated_at or datetime.utcnow()
    db.session.commit()

    # Link seeded demo donor/patient profiles to their registry records when possible.
    for u in User.query.filter(User.role.in_(['Donor','Recipient'])).all():
        prof=UserProfile.query.filter_by(user_id=u.id).first()
        if not prof: continue
        if u.role=='Donor' and not prof.donor_id:
            d=Donor.query.filter_by(name=u.name).first()
            if d: prof.donor_id=d.id
        if u.role=='Recipient' and not prof.recipient_id:
            r=Recipient.query.filter_by(name=u.name).first()
            if r: prof.recipient_id=r.id
    db.session.commit()

    # Seed a single welcome notification per demo account so the tab is visibly functional on first login.
    for u in User.query.filter(User.email.in_(['admin@lifelink.ai','doctor@lifelink.ai','hospital@lifelink.ai','donor@lifelink.ai','recipient@lifelink.ai'])).all():
        if Notification.query.filter_by(user_id=u.id,title='Welcome to LifeLink-AI').count()==0:
            db.session.add(Notification(user_id=u.id,title='Welcome to LifeLink-AI',message=f'Your {u.role} workspace is ready. Workflow updates will appear here.',category='SYSTEM',is_read=False))
    db.session.commit()

    # Create one pending AI case for first-run demonstration, without auto-approving it.
    if Match.query.count()==0:
        dd=Donor.query.filter_by(name='Donor G').first(); rr=Recipient.query.filter_by(name='Recipient 2').first()
        if dd and rr:
            result=enhanced_match(dd,rr)
            if result:
                mm=Match(donor_id=dd.id,recipient_id=rr.id,score=result['score'],ml_score=result['ml_score'],confidence=result['confidence'],priority_score=result['priority'],risk_level=result['risk'],match_breakdown=json.dumps(result['breakdown']),model=metrics()['best_model'],explanation=json.dumps(result['explanation']),shap_data=json.dumps(result['shap']),status='Candidate',workflow_state='MATCHING',final_status='CANDIDATE')
                db.session.add(mm); db.session.commit()

def get_profile(user=None):
    u=user or current_user
    return UserProfile.query.filter_by(user_id=u.id).first() if getattr(u,'id',None) else None

def security_event(event_type, severity='Low', risk_score=0, user=None, details=''):
    ip=request.remote_addr or 'unknown'
    db.session.add(SecurityEvent(user_id=(user.id if user else (current_user.id if current_user.is_authenticated else None)),event_type=event_type,severity=severity,risk_score=risk_score,ip_address=ip,details=details))
    db.session.commit()

def security_guard(user):
    profile=get_profile(user)
    return not profile or profile.account_status == 'Active'

def suspicious_score_for(user):
    from datetime import timedelta
    cutoff=datetime.utcnow()-timedelta(minutes=30)
    events=SecurityEvent.query.filter(SecurityEvent.user_id==user.id,SecurityEvent.created_at>=cutoff).all()
    failed=sum(1 for e in events if e.event_type=='Failed login')
    sensitive=sum(1 for e in events if e.event_type in ['Sensitive access','Bulk access'])
    score=min(100, failed*20+sensitive*15)
    return score, events

def monitor_sensitive_access(action):
    from datetime import timedelta
    cutoff=datetime.utcnow()-timedelta(minutes=10)
    recent=SecurityEvent.query.filter(SecurityEvent.user_id==current_user.id,SecurityEvent.event_type=='Sensitive access',SecurityEvent.created_at>=cutoff).count()
    if recent >= 10:
        p=get_profile()
        if p:
            p.risk_score=min(100,max(p.risk_score or 0,70))
            db.session.commit()
        security_event('Bulk access','High',75,current_user,f'Unusually frequent sensitive access: {action}; {recent+1} events in 10 minutes')

def page_data():
    unread=0
    if current_user.is_authenticated:
        unread=Notification.query.filter_by(user_id=current_user.id,is_read=False).count()
    return dict(active=request.endpoint or '', unread_notifications=unread)

@bp.route('/')
def home(): return render_template('landing.html',**page_data())
@bp.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        u=User.query.filter_by(email=email).first()
        if u and check_password_hash(u.password,password):
            profile=get_profile(u)
            if profile and profile.account_status != 'Active':
                security_event('Blocked login','High',80,u,'Login attempted on suspended/deactivated account')
                flash('This account is suspended or deactivated. Contact the authorized administrator.','danger')
                return render_template('login.html',**page_data())
            login_user(u); now=datetime.utcnow()
            if profile:
                profile.last_login=now; profile.last_ip=request.remote_addr or 'unknown'; profile.failed_logins=0; profile.risk_score=max(0,profile.risk_score-10)
                db.session.commit()
            security_event('Successful login','Low',0,u,'Role-based login completed')
            log('User login','User',u.id,f'Role={u.role}; IP={request.remote_addr or "unknown"}')
            return redirect(url_for('main.dashboard'))
        if u:
            profile=get_profile(u)
            if profile:
                profile.failed_logins=(profile.failed_logins or 0)+1; profile.risk_score=min(100,(profile.risk_score or 0)+20); db.session.commit()
            score_val,_=suspicious_score_for(u)
            severity='High' if score_val>=60 else 'Medium' if score_val>=20 else 'Low'
            security_event('Failed login',severity,score_val,u,f'Failed login for {email}')
            if profile and score_val>=80:
                profile.account_status='Suspended'; db.session.commit()
                security_event('Automatic security suspension','High',90,u,'Repeated failed login activity triggered the demo security rule')
                flash('Account temporarily suspended because of repeated suspicious login activity.','danger')
            else:
                flash('Invalid email or password','danger')
        else:
            security_event('Unknown account login attempt','Medium',25,None,f'Unknown email attempted login: {email}')
            flash('Invalid email or password','danger')
    return render_template('login.html',**page_data())

@bp.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        role=request.form.get('role','').strip().title()
        if role not in ['Donor','Recipient','Hospital']:
            flash('Please select Donor, Patient/Recipient, or Hospital.','danger'); return render_template('register.html',**page_data())
        name=request.form.get('name','').strip(); email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        if not name or not email or len(password)<8:
            flash('Name, email and a password of at least 8 characters are required.','danger'); return render_template('register.html',**page_data())
        if User.query.filter_by(email=email).first():
            flash('An account with this email already exists. Please sign in.','warning'); return render_template('register.html',**page_data())
        u=User(name=name,email=email,password=generate_password_hash(password),role=role,hospital=request.form.get('hospital','').strip() or None)
        db.session.add(u); db.session.flush()
        prof=UserProfile(user_id=u.id,phone=request.form.get('phone','').strip(),address=request.form.get('address','').strip(),government_id=request.form.get('government_id','').strip(),hospital_registration=request.form.get('hospital_registration','').strip(),verification_status='Pending',account_status='Active')
        db.session.add(prof); db.session.commit()
        if role=='Donor':
            d=Donor(name=name,organ=request.form.get('organ','Kidney'),blood_group=request.form.get('blood_group','O+'),age=int(request.form.get('age','18')),hla_match=float(request.form.get('hla_match','0') or 0),antibodies=float(request.form.get('antibodies','0') or 0),medical_history=float(request.form.get('medical_history','0') or 0),location=request.form.get('location','').strip() or 'Not provided',harvest_hours=float(request.form.get('harvest_hours','0') or 0),status='DOCTOR_PENDING')
            db.session.add(d); db.session.flush(); prof.donor_id=d.id
        elif role=='Recipient':
            r=Recipient(name=name,organ=request.form.get('organ','Kidney'),blood_group=request.form.get('blood_group','O+'),age=int(request.form.get('age','18')),hla_match=float(request.form.get('hla_match','0') or 0),antibodies=float(request.form.get('antibodies','0') or 0),medical_history=float(request.form.get('medical_history','0') or 0),waiting_days=0,location=request.form.get('location','').strip() or 'Not provided',urgency=request.form.get('urgency','Normal'),status='DOCTOR_PENDING')
            db.session.add(r); db.session.flush(); prof.recipient_id=r.id
        db.session.commit();
        if role=='Donor': notify_roles(['Doctor'],'Donor approval required',f'{name} registered as a donor and is waiting for doctor approval.','APPROVAL','')
        elif role=='Recipient': notify_roles(['Doctor'],'Recipient approval required',f'{name} submitted a {request.form.get("organ","Kidney")} request and is waiting for doctor approval.','APPROVAL','')
        db.session.commit(); security_event('Registration','Low',0,u,f'New {role} registration; verification pending')
        log('User registered','User',u.id,f'Role={role}; verification=Pending')
        flash('Registration submitted successfully. Your account is active, but profile verification is pending before sensitive workflow access.','success')
        return redirect(url_for('main.login'))
    return render_template('register.html',**page_data())

@bp.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    p=get_profile();
    if not p: abort(404)
    if request.method=='POST':
        p.phone=request.form.get('phone','').strip(); p.address=request.form.get('address','').strip()
        if current_user.role=='Donor' and p.donor:
            d=p.donor; d.location=request.form.get('location',d.location).strip(); d.blood_group=request.form.get('blood_group',d.blood_group); d.organ=request.form.get('organ',d.organ)
        elif current_user.role=='Recipient' and p.recipient:
            r=p.recipient; r.location=request.form.get('location',r.location).strip(); r.blood_group=request.form.get('blood_group',r.blood_group); r.organ=request.form.get('organ',r.organ); r.urgency=request.form.get('urgency',r.urgency)
        elif current_user.role=='Hospital':
            current_user.hospital=request.form.get('hospital',current_user.hospital).strip()
        db.session.commit(); log('Profile updated','User',current_user.id,'Self-service profile update'); flash('Profile updated successfully.','success'); return redirect(url_for('main.profile'))
    return render_template('profile.html',profile=p,organs=ORGANS,blood=BLOOD,**page_data())

@bp.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('main.home'))

@bp.route('/dashboard')
@login_required
@require_page('dashboard')
def dashboard():
    if current_user.role == 'Admin':
        return render_template('role_dashboard.html', role_key='admin', title='System Command Center', subtitle='Full coordination, governance and system health.',
            stats=[('Available Donors',Donor.query.filter_by(status='AVAILABLE').count(),'Ready for matching','bi-heart-pulse'),('Waiting Recipients',Recipient.query.filter(Recipient.status.in_(['REQUEST_ACTIVE','MATCHING','DONOR_SELECTED'])).count(),'Active requests','bi-person-heart'),('AI Recommendations',Match.query.count(),'Ranked candidates','bi-stars'),('Pending Approval',Match.query.filter_by(workflow_state='DOCTOR_PENDING').count(),'Doctor review required','bi-shield-exclamation')],
            actions=[('/security','Security Control','Verify users and investigate suspicious activity','bi-shield-lock'),('/data','Manage Registry','Add or update donor and recipient records','bi-people'),('/matching','Run AI Matching','Rank eligible donor candidates','bi-stars'),('/ai-lab','AI Research Lab','Cross-validation, fairness and data quality','bi-beaker'),('/reports','Reports','Review before/after decision records','bi-file-earmark-medical'),('/audit','Audit Trail','Review every important system action','bi-journal-text')], recent=Match.query.order_by(Match.created_at.desc()).limit(6).all(),**page_data())
    if current_user.role == 'Doctor':
        return render_template('role_dashboard.html', role_key='doctor', title='Clinical Review Dashboard', subtitle='Review AI recommendations, explanations and approval decisions.',
            stats=[('Waiting for Review',Match.query.filter_by(workflow_state='DOCTOR_PENDING').count(),'AI recommendations','bi-shield-exclamation'),('Hospital Queue',Match.query.filter_by(workflow_state='HOSPITAL_PENDING').count(),'Doctor-approved cases','bi-shield-check'),('Waiting Recipients',Recipient.query.filter(Recipient.status.in_(['REQUEST_ACTIVE','MATCHING','DONOR_SELECTED'])).count(),'Active requests','bi-person-heart'),('AI Candidates',Match.query.count(),'Total recommendations','bi-stars')],
            actions=[('/matching','AI Matching','Generate ranked donor candidates','bi-stars'),('/approvals','Approval Queue','Review and approve/reject','bi-check2-circle'),('/ai-lab','AI Research Lab','Cross-validation and fairness diagnostics','bi-beaker'),('/shap-dashboard','SHAP Explainability','Understand model contributions','bi-graph-up-arrow'),('/reports','Reports','View before/after decision records','bi-file-earmark-medical')], recent=Match.query.order_by(Match.created_at.desc()).limit(6).all(),**page_data())
    if current_user.role == 'Hospital':
        plans=Transport.query.order_by(Transport.created_at.desc()).limit(6).all()
        return render_template('role_dashboard.html', role_key='hospital', title='Hospital Operations Dashboard', subtitle='Coordinate approved organ transport and operational readiness.',
            stats=[('Admin Verified',Match.query.filter_by(workflow_state='FINAL_APPROVED').count(),'Route planning unlocked','bi-check-circle'),('Planned Routes',Transport.query.count(),'Calculated transport plans','bi-signpost-split'),('In Transit',Transport.query.filter_by(status='IN_TRANSIT').count(),'Active transfers','bi-truck'),('Completed',Transport.query.filter_by(status='COMPLETED').count(),'Completed transfers','bi-box-seam')],
            actions=[('/transport','Transport Control','Plan routes and update status','bi-signpost-split'),('/approvals','Approval Queue','Review hospital-ready cases','bi-check2-circle'),('/simulation','Transport Simulation','Test traffic and route re-optimization','bi-bezier2'),('/hospital-capabilities','Hospital Capabilities','Configure transplant and transport capability','bi-hospital'),('/reports','Operational Reports','View before/after reports','bi-file-earmark-medical'),('/profile','Hospital Profile','Update hospital information','bi-person')], recent=plans,**page_data())
    if current_user.role == 'Donor':
        profile=get_profile(); donor=profile.donor if profile else None;
        return render_template('role_dashboard.html', role_key='donor', title='Donor Portal', subtitle='Your donation profile and availability status.',
            stats=[('Donation Status', donor.status if donor else 'Not linked','Current registry state','bi-heart-pulse'),('Organ', donor.organ if donor else '—','Registered organ','bi-heart'),('Incoming Requests', MatchRequest.query.filter_by(donor_id=donor.id,donor_status='PENDING').count() if donor else 0,'Consent required','bi-inbox'),('HLA Match', f'{donor.hla_match:.0f}%' if donor else '—','Registry attribute','bi-shield-plus')],
            actions=[('/reports','My Reports','View your case reports','bi-file-earmark-medical'),('/profile','My Profile','Update permitted contact information','bi-person'),('/notifications','Notifications','Review incoming matching requests','bi-bell'),('/dashboard#status','Availability','Status is managed by authorized staff','bi-toggle-on')], recent=(Match.query.filter_by(donor_id=donor.id).order_by(Match.created_at.desc()).limit(6).all() if donor else []), incoming_requests=(MatchRequest.query.filter_by(donor_id=donor.id).order_by(MatchRequest.created_at.desc()).limit(10).all() if donor else []), profile=donor,**page_data())
    profile=get_profile(); recipient=profile.recipient if profile else None
    matches=Match.query.filter_by(recipient_id=recipient.id).order_by(Match.score.desc()).all() if recipient else []
    return render_template('role_dashboard.html', role_key='recipient', title='Recipient Portal', subtitle='Track your request and AI-assisted matching progress.',
        stats=[('Request Status',recipient.status if recipient else '—','Current request','bi-clipboard2-pulse'),('Organ',recipient.organ if recipient else '—','Requested organ','bi-heart'),('Urgency',recipient.urgency if recipient else '—','Priority level','bi-exclamation-triangle'),('Waiting Days',recipient.waiting_days if recipient else 0,'Time in registry','bi-hourglass-split')],
        actions=[('/matching','View Ranked Donors','Review AI-ranked available donors','bi-stars'),('/reports','My Reports','View your case reports','bi-file-earmark-medical'),('/profile','My Profile','Update permitted contact information','bi-person'),('/notifications','Notifications','View workflow updates','bi-bell')], recent=matches, profile=recipient,**page_data())

@bp.route('/matching')
@login_required
@require_page('matching')
@require_verified
def matching():
    monitor_sensitive_access('AI matching screen')
    if current_user.role=='Recipient':
        p=get_profile(); recipients=[p.recipient] if p and p.recipient and ensure_recipient_active(p.recipient) else []
    else:
        recipients=Recipient.query.filter(Recipient.status.in_(['REQUEST_ACTIVE','MATCHING','WAITING']),Recipient.doctor_approval=='ACCEPTED').all()
    rid=request.args.get('recipient_id',type=int)
    selected=Recipient.query.get(rid) if rid else (recipients[0] if recipients else None)
    if current_user.role=='Recipient' and selected and recipient_owner_id(selected.id)!=current_user.id: abort(403)
    rows=candidate_matches(selected) if selected else []
    return render_template('matching.html',recipients=recipients,selected=selected,rows=[(d,res) for d,res,_ in rows],**page_data())

@bp.route('/api/recommend',methods=['POST'])
@login_required
@require_page('matching')
@require_verified
def recommend():
    data=request.json or {}; r=Recipient.query.get_or_404(data.get('recipient_id'))
    if current_user.role=='Recipient' and recipient_owner_id(r.id)!=current_user.id: abort(403)
    rows=candidate_matches(r)
    return jsonify([{'donor_id':d.id,'donor':d.name,'score':res['score'],'ml_score':res['ml_score'],'confidence':res['confidence'],'priority':res['priority'],'risk':res['risk'],'breakdown':res['breakdown'],'reasons':res['reasons'],'shap':res['shap'],'feasibility':res['feasibility'],'gate':res.get('gate',{}),'model_source':res.get('model_source','')} for d,res,_ in rows])

@bp.route('/match/<int:did>/<int:rid>',methods=['POST'])
@login_required
@require_page('matching')
@require_verified
def create_match(did,rid):
    d=Donor.query.get_or_404(did); r=Recipient.query.get_or_404(rid)
    if current_user.role=='Recipient' and recipient_owner_id(r.id)!=current_user.id: abort(403)
    if d.status!='AVAILABLE' or not ensure_recipient_active(r):
        flash('This donor/recipient pair is not currently eligible for a matching request.','warning'); return redirect(url_for('main.matching',recipient_id=rid))
    result=enhanced_match(d,r)
    if not result: flash('This donor/recipient pair failed the compatibility gate.','warning'); return redirect(url_for('main.matching',recipient_id=rid))
    m=Match.query.filter_by(donor_id=did,recipient_id=rid).order_by(Match.created_at.desc()).first()
    if not m:
        m=Match(donor_id=did,recipient_id=rid,score=result['score'],ml_score=result['ml_score'],confidence=result['confidence'],priority_score=result['priority'],risk_level=result['risk'],match_breakdown=json.dumps(result['breakdown']),model=metrics()['best_model'],explanation=json.dumps(result['explanation']),shap_data=json.dumps(result['shap']),status='Donor Request Sent',workflow_state='DONOR_REQUEST_SENT',recipient_selected=True,selected_at=datetime.utcnow(),final_status='DONOR_PENDING')
        db.session.add(m); db.session.flush()
    else:
        m.recipient_selected=True; m.selected_at=datetime.utcnow(); m.workflow_state='DONOR_REQUEST_SENT'; m.status='Donor Request Sent'; m.final_status='DONOR_PENDING'
    req=MatchRequest.query.filter_by(match_id=m.id).first()
    if not req:
        req=MatchRequest(match_id=m.id,donor_id=d.id,recipient_id=r.id,recipient_status='ACCEPTED',donor_status='PENDING',doctor_status='PENDING',hospital_status='PENDING',admin_status='PENDING',final_status='DONOR_PENDING')
        db.session.add(req)
    r.status='DONOR_SELECTED'; d.status='RESERVED'
    owner=donor_owner_id(d.id)
    notify(owner,'Recipient sent a matching request',f'{r.name} selected you for an AI-ranked {d.organ} match. Please review and accept/reject the request.', 'MATCH', 'MatchRequest', req.id)
    notify_recipient_owner(r,'Donor request sent',f'Your request was sent to {d.name}. Waiting for donor consent.', 'MATCH', 'MatchRequest', req.id)
    db.session.commit(); log('Donor match request sent','MatchRequest',req.id,f'{r.name} -> {d.name}; score={m.score:.2f}')
    flash('Matching request sent to the donor. Final approval is not granted yet.','success')
    return redirect(url_for('main.dashboard'))

@bp.route('/match-request/<int:reqid>/donor-decision',methods=['POST'])
@login_required
@require_page('notifications')
@require_verified
def donor_decision(reqid):
    req=MatchRequest.query.get_or_404(reqid); d=req.donor
    if current_user.id!=donor_owner_id(d.id): abort(403)
    decision=request.form.get('decision','').upper()
    if req.donor_status!='PENDING' or req.final_status not in ('DONOR_PENDING','DONOR_REQUEST_SENT'): abort(400)
    req.donor_status='ACCEPTED' if decision=='ACCEPT' else 'REJECTED'; req.donor_decided_at=datetime.utcnow()
    if decision=='ACCEPT':
        req.final_status='DOCTOR_PENDING'; req.match.workflow_state='DOCTOR_PENDING'; req.match.status='Doctor Review Pending'; d.status='ALLOCATED'
        notify_roles(['Doctor'],'Selected match requires review',f'Case LL-{req.match_id:06d}: donor consent accepted; recipient and donor require doctor review.','APPROVAL',None)
        notify_recipient_owner(req.recipient,'Donor accepted request',f'{d.name} accepted your request. Doctor review is now required.','MATCH','MatchRequest',req.id)
    else:
        req.final_status='DONOR_REJECTED'; req.rejection_reason=request.form.get('reason','Donor rejected the request.'); req.match.workflow_state='MATCHING'; req.match.status='Donor Rejected'; req.match.final_status='REJECTED'; d.status='AVAILABLE'
        notify_recipient_owner(req.recipient,'Donor rejected request',f'{d.name} rejected the request. Your organ request remains active so you can select another ranked donor.','REJECTION','MatchRequest',req.id)
    db.session.commit(); log('Donor match request decision','MatchRequest',req.id,f'{decision}; state={req.final_status}')
    flash('Donor decision recorded.','success'); return redirect(url_for('main.dashboard'))

@bp.route('/approvals')
@login_required
@require_page('approvals')
@require_verified
def approvals():
    donor_pending=Donor.query.filter_by(status='DOCTOR_PENDING').all() if current_user.role=='Doctor' else []
    donor_hospital=Donor.query.filter_by(status='HOSPITAL_PENDING').all() if current_user.role=='Hospital' else []
    donor_admin=Donor.query.filter_by(status='ADMIN_PENDING').all() if current_user.role=='Admin' else []
    recipient_pending=Recipient.query.filter_by(status='DOCTOR_PENDING').all() if current_user.role=='Doctor' else []
    if current_user.role=='Doctor': requests=MatchRequest.query.filter_by(final_status='DOCTOR_PENDING').all()
    elif current_user.role=='Hospital': requests=MatchRequest.query.filter_by(final_status='HOSPITAL_PENDING').all()
    elif current_user.role=='Admin': requests=MatchRequest.query.filter_by(final_status='ADMIN_PENDING').all()
    else: requests=[]
    return render_template('approvals.html',matches=[x.match for x in requests],match_requests=requests,donor_pending=donor_pending,donor_hospital=donor_hospital,donor_admin=donor_admin,recipient_pending=recipient_pending,**page_data())

@bp.route('/donor/<int:did>/approval',methods=['POST'])
@login_required
@require_page('approvals')
@require_verified
def donor_approval(did):
    d=Donor.query.get_or_404(did); decision=request.form.get('decision','').upper(); reason=request.form.get('reason','').strip() or 'Reviewed by authorized role.'
    if decision not in ('ACCEPT','REJECT'): abort(400)
    if current_user.role=='Doctor' and d.status=='DOCTOR_PENDING':
        d.doctor_approval='ACCEPTED' if decision=='ACCEPT' else 'REJECTED'; d.approved_by_doctor=current_user.id; d.doctor_decided_at=datetime.utcnow(); d.rejection_reason=None if decision=='ACCEPT' else reason; d.status='HOSPITAL_PENDING' if decision=='ACCEPT' else 'DOCTOR_REJECTED'
        if decision=='ACCEPT': notify_roles(['Hospital'],'Donor approval required',f'Donor {d.name} ({d.organ}) passed doctor review and requires hospital authorization.','APPROVAL')
    elif current_user.role=='Hospital' and d.status=='HOSPITAL_PENDING':
        cap=HospitalCapability.query.filter_by(hospital_user_id=current_user.id,organ=d.organ,transplant_capable=True).first()
        if decision=='ACCEPT' and not cap: flash(f'Configure hospital capability for {d.organ} first.','danger'); return redirect(url_for('main.approvals'))
        d.hospital_approval='ACCEPTED' if decision=='ACCEPT' else 'REJECTED'; d.approved_by_hospital=current_user.id; d.hospital_decided_at=datetime.utcnow(); d.rejection_reason=None if decision=='ACCEPT' else reason; d.status='ADMIN_PENDING' if decision=='ACCEPT' else 'HOSPITAL_REJECTED'
        if decision=='ACCEPT': notify_roles(['Admin'],'Donor authorization required',f'Donor {d.name} ({d.organ}) requires final administrative authorization.','APPROVAL')
    elif current_user.role=='Admin' and d.status=='ADMIN_PENDING':
        d.admin_approval='ACCEPTED' if decision=='ACCEPT' else 'REJECTED'; d.approved_by_admin=current_user.id; d.admin_decided_at=datetime.utcnow(); d.rejection_reason=None if decision=='ACCEPT' else reason; d.status='AVAILABLE' if decision=='ACCEPT' else 'ADMIN_REJECTED'; d.available_at=datetime.utcnow() if decision=='ACCEPT' else None
        if decision=='ACCEPT':
            notify_donor_owner(d,'Donor is now AVAILABLE',f'Your {d.organ} donation record passed all authorization stages and is now available for matching.','APPROVAL')
    else: abort(403)
    db.session.commit();
    if decision=='ACCEPT' and d.status=='AVAILABLE': check_waiting_recipient_requests(d.id)
    log('Donor approval decision','Donor',d.id,f'role={current_user.role}; decision={decision}; state={d.status}; reason={reason}')
    flash(f'Donor {d.name}: {d.status}.','success' if decision=='ACCEPT' else 'warning'); return redirect(url_for('main.approvals'))

@bp.route('/recipient/<int:rid>/approval',methods=['POST'])
@login_required
@require_page('approvals')
@require_verified
def recipient_approval(rid):
    r=Recipient.query.get_or_404(rid)
    if current_user.role!='Doctor' or r.status!='DOCTOR_PENDING': abort(403)
    decision=request.form.get('decision','').upper(); reason=request.form.get('reason','').strip() or 'Reviewed by authorized doctor.'
    if decision=='ACCEPT':
        r.doctor_approval='ACCEPTED'; r.approved_by_doctor=current_user.id; r.doctor_decided_at=datetime.utcnow(); r.status='REQUEST_ACTIVE'; r.activated_at=datetime.utcnow(); notify_recipient_owner(r,'Recipient request approved',f'Your {r.organ} request is active and AI matching has started.','APPROVAL','Recipient',r.id)
        candidate_matches(r)
    elif decision=='REJECT':
        r.doctor_approval='REJECTED'; r.rejection_reason=reason; r.approved_by_doctor=current_user.id; r.doctor_decided_at=datetime.utcnow(); r.status='DOCTOR_REJECTED'; notify_recipient_owner(r,'Recipient request rejected',reason,'REJECTION','Recipient',r.id)
    else: abort(400)
    db.session.commit(); log('Recipient approval decision','Recipient',r.id,f'decision={decision}; state={r.status}; reason={reason}'); flash(f'Recipient request {decision.lower()}ed.','success' if decision=='ACCEPT' else 'warning'); return redirect(url_for('main.approvals'))

@bp.route('/match-request/<int:reqid>/approval',methods=['POST'])
@login_required
@require_page('approvals')
@require_verified
def match_request_approval(reqid):
    req=MatchRequest.query.get_or_404(reqid); decision=request.form.get('decision','').upper(); reason=request.form.get('reason','').strip() or 'Approved after role-authorized review.'
    if decision not in ('ACCEPT','REJECT'): abort(400)
    if current_user.role=='Doctor' and req.final_status=='DOCTOR_PENDING':
        req.doctor_status='ACCEPTED' if decision=='ACCEPT' else 'REJECTED'; req.doctor_decided_at=datetime.utcnow(); req.final_status='HOSPITAL_PENDING' if decision=='ACCEPT' else 'DOCTOR_REJECTED'; req.match.doctor_id=current_user.id; req.match.doctor_decision=decision; req.match.doctor_reason=reason; req.match.doctor_decided_at=datetime.utcnow(); req.match.workflow_state=req.final_status; req.match.status='Hospital Review Pending' if decision=='ACCEPT' else 'Doctor Rejected'; req.match.final_status=req.final_status if decision=='REJECT' else req.match.final_status;
        if decision=='ACCEPT': notify_roles(['Hospital'],'Selected match requires hospital approval',f'Case LL-{req.match_id:06d} is ready for hospital review.','APPROVAL')
        else: notify_recipient_owner(req.recipient,'Doctor rejected selected match',reason,'REJECTION','MatchRequest',req.id); req.recipient.status='REQUEST_ACTIVE'; req.donor.status='AVAILABLE'
    elif current_user.role=='Hospital' and req.final_status=='HOSPITAL_PENDING':
        cap=HospitalCapability.query.filter_by(hospital_user_id=current_user.id,organ=req.donor.organ,transplant_capable=True).first()
        if decision=='ACCEPT' and not cap: flash('Hospital capability is not configured for this organ.','danger'); return redirect(url_for('main.approvals'))
        req.hospital_status='ACCEPTED' if decision=='ACCEPT' else 'REJECTED'; req.hospital_decided_at=datetime.utcnow(); req.final_status='ADMIN_PENDING' if decision=='ACCEPT' else 'HOSPITAL_REJECTED'; req.match.hospital_id=current_user.id; req.match.hospital_decision=decision; req.match.hospital_reason=reason; req.match.hospital_decided_at=datetime.utcnow(); req.match.workflow_state=req.final_status; req.match.status='Admin Final Authorization Pending' if decision=='ACCEPT' else 'Hospital Rejected'; req.match.final_status=req.final_status if decision=='REJECT' else req.match.final_status;
        if decision=='ACCEPT': notify_roles(['Admin'],'Final match authorization required',f'Case LL-{req.match_id:06d} has donor, recipient, doctor and hospital acceptance and requires admin authorization.','APPROVAL')
        else: notify_recipient_owner(req.recipient,'Hospital rejected selected match',reason,'REJECTION','MatchRequest',req.id); req.recipient.status='REQUEST_ACTIVE'; req.donor.status='AVAILABLE'
    elif current_user.role=='Admin' and req.final_status=='ADMIN_PENDING':
        if decision=='ACCEPT':
            req.admin_status='ACCEPTED'
            if not can_finalize_match(req):
                req.admin_status='PENDING'; abort(400, description='Final approval blocked: recipient, donor, doctor, hospital and admin approvals are all required.')
        else:
            req.admin_status='REJECTED'
        req.admin_decided_at=datetime.utcnow(); req.final_status='FINAL_APPROVED' if decision=='ACCEPT' else 'ADMIN_REJECTED'; req.match.admin_verified_by=current_user.id; req.match.admin_verified_at=datetime.utcnow(); req.match.admin_reason=reason; req.match.workflow_state=req.final_status; req.match.status='Final Approved' if decision=='ACCEPT' else 'Admin Rejected'; req.match.final_status=req.final_status
        if decision=='ACCEPT': req.recipient.status='APPROVED'; req.donor.status='ALLOCATED'
        if decision=='ACCEPT': notify_recipient_owner(req.recipient,'Final match approved',f'Case LL-{req.match_id:06d} received all required approvals. Route optimization can now begin.','APPROVAL','MatchRequest',req.id); notify_donor_owner(req.donor,'Final match approved',f'Your donation match for {req.recipient.name} received all required approvals.','APPROVAL','MatchRequest',req.id); notify_roles(['Hospital'],'Final match approved',f'Case LL-{req.match_id:06d} is ready for route optimization.','TRANSPORT')
        else: notify_recipient_owner(req.recipient,'Final match rejected',reason,'REJECTION','MatchRequest',req.id); req.recipient.status='REQUEST_ACTIVE'; req.donor.status='AVAILABLE'
    else: abort(403)
    db.session.commit(); log('Match request approval','MatchRequest',req.id,f'role={current_user.role}; decision={decision}; state={req.final_status}; reason={reason}'); flash(f'Match request {decision.lower()}ed. Current state: {req.final_status}.','success' if decision=='ACCEPT' else 'warning'); return redirect(url_for('main.approvals'))

# Backward-compatible legacy Match approval endpoints remain available for old records.
@bp.route('/approve/<int:mid>',methods=['POST'])
@login_required
@require_page('approvals')
@require_verified
def approve(mid):
    m=Match.query.get_or_404(mid); req=MatchRequest.query.filter_by(match_id=m.id).first()
    if req: return match_request_approval(req.id)
    abort(403)

@bp.route('/reject/<int:mid>',methods=['POST'])
@login_required
@require_page('approvals')
@require_verified
def reject(mid):
    m=Match.query.get_or_404(mid); req=MatchRequest.query.filter_by(match_id=m.id).first()
    if req:
        request.form = request.form
        return match_request_approval(req.id)
    abort(403)

@bp.route('/admin/verify-case/<int:mid>',methods=['POST'])
@login_required
@require_page('security')
def verify_case(mid):
    if current_user.role!='Admin': abort(403)
    m=Match.query.get_or_404(mid); req=MatchRequest.query.filter_by(match_id=m.id).first()
    if req:
        if req.final_status!='ADMIN_PENDING': abort(403)
        req.admin_status='ACCEPTED'; req.admin_decided_at=datetime.utcnow(); req.final_status='FINAL_APPROVED'; m.admin_verified_by=current_user.id; m.admin_verified_at=datetime.utcnow(); m.workflow_state='FINAL_APPROVED'; m.final_status='FINAL_APPROVED'; m.status='Final Approved'; req.recipient.status='APPROVED'; req.donor.status='ALLOCATED'; notify_recipient_owner(req.recipient,'Final match approved',f'Case LL-{req.match_id:06d} received final authorization. Route optimization can now begin.','APPROVAL','MatchRequest',req.id); db.session.commit(); flash('Final match authorization recorded. Route planning is now unlocked.','success'); return redirect(url_for('main.security'))
    abort(403)

@bp.route('/admin/case-reject/<int:mid>',methods=['POST'])
@login_required
@require_page('security')
def reject_case_admin(mid):
    if current_user.role!='Admin': abort(403)
    m=Match.query.get_or_404(mid); req=MatchRequest.query.filter_by(match_id=m.id).first()
    if req and req.final_status=='ADMIN_PENDING':
        req.admin_status='REJECTED'; req.admin_decided_at=datetime.utcnow(); req.final_status='ADMIN_REJECTED'; m.workflow_state='ADMIN_REJECTED'; m.final_status='ADMIN_REJECTED'; m.status='Admin Rejected'; req.recipient.status='REQUEST_ACTIVE'; req.donor.status='AVAILABLE'; db.session.commit(); flash('Final match rejected; recipient request remains active.','warning'); return redirect(url_for('main.security'))
    abort(403)

@bp.route('/transport')
@login_required
@require_page('transport')
@require_verified
def transport():
    states=['FINAL_APPROVED','ROUTE_CALCULATED','READY_FOR_TRANSPORT','IN_TRANSIT','ARRIVED','COMPLETED']
    matches=Match.query.filter(Match.workflow_state.in_(states)).all()
    plans=Transport.query.order_by(Transport.created_at.desc()).all()
    return render_template('transport.html',matches=matches,plans=plans,traffic_factors=TRAFFIC_FACTORS,organ_limits=ORGAN_MAX_MINUTES,**page_data())

@bp.route('/route/<int:mid>',methods=['POST'])
@login_required
@require_page('transport')
@require_verified
def route(mid):
    if current_user.role not in ['Hospital','Admin']: abort(403)
    m=Match.query.get_or_404(mid)
    if m.workflow_state!='FINAL_APPROVED':
        flash('Route planning is locked until doctor approval, hospital approval and admin verification are complete.','warning'); return redirect(url_for('main.transport'))
    try:
        traffic=request.form.get('traffic','LOW').upper(); mode=request.form.get('mode','GROUND').upper()
        result=route_scenario(m.donor.location,m.recipient.location,traffic)
        mode_result=estimate_transport_mode(result['travel_hours'],mode)
        feasibility=check_transport_feasibility(m.donor.organ,mode_result['travel_hours'],m.donor.harvest_hours)
        transport_risk=calculate_transport_risk(feasibility,traffic,mode)
    except ValueError as exc:
        flash(str(exc),'danger'); return redirect(url_for('main.transport'))
    departure=datetime.utcnow(); eta_at=calculate_eta(mode_result['travel_hours'],departure)
    t=Transport(match_id=mid,source=m.donor.location,destination=m.recipient.location,distance=result['distance_km'],eta=mode_result['travel_hours'],estimated_minutes=feasibility['estimated_minutes'],max_allowed_minutes=feasibility['maximum_minutes'],remaining_minutes=feasibility['remaining_minutes'],feasible=feasibility['feasible'],traffic_condition=traffic, transport_mode=mode, route_risk=transport_risk['score'],departure_at=departure,eta_at=eta_at,algorithm=result['algorithm'],route_nodes=json.dumps(result['nodes']),route_points=json.dumps(result.get('route_points',[])),alternatives=json.dumps(result['alternatives']),status='ROUTE_CALCULATED')
    db.session.add(t); transition_match(m,'ROUTE_CALCULATED',['FINAL_APPROVED']); m.status='Route Calculated'; db.session.commit(); db.session.add(TransportEvent(transport_id=t.id,status='ROUTE_CALCULATED',changed_by=current_user.id,notes=feasibility['label'])); db.session.commit()
    # Notify the hospital and case stakeholders that a route is available.
    if m.hospital_id: notify(m.hospital_id,'Route calculated',f'Case LL-{m.id:06d} route calculated using {result["algorithm"]}.','TRANSPORT')
    notify_roles(['Admin'], 'Route calculated', f'Case LL-{m.id:06d} has a transport route and ETA.', 'TRANSPORT', exclude_user_id=current_user.id)
    db.session.commit()
    log('Fastest route calculated','Transport',t.id,f"{result['algorithm']}; mode={mode}; traffic={traffic}; {result['distance_km']} km; {mode_result['travel_hours']:.2f} hr; feasible={feasibility['feasible']}")
    flash(f"Fastest travel-time route calculated. {feasibility['label']}",'success'); return redirect(url_for('main.transport'))

@bp.route('/transport/<int:tid>/reoptimize',methods=['POST'])
@login_required
@require_page('transport')
@require_verified
def reoptimize_transport(tid):
    if current_user.role not in ['Hospital','Admin']: abort(403)
    t=Transport.query.get_or_404(tid); traffic=request.form.get('traffic','HIGH').upper(); mode=request.form.get('mode',t.transport_mode or 'GROUND').upper()
    try:
        result=route_scenario(t.source,t.destination,traffic); mode_result=estimate_transport_mode(result['travel_hours'],mode); f=check_transport_feasibility(t.match.donor.organ,mode_result['travel_hours'],t.match.donor.harvest_hours)
        transport_risk=calculate_transport_risk(f,traffic,mode)
    except ValueError as exc:
        flash(str(exc),'danger'); return redirect(url_for('main.transport'))
    t.distance=result['distance_km']; t.eta=mode_result['travel_hours']; t.route_risk=transport_risk['score']; t.estimated_minutes=f['estimated_minutes']; t.remaining_minutes=f['remaining_minutes']; t.feasible=f['feasible']; t.traffic_condition=traffic; t.transport_mode=mode; t.route_nodes=json.dumps(result['nodes']); t.route_points=json.dumps(result.get('route_points',[])); t.alternatives=json.dumps(result['alternatives']); t.algorithm=result['algorithm']; t.eta_at=calculate_eta(mode_result['travel_hours'],t.departure_at or datetime.utcnow()); db.session.add(TransportEvent(transport_id=t.id,status='ROUTE_REOPTIMIZED',changed_by=current_user.id,notes=f'Traffic={traffic}; mode={mode}; ETA={mode_result["travel_hours"]:.2f} hr')); notify_roles(['Admin','Doctor','Hospital'], 'Route re-optimized', f'Case LL-{t.match_id:06d}: ETA updated to {mode_result["travel_hours"]:.2f} hr under {traffic} traffic.', 'TRANSPORT', exclude_user_id=current_user.id); db.session.commit(); log('Transport route re-optimized','Transport',t.id,f'traffic={traffic}; mode={mode}; eta={mode_result["travel_hours"]:.2f}'); flash('Route re-optimized using the selected academic traffic scenario.','success'); return redirect(url_for('main.transport'))

@bp.route('/transport/<int:tid>/authorize',methods=['POST'])
@login_required
@require_page('transport')
@require_verified
def authorize_transport(tid):
    if current_user.role not in ['Hospital','Admin']: abort(403)
    t=Transport.query.get(tid) or Transport.query.filter_by(match_id=tid).order_by(Transport.created_at.desc()).first()
    if not t: abort(404)
    if t.status!='ROUTE_CALCULATED' or t.match.workflow_state!='ROUTE_CALCULATED': abort(403)
    t.status='TRANSPORT_AUTHORIZED'; t.match.workflow_state='READY_FOR_TRANSPORT'; t.match.status='Transport Authorized'
    db.session.add(TransportEvent(transport_id=t.id,status='TRANSPORT_AUTHORIZED',changed_by=current_user.id,notes='Authorized after final match approval and route calculation.'))
    notify_roles(['Admin','Doctor','Hospital'],'Transport authorized',f'Case LL-{t.match_id:06d}: transport is authorized and ready to start.','TRANSPORT',exclude_user_id=current_user.id)
    notify_recipient_owner(t.match.recipient,'Transport authorized',f'Case LL-{t.match_id:06d}: transport has been authorized after final match approval.','TRANSPORT','Match',t.match_id)
    db.session.commit(); log('Transport authorized','Transport',tid,f'case={t.match_id}; authorized_by={current_user.role}'); flash('Transport authorized. It can now be started.','success'); return redirect(url_for('main.transport'))

@bp.route('/transport/<int:tid>/status',methods=['POST'])
@login_required
@require_page('transport')
@require_verified
def transport_status(tid):
    if current_user.role not in ['Hospital','Admin']: abort(403)
    t=Transport.query.get_or_404(tid); new=request.form.get('status','').upper()
    allowed={ 'ROUTE_CALCULATED':[], 'TRANSPORT_AUTHORIZED':['IN_TRANSIT','CANCELLED'], 'READY_FOR_TRANSPORT':['IN_TRANSIT','CANCELLED'], 'IN_TRANSIT':['ARRIVED','CANCELLED'], 'ARRIVED':['COMPLETED'] }
    if new not in allowed.get(t.status,[]):
        flash(f'Invalid transport transition: {t.status} -> {new}','danger'); return redirect(url_for('main.transport'))
    t.status=new
    m=t.match
    if new=='TRANSPORT_AUTHORIZED': m.workflow_state='READY_FOR_TRANSPORT'; m.status='Transport Authorized'
    elif new=='READY_FOR_TRANSPORT': m.workflow_state='READY_FOR_TRANSPORT'; m.status='Ready for Transport'
    elif new=='IN_TRANSIT': m.workflow_state='IN_TRANSIT'; m.status='In Transit'
    elif new=='ARRIVED': m.workflow_state='ARRIVED'; m.status='Arrived'; t.actual_arrival_at=datetime.utcnow()
    elif new=='COMPLETED': m.workflow_state='COMPLETED'; m.status='Completed'; m.completed_at=datetime.utcnow(); m.recipient.status='COMPLETED'; m.donor.status='COMPLETED'; notify_recipient_owner(m.recipient,'Transport completed',f'Case LL-{m.id:06d} transport has been marked completed.','TRANSPORT','Match',m.id); notify_donor_owner(m.donor,'Transport completed',f'Case LL-{m.id:06d} transport has been marked completed.','TRANSPORT','Match',m.id)
    elif new=='CANCELLED': m.workflow_state='TRANSPORT_CANCELLED'; m.status='Transport Cancelled'
    db.session.add(TransportEvent(transport_id=t.id,status=new,changed_by=current_user.id,notes=request.form.get('notes','').strip())); notify_roles(['Admin','Doctor','Hospital'], 'Transport status updated', f'Case LL-{m.id:06d}: {new.replace("_"," " ).title()}.', 'TRANSPORT', exclude_user_id=current_user.id); db.session.commit(); log('Transport status updated','Transport',tid,f'{new}; case={m.id}'); flash(f'Transport status changed to {new.replace("_"," ").title()}.','success'); return redirect(url_for('main.transport'))

@bp.route('/data')
@login_required
@require_page('data')
def data():
    if current_user.role in ['Hospital','Doctor']: monitor_sensitive_access('Registry view')
    return render_template('data.html', donors=Donor.query.order_by(Donor.created_at.desc()).all(), recipients=Recipient.query.order_by(Recipient.created_at.desc()).all(), organs=ORGANS, blood=BLOOD, **page_data())

def _donor_form(form, donor=None):
    return dict(name=form.get('name','').strip(), organ=form.get('organ','Kidney'), blood_group=form.get('blood_group','O+'), age=int(form.get('age',0)), hla_match=float(form.get('hla_match',0)), antibodies=float(form.get('antibodies',0)), medical_history=float(form.get('medical_history',0)), location=form.get('location','').strip(), harvest_hours=float(form.get('harvest_hours',0)))

def _recipient_form(form, recipient=None):
    return dict(name=form.get('name','').strip(), organ=form.get('organ','Kidney'), blood_group=form.get('blood_group','O+'), age=int(form.get('age',0)), hla_match=float(form.get('hla_match',0)), antibodies=float(form.get('antibodies',0)), medical_history=float(form.get('medical_history',0)), waiting_days=int(form.get('waiting_days',0)), location=form.get('location','').strip(), urgency=form.get('urgency','Normal'))

@bp.route('/donor/<int:did>/edit', methods=['GET','POST'])
@login_required
@require_page('manage')
def donor_edit(did):
    d=Donor.query.get_or_404(did)
    if request.method=='POST':
        for k,v in _donor_form(request.form,d).items(): setattr(d,k,v)
        # Status is workflow-controlled; admin editing cannot bypass approval gates.
        db.session.commit(); log('Donor updated','Donor',d.id,d.name); flash('Donor record updated.','success'); return redirect(url_for('main.data'))
    return render_template('donor_edit.html', donor=d, organs=ORGANS, blood=BLOOD, **page_data())

@bp.route('/donor/<int:did>/delete', methods=['POST'])
@login_required
@require_page('manage')
def donor_delete(did):
    d=Donor.query.get_or_404(did)
    if Match.query.filter_by(donor_id=did).first():
        flash('This donor has matching history and cannot be deleted. Mark the donor unavailable instead.','warning'); return redirect(url_for('main.data'))
    name=d.name; db.session.delete(d); db.session.commit(); log('Donor deleted','Donor',did,name); flash('Donor record deleted.','success'); return redirect(url_for('main.data'))

@bp.route('/recipient/<int:rid>/edit', methods=['GET','POST'])
@login_required
@require_page('manage')
def recipient_edit(rid):
    r=Recipient.query.get_or_404(rid)
    if request.method=='POST':
        for k,v in _recipient_form(request.form,r).items(): setattr(r,k,v)
        # Recipient status is workflow-controlled; admin editing cannot bypass doctor approval.
        db.session.commit(); log('Recipient updated','Recipient',r.id,r.name); flash('Recipient request updated.','success'); return redirect(url_for('main.data'))
    return render_template('recipient_edit.html', recipient=r, organs=ORGANS, blood=BLOOD, **page_data())

@bp.route('/recipient/<int:rid>/delete', methods=['POST'])
@login_required
@require_page('manage')
def recipient_delete(rid):
    r=Recipient.query.get_or_404(rid)
    if Match.query.filter_by(recipient_id=rid).first():
        flash('This recipient has matching history and cannot be deleted. Close the request instead.','warning'); return redirect(url_for('main.data'))
    name=r.name; db.session.delete(r); db.session.commit(); log('Recipient deleted','Recipient',rid,name); flash('Recipient record deleted.','success'); return redirect(url_for('main.data'))

@bp.route('/donor/add',methods=['POST'])
@login_required
@require_page('manage')
def donor_add():
    d=Donor(name=request.form['name'],organ=request.form['organ'],blood_group=request.form['blood_group'],age=int(request.form['age']),hla_match=float(request.form['hla_match']),antibodies=float(request.form['antibodies']),medical_history=float(request.form['medical_history']),location=request.form['location'],harvest_hours=float(request.form.get('harvest_hours',0)),status='DOCTOR_PENDING'); db.session.add(d); db.session.commit(); notify_roles(['Doctor'],'Donor approval required',f'Donor {d.name} registered for {d.organ} and is waiting for doctor review.','APPROVAL'); db.session.commit(); log('Donor added','Donor',d.id,d.name); flash('Donor registered. Doctor approval is required before availability.','success'); return redirect(url_for('main.data'))

@bp.route('/recipient/add',methods=['POST'])
@login_required
@require_page('manage')
def recipient_add():
    r=Recipient(name=request.form['name'],organ=request.form['organ'],blood_group=request.form['blood_group'],age=int(request.form['age']),hla_match=float(request.form['hla_match']),antibodies=float(request.form['antibodies']),medical_history=float(request.form['medical_history']),waiting_days=int(request.form['waiting_days']),location=request.form['location'],urgency=request.form['urgency'],status='RECIPIENT_REGISTERED'); db.session.add(r); db.session.commit(); notify_roles(['Doctor'],'Recipient approval required',f'Recipient {r.name} registered a {r.organ} request and is waiting for doctor review.','APPROVAL'); db.session.commit(); log('Recipient added','Recipient',r.id,r.name); flash('Recipient request registered. Doctor approval is required before matching.','success'); return redirect(url_for('main.data'))

@bp.route('/donor/<int:did>/status',methods=['POST'])
@login_required
@require_page('manage')
def donor_status(did):
    d=Donor.query.get_or_404(did); new=request.form.get('status',''); allowed={'REGISTERED','DOCTOR_PENDING','DOCTOR_APPROVED','HOSPITAL_PENDING','HOSPITAL_APPROVED','ADMIN_PENDING','ADMIN_APPROVED','AVAILABLE','RESERVED','ALLOCATED','COMPLETED'};
    if current_user.role!='Admin' or new not in allowed: abort(403)
    d.status=new; d.available_at=datetime.utcnow() if new=='AVAILABLE' else d.available_at; db.session.commit();
    if new=='AVAILABLE': check_waiting_recipient_requests(d.id)
    log('Donor status updated','Donor',did,d.status); return redirect(url_for('main.data'))

@bp.route('/shap-dashboard')
@login_required
@require_page('matching')
def shap_dashboard():
    matches=Match.query.order_by(Match.created_at.desc()).limit(100).all()
    mid=request.args.get('match_id',type=int)
    selected=Match.query.get(mid) if mid else (matches[0] if matches else None)
    explanation=[]
    if selected:
        try:
            stored=json.loads(selected.shap_data or '[]')
        except Exception:
            stored=[]
        explanation=stored
        if not explanation:
            explanation=explain(feature_row(selected.donor,selected.recipient),load())
    positive=sorted([x for x in explanation if x['impact']>=0],key=lambda x:x['impact'],reverse=True)
    negative=sorted([x for x in explanation if x['impact']<0],key=lambda x:abs(x['impact']),reverse=True)
    return render_template('shap_dashboard.html',matches=matches,selected=selected,explanation=explanation,positive=positive,negative=negative,features=FEATURES,**page_data())

@bp.route('/api/shap/<int:mid>')
@login_required
@require_page('matching')
def shap_api(mid):
    m=Match.query.get_or_404(mid)
    try:
        data=json.loads(m.shap_data or '[]')
    except Exception:
        data=[]
    if not data:
        data=explain(feature_row(m.donor,m.recipient),load())
    return jsonify({'match_id':m.id,'donor':m.donor.name,'recipient':m.recipient.name,'score':round(m.score,2),'model':m.model,'features':data})

@bp.route('/reports')
@login_required
@require_page('reports')
def reports():
    if current_user.role in ['Donor','Recipient','Hospital','Doctor']: monitor_sensitive_access('Reports screen')
    if current_user.role in ['Admin','Doctor','Hospital']:
        matches=Match.query.order_by(Match.created_at.desc()).all()
    elif current_user.role=='Donor':
        profile=get_profile(); did=profile.donor_id if profile else None
        matches=Match.query.filter_by(donor_id=did).order_by(Match.created_at.desc()).all() if did else []
    else:
        profile=get_profile(); rid=profile.recipient_id if profile else None
        matches=Match.query.filter_by(recipient_id=rid).order_by(Match.created_at.desc()).all() if rid else []
    return render_template('reports.html',matches=matches,**page_data())

def _match_visible(m):
    if current_user.role in ['Admin','Doctor','Hospital']: return True
    if current_user.role=='Donor':
        p=get_profile(); return bool(p and p.donor_id==m.donor_id)
    if current_user.role=='Recipient':
        p=get_profile(); return bool(p and p.recipient_id==m.recipient_id)
    return False

@bp.route('/report/before/<int:mid>')
@login_required
@require_page('reports')
def before_report(mid):
    m=Match.query.get_or_404(mid)
    if not _match_visible(m): abort(403)
    try: explanation=json.loads(m.explanation or '[]')
    except Exception: explanation=[]
    transport=Transport.query.filter_by(match_id=mid).order_by(Transport.created_at.desc()).first()
    return render_template('report_before.html',match=m,explanation=explanation,transport=transport,**page_data())

@bp.route('/report/after/<int:mid>')
@login_required
@require_page('reports')
def after_report(mid):
    m=Match.query.get_or_404(mid)
    if not _match_visible(m): abort(403)
    if m.workflow_state not in ['ARRIVED','COMPLETED']: flash('The after-transplant report is available after authorized transport arrival.','warning'); return redirect(url_for('main.reports'))
    transport=Transport.query.filter_by(match_id=mid).order_by(Transport.created_at.desc()).first()
    return render_template('report_after.html',match=m,transport=transport,**page_data())

@bp.route('/security')
@login_required
@require_page('security')
def security():
    events=SecurityEvent.query.order_by(SecurityEvent.created_at.desc()).limit(200).all()
    profiles=UserProfile.query.join(User).order_by(User.created_at.desc()).all()
    high=sum(1 for e in events if e.severity=='High' and not e.resolved)
    pending_cases=Match.query.filter_by(workflow_state='ADMIN_PENDING').order_by(Match.created_at.desc()).all()
    return render_template('security.html',events=events,profiles=profiles,high_count=high,pending_cases=pending_cases,**page_data())

@bp.route('/security/user/<int:uid>/verify',methods=['POST'])
@login_required
@require_page('security')
def verify_user(uid):
    u=User.query.get_or_404(uid); p=get_profile(u); p.verification_status='Verified'; db.session.commit(); security_event('Profile verified','Low',0,u,f'Verified by {current_user.name}'); log('User verified','User',uid,u.email); flash('User profile verified.','success'); return redirect(url_for('main.security'))

@bp.route('/security/user/<int:uid>/suspend',methods=['POST'])
@login_required
@require_page('security')
def suspend_user(uid):
    u=User.query.get_or_404(uid);
    if u.role=='Admin': flash('Government/security administrator accounts cannot be suspended from this demo screen.','warning'); return redirect(url_for('main.security'))
    p=get_profile(u); p.account_status='Suspended'; db.session.commit(); security_event('Account suspended','High',80,u,f'Suspended by authorized administrator {current_user.name}'); log('Account suspended','User',uid,u.email); flash('Account suspended. Login is blocked until reactivated.','warning'); return redirect(url_for('main.security'))

@bp.route('/security/user/<int:uid>/activate',methods=['POST'])
@login_required
@require_page('security')
def activate_user(uid):
    u=User.query.get_or_404(uid); p=get_profile(u); p.account_status='Active'; p.risk_score=0; db.session.commit(); security_event('Account reactivated','Low',0,u,f'Reactivated by {current_user.name}'); log('Account reactivated','User',uid,u.email); flash('Account reactivated.','success'); return redirect(url_for('main.security'))

@bp.route('/security/user/<int:uid>/anonymize',methods=['POST'])
@login_required
@require_page('security')
def anonymize_user(uid):
    u=User.query.get_or_404(uid)
    if u.role=='Admin': flash('Administrator records cannot be anonymized from this screen.','warning'); return redirect(url_for('main.security'))
    p=get_profile(u); original=u.email; u.name=f'Deactivated User #{u.id}'; u.email=f'deactivated-{u.id}@lifelink.invalid'; u.password=generate_password_hash('disabled-account-'+str(u.id)); u.hospital=None
    if p: p.account_status='Deactivated'; p.verification_status='Revoked'; p.phone=None; p.address=None; p.government_id=None; p.hospital_registration=None
    db.session.commit(); security_event('Account data anonymized','High',90,u,f'Anonymized by authorized administrator {current_user.name}; original email {original}'); log('User data anonymized','User',uid,'Sensitive profile fields removed; audit history retained'); flash('User account data was anonymized and deactivated. Audit history was retained for traceability.','success'); return redirect(url_for('main.security'))

@bp.route('/security/event/<int:eid>/resolve',methods=['POST'])
@login_required
@require_page('security')
def resolve_security_event(eid):
    e=SecurityEvent.query.get_or_404(eid); e.resolved=True; db.session.commit(); log('Security alert resolved','SecurityEvent',eid,e.event_type); flash('Security alert marked as resolved.','success'); return redirect(url_for('main.security'))

@bp.route('/model-comparison')
@login_required
@require_page('analytics')
def model_comparison():
    from .ml.organ_models import get_bundle
    task=request.args.get('dataset','kidney').lower()
    if task not in ['heart','kidney','liver','lung']: task='kidney'
    bundle=get_bundle(task)
    return render_template('model_comparison.html', model_metrics=bundle.get('metrics',[]), best_model=bundle.get('best_model'), dataset=task, source=bundle.get('source',''), target=bundle.get('target',''), features=bundle.get('features',[]), **page_data())

@bp.route('/model-comparison/train',methods=['POST'])
@login_required
@require_page('analytics')
def retrain_models():
    from .ml.organ_models import train_task
    task=request.form.get('dataset','kidney').lower()
    if task not in ['heart','kidney','liver','lung']: task='kidney'
    bundle=train_task(task)
    for row in bundle.get('metrics',[]):
        db.session.add(ModelExperiment(model_name=row['model'],dataset=task,accuracy=row.get('accuracy',0),precision=row.get('precision',0),recall=row.get('recall',0),f1=row.get('f1',0),roc_auc=row.get('roc_auc',0)))
    db.session.commit()
    log('ML model retrained', 'Dataset', None, f'{task}: {bundle.get("best_model")}')
    flash(f'{task.title()} dataset models trained successfully. Best model: {bundle.get("best_model")}.','success')
    return redirect(url_for('main.model_comparison',dataset=task))

@bp.route('/datasets')
@login_required
@require_page('analytics')
def datasets_page():
    from .ml.organ_models import load_heart, get_bundle
    heart_df, heart_source = load_heart(); kb=get_bundle('kidney')
    artifacts=DatasetArtifact.query.order_by(DatasetArtifact.created_at.desc()).limit(20).all()
    experiments=ModelExperiment.query.order_by(ModelExperiment.created_at.desc()).limit(20).all()
    return render_template('datasets.html', heart_source=heart_source, heart_rows=len(heart_df), heart_columns=list(heart_df.columns), kidney_rows=3000, kidney_features=kb.get('features',[]), artifacts=artifacts, experiments=experiments, **page_data())

@bp.route('/datasets/upload',methods=['POST'])
@login_required
@require_page('analytics')
@require_verified
def dataset_upload():
    if current_user.role!='Admin': abort(403)
    file=request.files.get('dataset_file')
    if not file or not file.filename.lower().endswith('.csv'):
        flash('Please upload a CSV dataset.','danger'); return redirect(url_for('main.datasets_page'))
    import os, pandas as pd
    upload_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),'instance','uploads'); os.makedirs(upload_dir,exist_ok=True)
    safe=secrets.token_hex(6)+'_'+file.filename.replace(' ','_'); path=os.path.join(upload_dir,safe); file.save(path)
    try:
        df=pd.read_csv(path); missing=int(df.isna().sum().sum()); status='VALIDATED' if len(df)>20 and len(df.columns)>=2 else 'REJECTED'
        artifact=DatasetArtifact(filename=file.filename,stored_path=path,rows=len(df),columns=len(df.columns),missing_values=missing,validation_status=status,uploaded_by=current_user.id); db.session.add(artifact); db.session.commit(); log('Dataset uploaded','DatasetArtifact',artifact.id,f'rows={len(df)} columns={len(df.columns)} missing={missing}')
        flash(f'Dataset {file.filename} validated: {len(df)} rows, {len(df.columns)} columns, {missing} missing values.','success' if status=='VALIDATED' else 'warning')
    except Exception as exc:
        flash(f'Dataset validation failed: {exc}','danger')
    return redirect(url_for('main.datasets_page'))

@bp.route('/ai-lab')
@login_required
@require_page('ai_lab')
@require_verified
def ai_lab():
    cv_rows=model_cross_validation()
    quality=dataset_quality()
    fairness=fairness_audit()
    return render_template('ai_lab.html',cv_rows=cv_rows,quality=quality,fairness=fairness,**page_data())

@bp.route('/case/<int:mid>')
@login_required
@require_page('cases')
@require_verified
def case_detail(mid):
    m=Match.query.get_or_404(mid)
    if not _match_visible(m): abort(403)
    return render_template('case_detail.html',match=m,timeline=case_timeline(m),**page_data())

@bp.route('/what-if/<int:mid>', methods=['GET','POST'])
@login_required
@require_page('ai_lab')
@require_verified
def what_if_analysis(mid):
    m=Match.query.get_or_404(mid)
    if not _match_visible(m): abort(403)
    before=enhanced_match(m.donor,m.recipient)
    result=None; changes={}
    if request.method=='POST':
        changes={'hla_match':float(request.form.get('hla_match',m.donor.hla_match)), 'antibodies':float(request.form.get('antibodies',m.donor.antibodies)), 'urgency':request.form.get('urgency',m.recipient.urgency), 'waiting_days':int(request.form.get('waiting_days',m.recipient.waiting_days))}
        result=what_if(m.donor,m.recipient,changes)
        db.session.add(WhatIfRun(match_id=m.id,changes=json.dumps(changes),before_score=before['score'],after_score=result['score'],explanation=json.dumps(result['explanation']),created_by=current_user.id)); db.session.commit(); log('What-if analysis','Match',m.id,json.dumps(changes))
    return render_template('what_if.html',match=m,before=before,result=result,changes=changes,urgencies=['Normal','Urgent','Critical'],**page_data())

@bp.route('/simulation/run',methods=['POST'])
@login_required
@require_page('simulation')
@require_verified
def simulation_run():
    source=request.form.get('source','Pune'); destination=request.form.get('destination','Mumbai'); traffic=request.form.get('traffic','LOW'); mode=request.form.get('mode','GROUND')
    try:
        route=route_scenario(source,destination,traffic); mode_rows=compare_transport_modes(route['travel_hours']); selected=next(x for x in mode_rows if x['mode']==mode)
        return jsonify({'route':route,'traffic':traffic,'selected_mode':selected,'modes':mode_rows,'reoptimized':traffic.upper()!='LOW','disclaimer':'Academic scenario simulation; not live traffic/GPS or medical advice.'})
    except ValueError as exc: return jsonify({'error':str(exc)}),400

@bp.route('/simulation')
@login_required
@require_page('simulation')
@require_verified
def simulation():
    scenarios=[]
    for name,traffic,mode in [('Normal operations','LOW','Ground'),('Peak-hour pressure','HIGH','Ground'),('Emergency air transport','HIGH','Air')]:
        scenarios.append({'name':name,'traffic':traffic,'mode':mode,'description':'Configurable academic scenario; no live traffic is claimed.'})
    return render_template('simulation.html',scenarios=scenarios,**page_data())

@bp.route('/notifications')
@login_required
@require_page('notifications')
def notifications():
    items=Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).limit(100).all()
    unread=Notification.query.filter_by(user_id=current_user.id,is_read=False).count()
    return render_template('notifications.html',notifications=items,unread=unread,**page_data())

@bp.route('/notifications/read/<int:nid>',methods=['POST'])
@login_required
@require_page('notifications')
def notification_read(nid):
    n=Notification.query.filter_by(id=nid,user_id=current_user.id).first_or_404(); n.is_read=True; db.session.commit(); return redirect(url_for('main.notifications'))

@bp.route('/notifications/read-all',methods=['POST'])
@login_required
@require_page('notifications')
def notifications_read_all():
    Notification.query.filter_by(user_id=current_user.id,is_read=False).update({'is_read':True},synchronize_session=False)
    db.session.commit(); flash('All notifications marked as read.','success'); return redirect(url_for('main.notifications'))

@bp.route('/hospital-capabilities', methods=['GET','POST'])
@login_required
@require_page('transport')
@require_verified
def hospital_capabilities():
    if current_user.role not in ['Hospital','Admin']: abort(403)
    hu=current_user if current_user.role=='Hospital' else User.query.filter_by(role='Hospital').first()
    if request.method=='POST':
        hu_id=int(request.form.get('hospital_user_id',hu.id)); hu=User.query.get_or_404(hu_id)
        organ=request.form.get('organ','Kidney'); cap=HospitalCapability.query.filter_by(hospital_user_id=hu.id,organ=organ).first() or HospitalCapability(hospital_user_id=hu.id,organ=organ)
        cap.hospital_user_id=hu.id; cap.transplant_capable=request.form.get('transplant_capable')=='on'; cap.emergency_transport=request.form.get('emergency_transport')=='on'; cap.storage_capable=request.form.get('storage_capable')=='on'; cap.max_daily_cases=int(request.form.get('max_daily_cases','10')); db.session.add(cap); db.session.commit(); log('Hospital capability updated','HospitalCapability',cap.id,f'{hu.name} {organ}'); flash('Hospital capability updated.','success'); return redirect(url_for('main.hospital_capabilities'))
    caps=HospitalCapability.query.order_by(HospitalCapability.hospital_user_id,HospitalCapability.organ).all(); hospitals=User.query.filter_by(role='Hospital').all()
    return render_template('hospital_capabilities.html',caps=caps,hospitals=hospitals,organs=ORGANS,**page_data())

@bp.route('/analytics')
@login_required
@require_page('analytics')
def analytics():
    mm=metrics(); organ_counts={o:Donor.query.filter_by(organ=o).count() for o in ORGANS}; match_counts={s:Match.query.filter_by(status=s).count() for s in ['Recommended','Approved','Rejected','Completed']}
    completed=Match.query.filter_by(workflow_state='COMPLETED').count(); feasible=Transport.query.filter_by(feasible=True).count(); total_transport=Transport.query.count()
    return render_template('analytics.html',model_metrics=mm['metrics'],best_model=mm['best_model'],organ_counts=organ_counts,match_counts=match_counts,recipient_count=Recipient.query.count(),completed=completed,feasible=feasible,total_transport=total_transport,quality=dataset_quality(),**page_data())

@bp.route('/audit')
@login_required
@require_page('audit')
def audit(): return render_template('audit.html',logs=AuditLog.query.order_by(AuditLog.created_at.desc()).limit(100).all(),**page_data())

@bp.route('/admin/export-cases')
@login_required
@require_page('security')
def export_cases():
    if current_user.role!='Admin': abort(403)
    rows=Match.query.order_by(Match.created_at.desc()).all(); out=io.StringIO(); w=csv.writer(out); w.writerow(['case_id','donor','recipient','organ','score','workflow_state','doctor_decision','hospital_decision','admin_verified','created_at'])
    for m in rows: w.writerow([m.id,m.donor.name,m.recipient.name,m.donor.organ,round(m.score,2),m.workflow_state,m.doctor_decision or '',m.hospital_decision or '',bool(m.admin_verified_at),m.created_at.isoformat() if m.created_at else ''])
    log('Case data export','Match',None,f'Exported {len(rows)} case records by {current_user.name}')
    from flask import Response
    return Response(out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=lifelink_cases.csv'})

@bp.app_errorhandler(403)
def forbidden(e):
    if current_user.is_authenticated:
        security_event('Unauthorized endpoint attempt','Medium',35,current_user,f'Forbidden endpoint: {request.path}')
        log('Unauthorized endpoint attempt','Endpoint',None,request.path)
    return render_template('error.html',code=403,message='Your role does not have permission for this module.'),403
@bp.app_errorhandler(404)
def not_found(e): return render_template('error.html',code=404,message='The requested page was not found.'),404
