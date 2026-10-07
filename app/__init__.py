import os, json
from flask import Flask
from flask_login import LoginManager
from .config import Config
from .models import db,User,Match
from .routes import bp,admin_seed

login_manager=LoginManager()

def _migrate_sqlite_columns():
    """Small idempotent migration for projects upgraded from the supplied codebase."""
    from sqlalchemy import text, inspect
    inspector=inspect(db.engine)
    additions={
      'donor': [('doctor_approval','VARCHAR(20)'),('hospital_approval','VARCHAR(20)'),('admin_approval','VARCHAR(20)'),('consent','VARCHAR(20)'),('approved_by_doctor','INTEGER'),('approved_by_hospital','INTEGER'),('approved_by_admin','INTEGER'),('doctor_decided_at','DATETIME'),('hospital_decided_at','DATETIME'),('admin_decided_at','DATETIME'),('rejection_reason','TEXT'),('available_at','DATETIME')],
      'recipient': [('doctor_approval','VARCHAR(20)'),('consent','VARCHAR(20)'),('approved_by_doctor','INTEGER'),('doctor_decided_at','DATETIME'),('rejection_reason','TEXT'),('activated_at','DATETIME')],
      'match': [('workflow_state','VARCHAR(40)'),('ml_score','FLOAT'),('confidence','FLOAT'),('priority_score','FLOAT'),('risk_level','VARCHAR(20)'),('match_breakdown','TEXT'),('doctor_id','INTEGER'),('doctor_decision','VARCHAR(20)'),('doctor_reason','TEXT'),('doctor_decided_at','DATETIME'),('hospital_id','INTEGER'),('hospital_decision','VARCHAR(20)'),('hospital_reason','TEXT'),('hospital_decided_at','DATETIME'),('admin_verified_by','INTEGER'),('admin_verified_at','DATETIME'),('admin_reason','TEXT'),('recipient_selected','BOOLEAN'),('selected_at','DATETIME'),('final_status','VARCHAR(40)'),('completed_at','DATETIME')],
      'notification': [('reference_type','VARCHAR(40)'),('reference_id','INTEGER')],
      'transport': [('estimated_minutes','FLOAT'),('max_allowed_minutes','FLOAT'),('remaining_minutes','FLOAT'),('feasible','BOOLEAN'),('traffic_condition','VARCHAR(30)'),('departure_at','DATETIME'),('eta_at','DATETIME'),('actual_arrival_at','DATETIME'),('transport_provider','VARCHAR(160)'),('transport_mode','VARCHAR(40)'),('route_risk','FLOAT'),('notes','TEXT'),('alternatives','TEXT'),('route_points','TEXT')]
    }
    for table,cols in additions.items():
        if not inspector.has_table(table): continue
        existing={c['name'] for c in inspect(db.engine).get_columns(table)}
        for name,typ in cols:
            if name not in existing:
                db.session.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {typ}'))
    db.session.commit()

def create_app():
    app=Flask(__name__); app.config.from_object(Config); app.jinja_env.filters['fromjson']=json.loads
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)),'instance'),exist_ok=True)
    db.init_app(app); login_manager.init_app(app); login_manager.login_view='main.login'; app.register_blueprint(bp)
    @login_manager.user_loader
    def load_user(uid): return User.query.get(int(uid))
    with app.app_context():
        db.create_all(); _migrate_sqlite_columns()
        # Backfill legacy Match rows after an upgrade.
        for m in Match.query.filter((Match.workflow_state==None) | (Match.workflow_state=='')).all():
            if m.status=='Recommended': m.workflow_state='DOCTOR_PENDING'
            elif m.status in ('Approved','Doctor Approved'): m.workflow_state='HOSPITAL_PENDING'
            elif 'Rejected' in (m.status or ''): m.workflow_state='DOCTOR_REJECTED'
            else: m.workflow_state='DOCTOR_PENDING'
        db.session.commit(); admin_seed()
    return app
