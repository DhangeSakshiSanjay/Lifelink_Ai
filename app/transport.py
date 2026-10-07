"""LifeLink-AI academic transport optimisation.

The module supports two layers:
1. A deterministic A* travel-time graph that works offline.
2. Optional OpenStreetMap/OSRM road routing when ENABLE_LIVE_ROUTING=1.

The online layer is best-effort and automatically falls back to A* if unavailable.
No live traffic or medical preservation limit is claimed.
"""
import heapq, math, os, json
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from datetime import datetime, timedelta

CITY_COORDS = {
    'Pune': (18.5204,73.8567), 'Nashik': (20.0059,73.7797),
    'Mumbai': (19.0760,72.8777), 'Ahmednagar': (19.0948,74.7480),
    'Sinnar': (19.8450,73.9980), 'Shirdi': (19.7667,74.4770),
    'Thane': (19.2183,72.9781)
}
LOCATION_ALIASES = {
 'sanjivani medical center':'Pune', 'pune heart institute':'Pune',
 'nashik civil hospital':'Nashik', 'ahmednagar hospital':'Ahmednagar',
 'mumbai hospital':'Mumbai', 'sanjivani hospital':'Pune',
 'sinnar hospital':'Sinnar', 'shirdi hospital':'Shirdi', 'thane hospital':'Thane'
}
# distance_km, nominal_speed_kmph, base_traffic_multiplier
EDGES = {
 'Pune': {'Nashik': (170,80,1.00), 'Mumbai': (150,45,1.80), 'Ahmednagar': (120,60,1.15)},
 'Nashik': {'Pune': (170,80,1.00), 'Ahmednagar': (90,65,1.05), 'Mumbai': (210,80,1.00), 'Sinnar': (30,65,1.00), 'Shirdi': (90,70,1.00)},
 'Ahmednagar': {'Pune': (120,60,1.15), 'Nashik': (90,65,1.05), 'Mumbai': (260,60,1.25), 'Shirdi': (90,65,1.00)},
 'Mumbai': {'Pune': (150,45,1.80), 'Nashik': (210,80,1.00), 'Ahmednagar': (260,60,1.25), 'Thane': (35,55,1.10)},
 'Sinnar': {'Nashik': (30,65,1.00), 'Shirdi': (75,70,1.00), 'Ahmednagar': (105,65,1.00)},
 'Shirdi': {'Sinnar': (75,70,1.00), 'Ahmednagar': (90,65,1.00), 'Nashik': (90,70,1.00)},
 'Thane': {'Mumbai': (35,55,1.10)}
}
TRAFFIC_FACTORS={'LOW':1.0,'MEDIUM':1.25,'HIGH':1.50,'VERY_HIGH':1.80}
ORGAN_MAX_MINUTES={'Kidney':180,'Liver':360,'Heart':240,'Lung':240,'Pancreas':360}


def normalise_traffic(value):
    if isinstance(value,(int,float)): return max(0.1,float(value))
    return TRAFFIC_FACTORS.get(str(value or 'LOW').upper(),1.0)


def edge_time_hours(distance,speed,traffic_factor):
    return (distance/max(speed,1.0))*normalise_traffic(traffic_factor)


def city(location):
    text=(location or '').strip().lower()
    if not text: return None
    for alias,name in LOCATION_ALIASES.items():
        if alias in text: return name
    return next((n for n in CITY_COORDS if n.lower() in text),None)


def geocode_location(location):
    known=city(location)
    if known:
        return {'query':location,'city':known,'coordinates':CITY_COORDS[known],'source':'configured city/alias'}
    if os.getenv('ENABLE_GEOCODING','0')!='1': return None
    try:
        req=Request('https://nominatim.openstreetmap.org/search?format=json&limit=1&q='+quote_plus(location),headers={'User-Agent':'LifeLink-AI-BTech-Academic/1.0'})
        with urlopen(req,timeout=5) as response: data=json.loads(response.read().decode('utf-8'))
        if not data: return None
        lat,lon=float(data[0]['lat']),float(data[0]['lon'])
        nearest=min(CITY_COORDS,key=lambda n: math.hypot(CITY_COORDS[n][0]-lat,CITY_COORDS[n][1]-lon))
        return {'query':location,'city':nearest,'coordinates':(lat,lon),'source':'Nominatim; snapped to configured academic graph'}
    except Exception:
        return None


def _heuristic(node,destination):
    a,b=CITY_COORDS[node],CITY_COORDS[destination]
    dx=(a[0]-b[0])*111; dy=(a[1]-b[1])*104
    return math.hypot(dx,dy)/90.0


def _path_cost(path,traffic_factor=1.0,overrides=None):
    distance=hours=0.0; overrides=overrides or {}
    for a,b in zip(path,path[1:]):
        km,speed,base=EDGES[a][b]
        # Preserve each road's base factor, then apply scenario traffic globally.
        edge_factor=base*normalise_traffic(traffic_factor)
        if f'{a}->{b}' in overrides: edge_factor=normalise_traffic(overrides[f'{a}->{b}'])
        distance += km; hours += edge_time_hours(km,speed,edge_factor)
    return distance,hours


def alternative_routes(src,dst,traffic_factor=1.0,overrides=None):
    candidates=[]
    if dst in EDGES.get(src,{}): candidates.append([src,dst])
    for mid in EDGES:
        if mid not in (src,dst) and mid in EDGES.get(src,{}) and dst in EDGES.get(mid,{}): candidates.append([src,mid,dst])
    # Also permit two intermediate nodes so the optimizer has real alternatives.
    for a in EDGES.get(src,{}):
        for b in EDGES.get(a,{}):
            if b not in (src,dst,a) and dst in EDGES.get(b,{}): candidates.append([src,a,b,dst])
    rows=[]; seen=set()
    for path in candidates:
        key=tuple(path)
        if key in seen: continue
        seen.add(key); distance,hours=_path_cost(path,traffic_factor,overrides)
        rows.append({'nodes':path,'distance_km':round(distance,2),'travel_hours':round(hours,3),'optimal':False})
    rows=sorted(rows,key=lambda x:x['travel_hours'])
    if rows: rows[0]['optimal']=True
    return rows


def _astar(source,destination,traffic_factor=1.0,overrides=None):
    if source==destination:
        return {'distance_km':0.0,'travel_hours':0.0,'nodes':[source],'algorithm':'A* (travel-time weighted)','alternatives':[],'route_points':[list(CITY_COORDS[source])], 'provider':'Offline academic A* graph'}
    overrides=overrides or {}; queue=[(_heuristic(source,destination),0.0,source,[source],0.0)]; best={}
    while queue:
        _,cost,node,path,distance=heapq.heappop(queue)
        if node in best and best[node] <= cost: continue
        best[node]=cost
        if node==destination:
            alts=alternative_routes(source,destination,traffic_factor,overrides)
            return {'distance_km':round(distance,2),'travel_hours':round(cost,3),'nodes':path,'algorithm':'A* (travel-time weighted)','alternatives':alts,'route_points':[list(CITY_COORDS[n]) for n in path], 'provider':'Offline academic A* graph'}
        for nxt,(km,speed,base) in EDGES.get(node,{}).items():
            factor=base*normalise_traffic(traffic_factor)
            if f'{node}->{nxt}' in overrides: factor=normalise_traffic(overrides[f'{node}->{nxt}'])
            step=edge_time_hours(km,speed,factor)
            heapq.heappush(queue,(cost+step+_heuristic(nxt,destination),cost+step,nxt,path+[nxt],distance+km))
    raise ValueError('No route exists in the configured transport graph.')


def _osrm_route(source_info,destination_info):
    """Best-effort real road routing using OSRM/OpenStreetMap geometry."""
    if os.getenv('ENABLE_LIVE_ROUTING','0')!='1': return None
    a=source_info['coordinates']; b=destination_info['coordinates']
    url=f'https://router.project-osrm.org/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}?overview=full&geometries=geojson&alternatives=true&steps=false'
    try:
        req=Request(url,headers={'User-Agent':'LifeLink-AI-BTech-Academic/1.0'})
        with urlopen(req,timeout=6) as response: data=json.loads(response.read().decode('utf-8'))
        routes=data.get('routes') or []
        if not routes: return None
        primary=routes[0]
        alts=[]
        for r in routes:
            coords=r.get('geometry',{}).get('coordinates',[])
            alts.append({'nodes':[source_info['city'],destination_info['city']], 'distance_km':round(r.get('distance',0)/1000,2),'travel_hours':round(r.get('duration',0)/3600,3),'optimal':r is primary,'route_points':[[float(y),float(x)] for x,y in coords]})
        return {'distance_km':round(primary.get('distance',0)/1000,2),'travel_hours':round(primary.get('duration',0)/3600,3),'nodes':[source_info['city'],destination_info['city']], 'algorithm':'OSRM road routing (OpenStreetMap)','alternatives':alts,'route_points':[[float(y),float(x)] for x,y in primary.get('geometry',{}).get('coordinates',[])], 'provider':'OSRM/OpenStreetMap'}
    except Exception:
        return None


def find_fastest_route(source,destination,traffic_overrides=None,traffic_factor=1.0):
    src_info,dst_info=geocode_location(source),geocode_location(destination)
    src=src_info['city'] if src_info else None; dst=dst_info['city'] if dst_info else None
    if not src or not dst:
        raise ValueError('Enter a supported demo city/hospital: Pune, Nashik, Mumbai, Ahmednagar, Sinnar, Shirdi or Thane. You can enable optional geocoding with ENABLE_GEOCODING=1.')
    live=_osrm_route(src_info,dst_info)
    if live:
        # Apply academic scenario factor to the live baseline because OSRM does not provide live traffic here.
        factor=normalise_traffic(traffic_factor)
        if factor != 1.0:
            live['travel_hours']=round(live['travel_hours']*factor,3)
            for row in live['alternatives']: row['travel_hours']=round(row['travel_hours']*factor,3)
        return live
    return _astar(src,dst,traffic_factor,traffic_overrides)


def calculate_eta(travel_hours,departure=None): return (departure or datetime.utcnow())+timedelta(hours=float(travel_hours))


def check_transport_feasibility(organ,travel_hours,harvest_hours=0):
    maximum=ORGAN_MAX_MINUTES.get(organ,180); elapsed=max(0.0,float(harvest_hours or 0))*60; estimated=float(travel_hours)*60; available=max(0.0,maximum-elapsed)
    return {'maximum_minutes':maximum,'estimated_minutes':round(estimated,1),'remaining_minutes':round(available-estimated,1),'feasible':estimated<=available,'label':'TRANSPORT FEASIBLE' if estimated<=available else 'TRANSPORT NOT FEASIBLE','disclaimer':'Academic configurable parameter; not a medically validated organ-preservation limit.'}

TRANSPORT_MODES = {
    'GROUND': {'speed_factor': 1.0, 'cost_factor': 1.0, 'risk_factor': 1.0},
    'EMERGENCY_GROUND': {'speed_factor': 0.82, 'cost_factor': 1.8, 'risk_factor': 0.9},
    'AIR_AMBULANCE': {'speed_factor': 0.35, 'cost_factor': 6.0, 'risk_factor': 0.7},
}

def estimate_transport_mode(travel_hours, mode='GROUND'):
    cfg=TRANSPORT_MODES.get(str(mode).upper(), TRANSPORT_MODES['GROUND'])
    adjusted=max(0.05,float(travel_hours)*cfg['speed_factor'])
    return {'mode':mode.upper(),'travel_hours':round(adjusted,3),'cost_index':round(cfg['cost_factor'],2),'risk_index':round(cfg['risk_factor'],2)}

def compare_transport_modes(travel_hours): return sorted([estimate_transport_mode(travel_hours,m) for m in TRANSPORT_MODES],key=lambda x:x['travel_hours'])

def route_scenario(source,destination,traffic='LOW'):
    return find_fastest_route(source,destination,traffic_factor=normalise_traffic(traffic))

def calculate_transport_risk(feasibility, traffic='LOW', mode='GROUND'):
    traffic_penalty={'LOW':5,'MEDIUM':20,'HIGH':40,'VERY_HIGH':65}.get(str(traffic).upper(),20)
    mode_bonus={'AIR_AMBULANCE':-12,'EMERGENCY_GROUND':-6,'GROUND':0}.get(str(mode).upper(),0)
    max_m=max(1,float(feasibility.get('maximum_minutes') or 1)); remaining=float(feasibility.get('remaining_minutes') or 0)
    time_pressure=max(0,min(100,100*(1-max(0,remaining)/max_m)))
    score=max(0,min(100,0.55*time_pressure+0.35*traffic_penalty+10+mode_bonus))
    label='LOW' if score<30 else 'MEDIUM' if score<60 else 'HIGH'
    return {'score':round(score,1),'label':label,'disclaimer':'Academic configurable operational-risk heuristic; not a medical risk prediction.'}
