"""Read-only audit of saved results; write a separate audit and display summary."""
from pathlib import Path
import ast
import hashlib
import json
import math
import platform
from datetime import datetime, timezone

import geopandas as gpd
import networkx as nx
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/pskov'
DATA = ROOT / 'data/pskov/processed'
AUDIT = OUT / 'audit'
AUDIT.mkdir(exist_ok=True)
manifest = json.loads((OUT / 'manifest.json').read_text())
checks = []


def check(name, condition, detail=''):
    checks.append(dict(name=name, passed=bool(condition), detail=str(detail)))


def table(name):
    return pd.read_csv(OUT / f'{name}.csv', index_col='block_id')


def geo(name):
    frame = gpd.read_file(DATA / f'{name}.geojson').set_index('row_id')
    frame.index.name = None
    return frame


def close(a, b):
    return np.allclose(a, b, rtol=1e-6, atol=1e-5, equal_nan=True)


metrics = table('blocks_metrics')
density = table('density')
provision = table('school_provision')
accessibility = table('area_accessibility')
buildings = geo('buildings')
schools = geo('schools')
snaps = geo('graph_snaps')
blocks = geo('blocks').to_crs(manifest['crs'])
with np.load(DATA / 'accessibility_matrix.npz', allow_pickle=False) as bundle:
    matrix, ids = bundle['minutes'], bundle['block_ids']

check('Unique block IDs', metrics.index.is_unique and metrics.index.equals(density.index))
check('Matrix IDs equal declared transport scope', ids.tolist() == manifest['transport_scope']['block_ids'])
check('Unknown transport values outside coverage', metrics.loc[~metrics.transport_covered, 'area_accessibility'].isna().all())
check('Finite nonnegative transport matrix', np.isfinite(matrix).all() and (matrix >= 0).all())
check('Zero diagonal', (matrix.diagonal() == 0).all())
check('Area conversion / CRS', close(blocks.area, density.site_area))
check('FSI formula', close(density.fsi, density.build_floor_area / density.site_area))
check('GSI formula', close(density.gsi, (density.footprint_area / density.site_area).clip(0, 1)))
check('MXI formula', close(density.mxi, (density.living_area / density.build_floor_area).clip(0, 1)))
check('L formula', close(density.l, (density.build_floor_area / density.footprint_area).clip(lower=1)))
check('OSR formula', close(density.osr, ((density.site_area-density.footprint_area)/density.build_floor_area).replace([np.inf,-np.inf],np.nan)))
check('Living-area ratio', close(density.share_living, density.living_area/density.footprint_area))
check('Nonliving-area ratio', close(density.share_non_living, density.non_living_area/density.footprint_area))

population_expected = np.where(buildings.is_living, np.maximum(1, np.floor(buildings.living_area/30)), 0)
check('Population reconstruction', close(buildings.population, population_expected))
points = buildings[['geometry']].to_crs(manifest['crs'])
points.geometry = points.representative_point()
joined = points.sjoin(blocks[['geometry']], predicate='intersects')
check('No duplicate building assignment', not joined.index.duplicated().any())
check('No exact duplicate building geometries', not buildings.geometry.to_wkb().duplicated().any())
population_by_block = buildings.loc[joined.index, 'population'].groupby(joined.index_right.to_numpy()).sum().reindex(metrics.index, fill_value=0)
check('Independent population aggregation', close(population_by_block, metrics.building_population))
check('Imputed school capacity = 250', (schools.loc[schools.capacity_imputed, 'capacity'] == 250).all())
check('School capacity preserved in aggregation', close(schools.capacity.sum(), metrics.school_capacity.sum()))

demand_expected = metrics.loc[ids, 'building_population'].map(lambda x: math.ceil(x/1000*120))
check('Demand rounding', close(provision.demand, demand_expected))
check('Demand balance', close(provision.demand, provision.demand_left+provision.demand_within+provision.demand_without))
check('Capacity balance', close(provision.capacity.sum()-provision.capacity_left.sum(), provision.demand_within.sum()+provision.demand_without.sum()))
check('Provision ratio', close(provision.provision_strong, provision.demand_within/provision.demand))
check('Zero-demand provision remains missing', provision.loc[provision.demand==0, 'provision_strong'].isna().all())

areas = density.loc[ids, 'site_area'].to_numpy()
area_expected = np.array([math.fsum(float(t)*float(a) for t,a in zip(row,areas))/math.fsum(areas) for row in matrix])
check('Area-weighted time (independent math.fsum)', close(area_expected, accessibility.loc[ids, 'area_accessibility']))
check('Reachable-only time equals full time in connected scope', close(area_expected, accessibility.loc[ids, 'reachable_only_time_min']))
check('Reachable share = 1 inside main component', close(accessibility.reachable_area_share, 1))

# Independently reconstruct edge weights from raw GraphML and verify sample routes.
raw_graph = nx.read_graphml(ROOT / 'data/pskov/raw/pskov-440.graphml')
network = nx.DiGraph()
network.add_nodes_from(raw_graph.nodes)
speeds = manifest['assumptions']['speed_kmh']
for u,v,d in raw_graph.edges(data=True):
    types = d.get('highway','unknown')
    if isinstance(types,str) and types.startswith('['):
        types=ast.literal_eval(types)
    types=types if isinstance(types,list) else [types]
    speed=min(speeds.get(t,30) for t in types)
    weight=float(d['length'])*60/(1000*speed)
    if not network.has_edge(u,v) or weight<network[u][v]['minutes']:
        network.add_edge(u,v,minutes=weight)
route_errors=[]
rng=np.random.default_rng(42)
for i,j in rng.integers(0,len(ids),size=(24,2)):
    if i==j: continue
    a,b=snaps.loc[ids[i]],snaps.loc[ids[j]]
    expected=nx.shortest_path_length(network,str(a.graph_node),str(b.graph_node),weight='minutes')
    expected+=(float(a.snap_distance_m)+float(b.snap_distance_m))/(5*1000/60)
    route_errors.append(abs(expected-float(matrix[i,j])))
check('24 sampled routes vs independent NetworkX', max(route_errors)<1e-4, f'max abs error={max(route_errors):.8f} min')

# Verify KMeans results without BLAS matrix multiplication: labels must select nearest centroids.
space=table('spacematrix')
features=space[['fsi','l','mxi']].to_numpy()
scale=features.std(axis=0);scale[scale==0]=1
z=(features-features.mean(axis=0))/scale
labels=space.cluster.to_numpy(dtype=int)
clusters=sorted(set(labels))
centers=np.array([z[labels==k].mean(axis=0) for k in clusters])
distances=((z[:,None,:]-centers[None,:,:])**2).sum(axis=2)
nearest=np.array(clusters)[distances.argmin(axis=1)]
check('KMeans assignments vs explicit squared distances', np.array_equal(labels,nearest), f'mismatches={np.count_nonzero(labels!=nearest)}')

def strelka(row):
    l,f,m=row.l,row.fsi,row.mxi
    if l>=9:
        return 'high-rise non-residential' if m<.1 else 'high-rise soviet microdistrict' if f<=1.5 else 'high-rise modern microdistrict'
    if l>=4:
        if m<.2:return 'mid-rise non-residential'
        if m<.45:return 'mid-rise microdistrict' if f<=.8 else 'mid-rise block'
        return 'central model' if m>=.6 and f>1.5 else 'mid-rise'
    return 'low-rise non-residential' if m<.05 else 'individual residential' if f<=.1 else 'low-rise model'

strelka_table=table('strelka')
check('Strelka rules', (strelka_table.apply(strelka,axis=1)==strelka_table.morphotype).all())
notebook=nbformat.read(ROOT/'examples/pskov.ipynb',as_version=4)
code_cells=[c for c in notebook.cells if c.cell_type=='code']
check('Every notebook code cell executed without errors', all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in code_cells))
check('Saved method output tables complete', len(pd.read_csv(OUT/'metric_coverage.csv'))==15)
for relative,expected in manifest['outputs'].items():
    check('Output SHA256: '+relative,hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected)

# Display summaries are calculated once here, not in the HTML rendering script.
summary={
    'blocks':len(metrics),'area_km2':float(density.site_area.sum()/1e6),
    'city_area_km2':float(gpd.read_file(ROOT/'data/pskov/raw/boundary.geojson').to_crs(manifest['crs']).area.sum()/1e6),
    'population_estimate':int(metrics.building_population.sum()),
    'school_objects':len(schools),'school_places':int(metrics.school_capacity.sum()),
    'transport_blocks':len(ids),'transport_excluded':len(metrics)-len(ids),
    'transport_area_km2':float(areas.sum()/1e6),
    'transport_population_estimate':int(metrics.loc[ids,'building_population'].sum()),
    'transport_places':int(provision.capacity.sum()),'demand':int(provision.demand.sum()),
    'demand_served':int(provision.demand_within.sum()),
    'provision_fraction':float(provision.demand_within.sum()/provision.demand.sum()),
    'mean_area_accessibility_min':float(accessibility.area_accessibility.mean()),
    'median_area_accessibility_min':float(accessibility.area_accessibility.median()),
    'imputed_school_capacity_fraction':float(schools.capacity_imputed.mean()),
    'imputed_floors_fraction':float(buildings.floors_imputed.mean()),
    'classified_blocks':len(space),'no_demand_in_transport':int((provision.demand==0).sum()),
    'unknown_land_use':int(metrics.land_use.isna().sum()),
    'snap_covered_median_m':float(snaps.loc[ids,'snap_distance_m'].median()),
    'snap_covered_over_300m':int((snaps.loc[ids,'snap_distance_m']>300).sum()),
    'source_date':manifest['run_at'],
}
limitations=[
    'Transport estimates apply only to 781 covered blocks in the main connected network, not the whole municipality.',
    'Speeds and straight-line access legs are assumptions; barriers, parking and congestion are not modelled.',
    'OSM buildings and school capacities have extensive imputation; no official population or capacity validation.',
    'The school demand parameter is applied to all modelled residents; specialist schools remain in the OSM-based teaching scenario.',
    'Clusters describe this dataset; no predictive validation or causal claim is made.',
    'Native numerical-library warnings were checked against explicit scalar sums, route calculations and centroid distances.',
]
report={'checked_at':datetime.now(timezone.utc).isoformat(),'executor':'same assistant, separate audit pass; not external independent review',
        'python':platform.python_version(),'status':'PASS_WITH_LIMITATIONS' if all(c['passed'] for c in checks) else 'FAIL',
        'checks':checks,'limitations':limitations,'summary':summary}
input_paths = [OUT/'manifest.json', OUT/'blocks_metrics.geojson', OUT/'blocks_metrics.csv',
               DATA/'buildings.geojson', DATA/'schools.geojson', DATA/'accessibility_matrix.npz']
report['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
(AUDIT/'quality-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
pd.DataFrame(checks).to_csv(AUDIT/'checks.csv',index=False)
if not all(c['passed'] for c in checks):
    print(json.dumps([c for c in checks if not c['passed']],ensure_ascii=False,indent=2))
    raise SystemExit('Audit failed; dashboard summary not published.')
(OUT/'dashboard_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({'status':report['status'],'checks':len(checks),'summary':summary},ensure_ascii=False,indent=2))
