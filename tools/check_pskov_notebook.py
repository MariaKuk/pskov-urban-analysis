"""Offline code checks only: synthetic objects, no Pskov data or notebook execution."""
import ast
import json
import tempfile
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / 'examples/pskov.ipynb', as_version=4)
nbformat.validate(notebook)
namespace = {}
trees = []
for index, cell in enumerate(notebook.cells):
    if cell.cell_type != 'code':
        continue
    assert not any(o.output_type == 'error' for o in cell.get('outputs', [])), index
    trees.append(ast.parse(cell.source, filename=f'cell_{index}'))
imports = [node for tree in trees for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
definitions = [node for tree in trees for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
exec(compile(ast.Module(body=imports + definitions, type_ignores=[]), '<definitions only>', 'exec'), namespace)

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from unittest.mock import patch

# Neither API calls nor research-cell bodies are executed by this checker.
with tempfile.TemporaryDirectory(prefix='pskov-code-check-') as temporary, patch(
    'requests.sessions.Session.request', side_effect=AssertionError('Network forbidden in code checks')
):
    temp = Path(temporary)
    source = gpd.GeoDataFrame({'value': [np.inf, 2.0], 'tags': [['a', 'b'], None]},
                             geometry=[Point(28.3, 57.8), Point(28.31, 57.81)], crs=4326, index=[3, 9])
    namespace['save_geo'](source, temp / 'roundtrip.geojson')
    loaded = namespace['read_geo'](temp / 'roundtrip.geojson')
    assert loaded.index.tolist() == [3, 9]
    assert pd.isna(loaded.loc[3, 'value']) and loaded.loc[9, 'value'] == 2
    assert json.loads(loaded.loc[3, 'tags']) == ['a', 'b']
    namespace['save_geo'](source.iloc[:0], temp / 'empty.geojson')
    assert namespace['read_geo'](temp / 'empty.geojson').empty
    assert 'Infinity' not in (temp / 'roundtrip.geojson').read_text()

    # Detect double counting and preserve unmatched objects on tiny invented geometry.
    namespace['PROCESSED'] = temp
    namespace['blocks'] = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=32635)
    objects = gpd.GeoDataFrame({'capacity': [5, 7, 11]}, geometry=[Point(2, 2), Point(12, 2), Point(25, 2)], crs=32635)
    aggregated, unmatched = namespace['aggregate_checked'](objects, ['capacity'], 'synthetic')
    assert aggregated.capacity.tolist() == [5, 7] and unmatched.capacity.sum() == 11
    try:
        namespace['aggregate_checked'](gpd.GeoDataFrame({'capacity': [1]}, geometry=[Point(10, 2)], crs=32635), ['capacity'], 'boundary')
    except AssertionError as exc:
        assert 'два квартала' in str(exc)
    else:
        raise AssertionError('Double counting was not detected')

    # Rendering compatibility check, no basemap fetch or output research artifact.
    html = source.drop(columns='tags').assign(value=[1.0, 2.0]).explore(column='value').get_root().render()
    assert 'leaflet' in html.lower()

    # Native CBC integration: invented one-variable problem, never urban calculations.
    import shutil
    import pulp
    cbc = shutil.which('cbc')
    assert cbc, 'Native cbc must be installed before verification.'
    problem = pulp.LpProblem('synthetic_installation_check', pulp.LpMaximize)
    x = pulp.LpVariable('x', lowBound=0, cat='Integer')
    problem += x
    problem += x <= 3
    problem.solve(namespace['CheckedCBC'](path=cbc, msg=False))
    assert x.value() == 3 and problem.status == pulp.LpStatusOptimal

print(f'PASS: {len(trees)} code cells parse; saved outputs have no errors; imports, GeoJSON roundtrip, aggregation guards, map rendering, native CBC checked on synthetic objects.')
print('Pskov downloads and research calculations were NOT run.')
