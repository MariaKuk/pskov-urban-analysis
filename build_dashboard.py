"""Render one self-contained HTML from completed Pskov results; no downloads or modelling."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/pskov'

METRICS = [
    ('density_fsi', 'FSI · интенсивность застройки', 'м²/м²', 'Площадь всех этажей / площадь квартала. Чем выше, тем больше площади зданий приходится на единицу территории.', '7.2', 'fsi'),
    ('density_gsi', 'GSI · покрытие зданиями', 'доля', 'Площадь пятен зданий / площадь квартала; в BlocksNet ограничено диапазоном 0–1.', '7.2', 'gsi'),
    ('density_mxi', 'MXI · жилая функция', 'доля', 'Жилая площадь / площадь всех этажей. Пропуск означает, что отношение нельзя определить.', '7.2', 'mxi'),
    ('density_l', 'L · средняя этажность', 'этажи', 'Площадь всех этажей / площадь пятен зданий. Этажность взвешена площадью пятен; часть этажности восстановлена.', '7.2', 'l'),
    ('density_osr', 'OSR · открытое пространство', 'м²/м²', '(Площадь квартала − площадь пятен зданий) / площадь всех этажей. Это не площадь благоустроенных общественных пространств.', '7.2', 'osr'),
    ('density_share_living', 'Жилая площадь / пятно зданий', 'м²/м²', 'Жилая площадь / площадь пятен зданий. Значение может превышать 1; это не процент.', '7.2', 'share_living'),
    ('density_share_non_living', 'Нежилая площадь / пятно зданий', 'м²/м²', 'Нежилая площадь / площадь пятен зданий. Значение может превышать 1; это не процент.', '7.2', 'share_non_living'),
    ('building_population', 'Население · модельная оценка', 'чел.', 'Оценка из жилой площади при 30 м² на человека. Не данные переписи. Неизвестная этажность принята равной одному этажу.', '4', 'population'),
    ('school_capacity', 'Школьные места · с оценками', 'мест', 'Сумма вместимостей школьных объектов. У 40 из 41 объекта нет исходной вместимости: использовано по 250 мест.', '5', 'capacity'),
    ('provision_provision_strong', 'Обеспеченность · в пределах порога', '% спроса', 'Обеспеченный спрос / модельный спрос. Порог 15 минут в автомобильном сценарии с условным подходом. Только транспортная область; при нулевом спросе значение отсутствует.', '7.1', 'provision_strong'),
    ('provision_provision_weak', 'Обеспеченность · расширенная', '% спроса', 'Обеспеченный спрос с учётом разрешённых дополнительных зон / спрос. При max_depth=1 дополнительных зон нет, поэтому показатель совпадает с предыдущим.', '7.1', 'provision_weak'),
    ('area_accessibility', 'Доступность территории', 'мин', 'Сумма времени до кварталов, взвешенная их площадью, / площадь транспортной области. Меньше — быстрее в заданном сценарии. Вне покрытия графа показатель неизвестен.', '7.3', 'area_accessibility'),
    ('land_use', 'Землепользование по OSM', 'категория', 'Преобладающий сопоставленный тег landuse по правилам исходного примера. Это не официальные ПЗЗ; неизвестные классы не выдумываются.', '2', 'land_use'),
    ('spacematrix_morphotype', 'Морфотип · Spacematrix', 'категория', 'KMeans по стандартизированным FSI, L и MXI, 11 кластеров. Название описывает медианные параметры кластера; это не рейтинг качества.', '7.2', 'spacematrix_morphotype'),
    ('strelka_morphotype', 'Морфотип · «Стрелка»', 'категория', 'Пороговая классификация по L, FSI и MXI. Названия типов не подтверждают исторический период постройки.', '7.2', 'strelka_morphotype'),
    ('snap_distance_m', 'Расстояние до узла графа', 'м', 'Прямая от внутренней представительной точки квартала до ближайшего узла. Препятствия на подходе не учитываются. Порог включения: 1000 м и основная сильно связная компонента.', '3', None),
    ('transport_covered', 'Покрытие транспортным расчётом', 'категория', '781 квартал включён в транспортную область, 152 исключены из транспортных оценок. Показатели застройки рассчитаны для всей городской территории.', '3', None),
    ('reachable_area_share', 'Доля достижимой площади', '% площади', 'Доля достижимой площади внутри выбранной транспортной области. В основной сильно связной компоненте равна 100%; это не покрытие всего города.', '7.3', None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dashboard.html')
    args = parser.parse_args()
    report = json.loads((RESULTS/'audit/quality-report.json').read_text())
    if report['status'] != 'PASS_WITH_LIMITATIONS':
        raise SystemExit('Results need a successful audit before rendering.')
    for relative, expected in report.get('input_sha256', {}).items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise SystemExit('Results changed after audit. Run tools/audit_pskov_results.py again.')
    if not report.get('input_sha256'):
        raise SystemExit('Audit lacks input hashes. Run tools/audit_pskov_results.py again.')
    geojson = json.loads((RESULTS/'blocks_metrics.geojson').read_text())
    required = {m[0] for m in METRICS}
    if not all(required <= set(f['properties']) for f in geojson['features']):
        raise SystemExit('Missing required metric fields in the saved GeoJSON.')
    payload = {
        'blocks': geojson,
        'boundary': json.loads((ROOT/'data/pskov/raw/boundary.geojson').read_text()),
        'schools': json.loads((ROOT/'data/pskov/processed/schools.geojson').read_text()),
        'summary': json.loads((RESULTS/'dashboard_summary.json').read_text()),
        'manifest': json.loads((RESULTS/'manifest.json').read_text()),
        'metrics': [dict(key=k,label=l,unit=u,description=d,section=s,metric=m,
                         categorical=u=='категория',percent=u.startswith('%')) for k,l,u,d,s,m in METRICS],
    }
    payload['summary']['audit_checks'] = len(report['checks'])
    # Avoid local machine identifiers in the portable page.
    payload['manifest'].pop('cbc_executable', None)
    template = (ROOT/'assets/dashboard.template.html').read_text()
    replacements = {
        '__LEAFLET_LICENSE__': (ROOT/'assets/vendor/leaflet.LICENSE').read_text().replace('--', '—'),
        '__LEAFLET_CSS__': (ROOT/'assets/vendor/leaflet.css').read_text(),
        '__LEAFLET_JS__': (ROOT/'assets/vendor/leaflet.js').read_text(),
        '__PAYLOAD__': json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('</', '<\\/'),
    }
    for token,value in replacements.items():
        template=template.replace(token,value)
    args.output.write_text(template,encoding='utf-8')
    mapping=[dict(metric=m,notebook_section=s,dashboard_control=l,geojson_field=k,
                  dashboard='dashboard.html: map / distribution / table / selected block')
             for k,l,u,d,s,m in METRICS if m]
    with (RESULTS/'dashboard_metric_coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(mapping[0]));writer.writeheader();writer.writerows(mapping)
    build_info={'output':args.output.name,'bytes':args.output.stat().st_size,
                'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'data_sha256':hashlib.sha256((RESULTS/'blocks_metrics.geojson').read_bytes()).hexdigest(),
                'primary_metrics':len(mapping),'features':len(geojson['features']),
                'downloads':False,'analytical_models_recomputed':False}
    (RESULTS/'dashboard-build.json').write_text(json.dumps(build_info,ensure_ascii=False,indent=2))
    print(json.dumps(build_info,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
