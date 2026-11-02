"""Fetch only the GRIB messages FCN3 needs from one public GFS 0.25-degree f000 file.

Runs on the internet-connected workstation. The FCN3 variable list and the GFS
lexicon are read from the Earth2Studio wheel itself, so the subset always matches
what the pinned Earth2Studio version expects. Each message is fetched with an HTTP
Range request taken from the NOAA .idx file and checked for GRIB...7777 framing.
"""
import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path

BUCKET = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel', type=Path, required=True, help='earth2studio-*.whl')
    p.add_argument('--date', required=True, help='YYYYMMDD')
    p.add_argument('--cycle', default='00')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    wheel = zipfile.ZipFile(a.wheel)
    fcn3_src = wheel.read('earth2studio/models/px/fcn3.py').decode()
    variables = re.findall(r'"([^"]+)"', re.search(r'VARIABLES\s*=\s*\[(.*?)\]', fcn3_src, re.S).group(1))
    lexicon = dict(re.findall(r'"(\w+)":\s*"([^"]+)"', wheel.read('earth2studio/lexicon/gfs.py').decode()))

    url = f'{BUCKET}/gfs.{a.date}/{a.cycle}/atmos/gfs.t{a.cycle}z.pgrb2.0p25.f000'
    idx = urllib.request.urlopen(url + '.idx').read().decode().splitlines()
    records = [line.split(':') for line in idx]
    offsets = [int(r[1]) for r in records]
    a.output.mkdir(parents=True, exist_ok=True)
    grib = a.output / f'gfs_{a.date}T{a.cycle}_f000_fcn3.grib2'
    manifest = []
    with grib.open('wb') as out:
        for variable in variables:
            name, level = lexicon[variable].split('::')
            hits = [i for i, r in enumerate(records) if r[3] == name and r[4] == level]
            if not hits:
                raise SystemExit(f'{variable}: {lexicon[variable]} not in {url}.idx')
            i = hits[0]
            end = offsets[i + 1] - 1 if i + 1 < len(offsets) else ''
            request = urllib.request.Request(url, headers={'Range': f'bytes={offsets[i]}-{end}'})
            data = urllib.request.urlopen(request).read()
            if data[:4] != b'GRIB' or data[-4:] != b'7777':
                raise SystemExit(f'{variable}: incomplete GRIB message')
            out.write(data)
            manifest.append(dict(variable=variable, lexicon=lexicon[variable], idx_line=idx[i], bytes=len(data)))
    init = f'{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}T{a.cycle}:00:00Z'
    (a.output / 'manifest.json').write_text(json.dumps(dict(source=url, init=init, variables=variables,
                                                            messages=manifest), indent=1))
    print(f'{len(manifest)} messages, {grib.stat().st_size / 2**20:.1f} MiB -> {grib}')


if __name__ == '__main__':
    main()
