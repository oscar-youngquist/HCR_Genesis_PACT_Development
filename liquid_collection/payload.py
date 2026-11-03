"""Bounds come from the existing presets; litres are never used as volume."""
import ast
import math
from pathlib import Path

PRESETS = Path(__file__).resolve().parents[1] / 'legged_gym/scripts/liquid_payload_configs.py'
KEYS = ('rho', 'mu', 'gamma', 'offset', 'scale_x', 'scale_y', 'scale_z')


def supported_bounds(property_expansion=0., geometry_expansion=0.):
    if not 0 <= property_expansion <= .25 or not 0 <= geometry_expansion <= .05:
        raise ValueError('Expansion limits: properties 0–25%, geometry 0–5%; mounts never expand')
    presets = []
    for node in ast.parse(PRESETS.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            value = ast.literal_eval(node.value)
            if set(KEYS) <= value.keys():
                presets.append(value)
    result = {k: [min(p[k] for p in presets), max(p[k] for p in presets)] for k in KEYS}
    for i, key in enumerate(('mount_x', 'mount_y')):
        values = [p.get('mount_offset', [0, 0])[i] for p in presets]
        result[key] = [min(values), max(values)]
    for key in KEYS:
        fraction = property_expansion if key in ('rho', 'mu', 'gamma') else geometry_expansion
        result[key] = [result[key][0]*(1-fraction), result[key][1]*(1+fraction)]
    return result


def geometry(p, mount_height, lid_overlap=.007):
    scale = [p['scale_x'], p['scale_y'], p['scale_z']]
    interior = [a*b for a, b in zip((.17, .12, .185), scale)]
    size = [a-p['offset'] for a in interior]
    xy = p.get('mount_offset', [0., 0.])
    bottom = mount_height + .015*scale[2]
    top = mount_height + .20*scale[2] - lid_overlap
    center = [*xy, bottom + interior[2]/2]
    lower = [xy[0]-interior[0]/2, xy[1]-interior[1]/2, bottom]
    upper = [xy[0]+interior[0]/2, xy[1]+interior[1]/2, top]
    clearance = [center[i]-size[i]/2-lower[i] for i in range(3)]
    clearance += [upper[i]-center[i]-size[i]/2 for i in range(3)]
    return dict(scale=scale, interior=interior, fill_size=size, fill_center_B=center,
                lower_B=lower, upper_B=upper, clearance=clearance,
                outer_lower_B=[xy[0]-.10*scale[0], xy[1]-.075*scale[1], mount_height],
                outer_upper_B=[xy[0]+.10*scale[0], xy[1]+.075*scale[1], top+.015],
                mount_translation_B=[*xy, mount_height], mount_quaternion_wxyz=[1, 0, 0, 0],
                lid_z_B=top, fill_box_volume=math.prod(size))


def validate_payload(p, mount_height, particle_size=.012, safety_margin=.001, lid_overlap=.007,
                     property_expansion=0., geometry_expansion=0.):
    if particle_size <= 0 or safety_margin < 0 or not all(map(math.isfinite, (particle_size, safety_margin, mount_height, lid_overlap))):
        raise ValueError('Invalid particle size or safety margin')
    bounds = supported_bounds(property_expansion, geometry_expansion)
    values = dict(p, mount_x=p.get('mount_offset', [0, 0])[0], mount_y=p.get('mount_offset', [0, 0])[1])
    for k, (lo, hi) in bounds.items():
        if not math.isfinite(values[k]) or not lo <= values[k] <= hi:
            raise ValueError(f'{k} outside supported bounds [{lo}, {hi}]')
    g = geometry(p, mount_height, lid_overlap)
    if min(g['fill_size']) <= 0 or min(g['clearance']) < particle_size/2+safety_margin-1e-12:
        raise ValueError('Unsafe fill: particle radius plus safety margin intersects wall/bottom/overlapped lid')
    return g


def sample_payload(rng, particle_size=.012, safety_margin=.001, attempts=1000,
                   property_expansion=0., geometry_expansion=0.):
    bounds = supported_bounds(property_expansion, geometry_expansion)
    for _ in range(attempts):
        p = {k: rng.uniform(*bounds[k]) for k in KEYS}
        p['mount_offset'] = [rng.uniform(*bounds[k]) for k in ('mount_x', 'mount_y')]
        try:
            validate_payload(p, 0, particle_size, safety_margin,
                             property_expansion=property_expansion, geometry_expansion=geometry_expansion)
            return p
        except ValueError:
            pass
    raise ValueError(f'No safe payload found after {attempts} samples within preset bounds')


def box_parts(p, mount_height, lid_overlap=.007):
    """Six closed cuboids with the same inner/outer dimensions and overlapped lid."""
    sx, sy, sz = [p['scale_'+k] for k in ('x', 'y', 'z')]
    x, y = p.get('mount_offset', [0., 0.])
    lx, ly, height = .20*sx, .15*sy, .20*sz
    tx, ty, bottom = .015*sx, .015*sy, .015*sz
    z = mount_height+bottom+(height-bottom)/2
    parts = [dict(name='bottom', size=[lx, ly, bottom], pos=[x, y, mount_height+bottom/2])]
    overlap = min(.002, tx/4)
    for sign in (-1, 1):
        parts.append(dict(name=f'x{sign}', size=[tx, ly, height-bottom], pos=[x+sign*(lx-tx)/2, y, z]))
        # Overlap into the X walls, never into the nominal free cavity. Correct
        # the overlapping box mass so the total equals the nonoverlapping shell.
        parts.append(dict(name=f'y{sign}', size=[lx-2*tx+2*overlap, ty, height-bottom],
                          pos=[x, y+sign*(ly-ty)/2, z], rho=200*(lx-2*tx)/(lx-2*tx+2*overlap)))
    parts.append(dict(name='lid', size=[lx, ly, .015], pos=[x, y, mount_height+height-lid_overlap+.0075]))
    return parts
