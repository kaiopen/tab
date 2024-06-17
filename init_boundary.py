from typing import Any, Dict, Sequence, Tuple
import json
from pathlib import Path

from tqdm import tqdm
import torch

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.data import mask_in_range, squared_euclidean_distance, xy_to_rt
from tab import Sampler, TAB


def get_dense_points(
    linestrip: Sequence[Tuple[float, float]],  step: float = 0.01
) -> Sequence[Tuple[float, float]]:
    r''' Get dense points from a linestrip.

    ### Args:
        - linestrip: a linestrip consisted of some ordered 2D points. Its shape
            should be `(N, 2)`.
        - step: density of the dense points. It works on the X axis and Y axis
            seperately.

    ### Returns:
        - Unordered dense 2D points.

    '''
    ret = []
    for i in range(len(linestrip) - 1):
        x_a, y_a = linestrip[i]
        x_b, y_b = linestrip[i + 1]
        if x_a < x_b:
            x = x_a + step
            while x < x_b:
                ret.append((x, (x - x_a) * (y_b - y_a) / (x_b - x_a) + y_a))
                x += step
        elif x_a > x_b:
            x = x_a - step
            while x > x_b:
                ret.append((x, (x - x_a) * (y_b - y_a) / (x_b - x_a) + y_a))
                x -= step

        if y_a < y_b:
            y = y_a + step
            while y < y_b:
                ret.append(((y - y_a) * (x_b - x_a) / (y_b - y_a) + x_a, y))
                y += step
        elif y_a > y_b:
            y = y_a - step
            while y > y_b:
                ret.append(((y - y_a) * (x_b - x_a) / (y_b - y_a) + x_a, y))
                y -= step

    return ret + linestrip


def squared_distances_points_to_linestrip(
    points: TorchTensor[TorchFloat],  # (N, 2)
    linestrip: TorchTensor[TorchFloat]  # (M, 2)
) -> TorchTensor[TorchFloat]:
    r'''Squared distances from points to a linestrip.

    The distance from a point to a linestrip is the minimum distance between
    the point and the linestrip.

    All points should be represented in the 2D rectangular coordinate system.

    ### Args:
        - points: Its shape should be `(N, 2)`.
        - linestrip: Linestrip is comprised of a sequence of points. Its shape
            should be `(M, 2)` and `M >= 2`.

    ### Returns:
        - Squared distances between the points and the linestrip. Its shape is
            `(N,)`.

    '''
    a = linestrip[:-1]  # (M - 1, 2)
    b = linestrip[1:]  # (M - 1, 2)

    num_p = len(points)
    num_seg = len(a)
    distances = torch.zeros(
        (num_p, num_seg), dtype=points.dtype, device=points.device
    )  # (N, M)

    points = points.unsqueeze(1)  # (N, 1, 2)
    a = a.unsqueeze(0)  # (1, M, 2)
    b = b.unsqueeze(0)  # (1, M, 2)

    ab = b - a  # vector AB  (1, M, 2)
    ap = points - a  # vector AP  (N, M, 2)
    pb = b - points  # vector PB  (N, M, 2)

    dot = ab * ap
    dot = dot[..., 0] + dot[..., 1]  # (N, M)
    mask_1 = dot <= 0  # (N, M)
    _ap = ap[mask_1] ** 2
    distances[mask_1] = _ap[..., 0] + _ap[..., 1]

    d2 = ab ** 2
    d2 = d2[..., 0] + d2[..., 1]  # (1, M)
    mask_2 = dot >= d2  # (N, M)
    _pb = pb[mask_2] ** 2
    distances[mask_2] = _pb[..., 0] + _pb[..., 1]

    mask = torch.logical_not(torch.logical_or(mask_1, mask_2))
    a = a.expand(num_p, num_seg, 2)[mask]  # (X, 2)
    c = (dot[mask] / d2.expand_as(mask)[mask]).unsqueeze_(-1) \
        * (b.expand(num_p, num_seg, 2)[mask] - a) \
        + a  # (X, 2)
    cp = (points.expand(num_p, num_seg, 2)[mask] - c) ** 2
    distances[mask] = cp[..., 0] + cp[..., 1]
    return torch.min(distances, dim=1)[0]


def get_attributes(
    points: TorchTensor[TorchFloat], comxs: Sequence[Dict[str, Any]]
):
    n = len(points)
    distances = torch.ones(n, dtype=points.dtype)
    indices = torch.ones(n, dtype=torch.long) * -1

    for i, comx in enumerate(comxs):
        if comx['matched']:
            continue

        linestrip = comx['linestrip']
        if 1 == len(linestrip):
            ds = squared_euclidean_distance(points, linestrip)
        else:
            ds = squared_distances_points_to_linestrip(points, linestrip)
        # Find the nearest.
        mask = ds < distances
        if torch.any(mask):
            distances[mask] = ds[mask]
            indices[mask] = i

    if torch.any(-1 == indices):
        raise Exception('There are unmatched dense points.')

    rets = []
    for i, point in zip(indices.tolist(), points.tolist()):
        comx = comxs[i]
        comx['matched'] = True

        rets.append(
            {
                'xy': point,
                'curve': comx['curve'],
                'unstructured': comx['unstructured'],
                'irregular': comx['irregular'],
                'occluded': comx['occluded'],
                'blind': comx['blind'],
                'distorted': comx['distorted'],
                'lengthened': comx['lengthened'],
                'single': comx['single'],
                'end': False
            }
        )
    return rets


fs = []
for d in Path.cwd().joinpath('boundary').glob('*-*-*-*-*-*-*'):
    for f in d.glob('*.json'):
        fs.append(f)

if len(fs) != 6350:
    print(
        f'WARNING: The number of frames is not equal to 6350, but {len(fs)}.'
    )

range_rt = list(TAB.RANGE_RHO) + list(TAB.RANGE_THETA)
sampler = Sampler()

for f in tqdm(fs):
    data = json.load(f.open('r'))
    bounds = data['boundaries']
    comxs = data['complexity']

    for comx in comxs:
        comx['matched'] = False
        comx['linestrip'] = torch.as_tensor(comx['linestrip'])

    for bound in bounds:
        linestrip = bound['linestrip']
        points = torch.as_tensor(get_dense_points(linestrip))
        points = get_attributes(
            points[mask_in_range(xy_to_rt(points), range_rt)], comxs
        )

        points[-1]['end'] = True
        points[-len(linestrip)]['end'] = True

        p = points[0]
        single = p.pop('single')
        for p in points[1:]:
            if p.pop('single') ^ single:
                raise Exception('confused number of beams.')
        bound['single'] = single

        bound['points'] = points
        bound['keypoints'] = sampler(points)

    json.dump(bounds, f.open('w'), indent=2)
