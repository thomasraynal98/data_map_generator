import math
from scipy.stats import norm

def generate_gaussian_grid_map(ox, oy, xyreso, std):
    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso)

    gmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        for iy in range(yw):

            x = ix * xyreso + minx
            y = iy * xyreso + miny

            # Search minimum distance
            mindis = float("inf")
            for (iox, ioy) in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if mindis >= d:
                    mindis = d

            pdf = (1.0 - norm.cdf(mindis, 0.0, std))
            gmap[ix][iy] = pdf
    return gmap, minx, maxx, miny, maxy

def calc_grid_map_config(ox, oy, xyreso):
    # minx = round(min(ox)) # - EXTEND_AREA / 2.0)
    # miny = round(min(oy)) # - EXTEND_AREA / 2.0)
    # maxx = round(max(ox)) # + EXTEND_AREA / 2.0)
    # maxy = round(max(oy)) # + EXTEND_AREA / 2.0)
    minx = 0
    miny = 0
    maxx = 20
    maxy = 20
    xw = int(round((maxx - minx) / xyreso))
    yw = int(round((maxy - miny) / xyreso))

    return minx, miny, maxx, maxy, xw, yw
