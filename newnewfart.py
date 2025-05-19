import math, random, pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# ---------------------------------------------------------------------------
#  Global parameters (kept “easy escape” settings)
# ---------------------------------------------------------------------------
SOFT = 1e-8         # almost-none softening
EPS  = 1e-8         # declare unbound as soon as E_tot > EPS
ESC_R = 4.0         # main escape sphere
FAR_R = 6.0         # absolute “gone” radius

# ---------------------------------------------------------------------------
class Body:
    def __init__(self, x, y, m, vx, vy, r=0.1, col='k'):
        self.x, self.y, self.mass = x, y, m
        self.vx, self.vy = vx, vy
        self.radius, self.color = r, col
        self.orbit   = [(x, y)]
        self.velhist = [(vx, vy)]
    def reset(self):
        self.orbit = [(self.x, self.y)]
        self.velhist = [(self.vx, self.vy)]

# ---------------------------------------------------------------------------
def force(b1, b2, G=1.0):
    dx, dy = b2.x-b1.x, b2.y-b1.y
    d2 = dx*dx + dy*dy + SOFT
    f  = G * b1.mass * b2.mass / (d2 * math.sqrt(d2))
    return f*dx, f*dy

def recenter(bs):
    M   = sum(b.mass for b in bs)
    xcm = sum(b.x*b.mass for b in bs) / M
    ycm = sum(b.y*b.mass for b in bs) / M
    vxcm= sum(b.vx*b.mass for b in bs) / M
    vycm= sum(b.vy*b.mass for b in bs) / M
    for b in bs:
        b.x-=xcm; b.y-=ycm; b.vx-=vxcm; b.vy-=vycm

# ---------------------------------------------------------------------------
def verlet(bs, dt, N, recenter_every=0, store_each=1):
    for b in bs:
        b.reset()

    acc = {b:[0,0] for b in bs}
    for i,b1 in enumerate(bs):
        for b2 in bs[i+1:]:
            fx,fy = force(b1,b2)
            acc[b1][0]+=fx/b1.mass; acc[b1][1]+=fy/b1.mass
            acc[b2][0]-=fx/b2.mass; acc[b2][1]-=fy/b2.mass

    for step in range(N):
        for b in bs:
            b.vx+=0.5*acc[b][0]*dt; b.vy+=0.5*acc[b][1]*dt
            b.x +=b.vx*dt;          b.y +=b.vy*dt

        nacc = {b:[0,0] for b in bs}
        for i,b1 in enumerate(bs):
            for b2 in bs[i+1:]:
                fx,fy = force(b1,b2)
                nacc[b1][0]+=fx/b1.mass; nacc[b1][1]+=fy/b1.mass
                nacc[b2][0]-=fx/b2.mass; nacc[b2][1]-=fy/b2.mass
        for b in bs:
            b.vx+=0.5*nacc[b][0]*dt; b.vy+=0.5*nacc[b][1]*dt
        acc=nacc

        if recenter_every and (step+1)%recenter_every==0:
            recenter(bs)
        if step%store_each==0:
            for b in bs:
                b.orbit.append((b.x,b.y))
                b.velhist.append((b.vx,b.vy))

# ---------------------------------------------------------------------------
def Etot(bs,t):
    E = 0.0
    for b in bs:
        vx,vy = b.velhist[t]
        E += 0.5*b.mass*(vx*vx+vy*vy)
    for i,bi in enumerate(bs):
        xi,yi = bi.orbit[t]
        for bj in bs[i+1:]:
            xj,yj = bj.orbit[t]
            r = math.sqrt((xi-xj)**2+(yi-yj)**2 + SOFT)
            E -= bi.mass*bj.mass / r
    return E

def is_div(bs,t):
    if Etot(bs,t) > EPS:
        return True
    for b in bs:
        x,y = b.orbit[t]; vx,vy = b.velhist[t]
        r2  = x*x + y*y
        if r2 > ESC_R**2 and x*vx + y*vy > 0:  # still heading outward
            return True
        if r2 > FAR_R**2:                      # definitely far away
            return True
    return False

def collide(bs,t):
    rsum = 2*bs[0].radius
    for i,b1 in enumerate(bs):
        for b2 in bs[i+1:]:
            x1,y1 = b1.orbit[t]; x2,y2 = b2.orbit[t]
            if (x1-x2)**2 + (y1-y2)**2 < rsum*rsum:
                return True
    return False

# ---------------------------------------------------------------------------
def classify(bs):
    n = len(bs[0].orbit)
    for t in range(n):
        if is_div(bs,t):
            return 'divergence'
        if collide(bs,t):
            return 'convergence'
    return 'stable'

# ---------------------------------------------------------------------------
def make_dataset(nC=50, nD=50, dt=0.0001, out='dataset.txt'):
    want = {'convergence': nC, 'divergence': nD}
    have = {k: 0 for k in want}
    rows = []

    # -------- narrow figure-eight window (your requested ranges) -----------
    m1_min, m1_max = 0.800, 0.820
    m2_min, m2_max = 0.751, 0.760
    m3_val         = 1.000
    x1_min, x1_max = -0.1385, -0.1335
    v1_min, v1_max =  2.4800,  2.5200
    v2_min, v2_max = -1.3160, 1.3320
    T_min,  T_max  =  5.16008719949432, 5.23790395158023
    # ----------------------------------------------------------------------

    while any(have[k] < want[k] for k in want):
        # sample initial conditions in the narrow window --------------------
        m1 = random.uniform(m1_min, m1_max)
        m2 = random.uniform(m2_min, m2_max)
        m3 = m3_val
        x1 = random.uniform(x1_min, x1_max)
        v1 = random.uniform(v1_min, v1_max)
        v2 = random.uniform(v2_min, v2_max)
        T  = random.uniform(T_min, T_max)

        steps = int(10 * T / dt)      # integrate 10 nominal periods

        b1 = Body(x1, 0, m1, 0, v1, 0.1, 'r')
        b2 = Body(1,  0, m2, 0, v2, 0.1, 'g')
        vy3 = -(m1*v1 + m2*v2) / m3
        b3 = Body(0,  0, m3, 0, vy3, 0.1, 'b')
        bodies = [b1, b2, b3]
        recenter(bodies)

        verlet(bodies, dt, steps, recenter_every=0, store_each=100)

        lab = classify(bodies)
        if have.get(lab,0) < want.get(lab,0):
            rows.append({
                'm1': f'{m1:.3f}', 'm2': f'{m2:.3f}', 'm3': f'{m3:.3f}', 'x1':f'{x1:.14e}',
                'v1': f'{v1:.14e}', 'v2': f'{v2:.14e}', 'T': f'{T:.14e}',
                'stability': 'C' if lab == 'convergence' else 'D'
            })
            have[lab] += 1
            print(lab, have[lab], '/', want[lab])

    pd.DataFrame(rows).to_csv(out, sep=' ', index=False)
    print('dataset written:', out)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    make_dataset(nC=100, nD=100)
