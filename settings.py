nx = 120+1
nz = 120+1
xmax = 20 # μm
# zmax = (nz/nx)*xmax
zmax = 20
dt = 5e-4 # ms
dx = 2*xmax/(nx-1)
dz = 2*zmax/(nz-1)
hb = 63.5078 #("u" ("μm")^2)/("ms")

m3 = 3   # u
m4 = 4 

pxmax= (nx-1)/2 * 2*pi/(2*xmax)*hb # want this to be greater than p
pzmax= (nz-1)/2 * 2*pi/(2*zmax)*hb

dpx = 2*pi/(2*xmax)*hb
dpz = 2*pi/(2*zmax)*hb