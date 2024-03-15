'''

'''


from random import seed
from random import random
import numpy as np
from numba import jit, config, int32, float64, prange
from numba.experimental import jitclass

config.DISABLE_JIT = False

pac_def = [
    ('flag', int32),      
    ('id', int32),
    ('cost', float64),
    ('nzp', float64),
    ('zc', int32),
    ('zp', float64),
    ('tau_p', float64),
    ('tau', float64),
    ('iscat',int32)
]

@jitclass(pac_def)
class pac:
  def __init__(self, flag, id, cost, nzp, zc, zp, tau_p, tau, iscat):
    self.flag = flag
    self.id = id
    self.cost = cost
    self.nzp = nzp
    self.zc = zc
    self.zp = zp
    self.tau_p = tau_p
    self.tau = tau
    self.iscat = iscat

@jit(nopython=True, cache=True)
def tauint_1D_pp(ph, nlay, z, sig_ext, Jdot):

  ph.tau = 0.0

  while (ph.tau < ph.tau_p):

    #Calculate dsz, the distance to the next vertical level
    if (ph.nzp > 0.0):
        # Packet travelling upward, find distance to upper level
      dsz = (z[ph.zc+1]-ph.zp)/ph.nzp
      zoffset = 1
    elif (ph.nzp < 0.0):
        #Packet travelling downward, find distance to lower level
      dsz = (z[ph.zc]-ph.zp)/ph.nzp
      zoffset = -1
    else:
      #Packet travelling directly in z plane
      #Return, packet does not move in z direction
      break
    
    #Calculate optical depth to level
    taucell = dsz * sig_ext[ph.zc]

    # Check if packet ends path in this layer
    if ((ph.tau + taucell) >= ph.tau_p):
      # Packet stops in this cell - move distance then exit loop
      d1 = (ph.tau_p-ph.tau)/sig_ext[ph.zc]
      ph.zp +=  d1 * ph.nzp

      # Update estimator
      Jdot[ph.zc] += d1
      #Jdot[ph.zc] += d1*ph.nzp

      ph.tau = ph.tau_p
    else:
      #Packet continues to level edge - update position, cell index and tau
      ph.zp += (dsz + 1.0e-12) * ph.nzp

      # Update estimator
      Jdot[ph.zc] += dsz
      #Jdot[ph.zc] += dsz*ph.nzp 

      ph.zc += zoffset

      ph.tau += taucell

      #Check is packet has exited the domain
      if ((ph.zc > nlay-1) or (ph.zp >= z[-1])):
        ph.flag = 1
        break
      elif((ph.zc < 0) or (ph.zp <= z[0])):
        ph.flag = -2
        break

  return

@jit(nopython=True, cache=True)
def inc_stellar(ph, nlay, z, mu_z):

  ph.cost = -mu_z
  ph.nzp = ph.cost
  ph.zc = nlay-1
  ph.zp = z[-1] - 1.0e-12

  return


@jit(nopython=True, cache=True)
def scatter_surf(ph, z):
    
    ph.cost = np.sqrt(random())
    ph.nzp = ph.cost
    ph.zp = z[0] + 1.0e-12
    ph.zc = 0

    return

@jit(nopython=True, cache=True)
def scatter(ph, g):

  if (ph.iscat == 1):
    # Isotropic scattering
    ph.cost = 2.0 * random() - 1.0
    ph.nzp = ph.cost
    return
  
  elif (ph.iscat == 2):
    # Rayleigh scattering via direct spherical coordinate sampling
    # Assumes non-polarised incident packet
    q = 4.0*random() - 2.0
    u = (-q + np.sqrt(1.0 + q**2))**(1.0/3.0)
    bmu = u - 1.0/u

  elif (ph.iscat == 3):
    # Sample from single HG function
    if (g != 0.0):
      hgg = g
      g2 = hgg**2

      bmu = ((1.0 + g2) - \
        ((1.0 - g2) / (1.0 - hgg + 2.0 * hgg * random()))**2) \
        / (2.0*hgg)
    else:
      # g = 0, Isotropic scattering
      ph.cost = 2.0 * random() - 1.0
      ph.nzp = ph.cost
      return

  # Now apply rotation if non-isotropic scattering
  # Change direction of packet given by sampled direction

  # avoid rare numerical issues with sampling bmu
  if (bmu > 1.0):
    bmu = 1.0
  elif (bmu < -1.0):
    bmu = -1.0

  # Calculate change in direction in grid reference frame
  if (bmu >= 1.0):
    # Packet directly forward scatters - no change in direction
    ph.cost = ph.cost
  elif (bmu <= -1.0):
    # Packet directly backward scatters - negative sign for cost
    ph.cost = -ph.cost
  else:
    # Packet scatters according to sampled cosine and current direction
    # Save current cosine direction of packet and calculate sine
    costp = ph.cost
    sintp = 1.0 - costp**2
    if (sintp <= 0.0):
      sintp = 0.0
    else:
      sintp = np.sqrt(sintp)

    # Randomly decide if scattered in +/- quadrant direction and find new cosine direction
    sinbt = np.sqrt(1.0 - bmu**2)
    ri1 = 2.0 * np.pi * random()
    if (ri1 > np.pi):
      cosi3 = np.cos(2.0*np.pi - ri1)
      # Calculate new cosine direction
      ph.cost = costp * bmu + sintp * sinbt * cosi3
    else: #(ri1 <= pi)
      cosi1 = np.cos(ri1)
      ph.cost = costp * bmu + sintp * sinbt * cosi1

    #Give nzp the cost value
    ph.nzp = ph.cost

  return

@jit(nopython=True, cache=True, parallel=False)
def gCMCRT_main(nlay, nwl, wl, n_cross, cross, VMR_cross, n_ray, ray, VMR_ray, g, surf_alb, nd, Iinc, mu_z, z):
  
  # We need to get sent the number of layers, wavelengths, cross sections, Rayleigh cross sections, 
  # atmospheric density, incident intensity TOA, altitude, zenith angle

  # We index from 0 starting from the bottom boundary 

  Nph = 10000

  nlev = nlay + 1

  # Find altitude difference of level edges
  dze = np.zeros(nlay)
  dze[:] = z[1:nlev] - z[0:nlay]

  # Initalise arrays
  tau = np.zeros(nlev)
  sig_ext = np.zeros(nlay)
  sig_sca = np.zeros(nlay)
  alb = np.zeros(nlay)
  g = np.zeros((nwl,nlay))
  Jdot = np.zeros((nwl,nlay))
  J_mean = np.zeros(nlay)

  # Wavelength loop
  for l in range(nwl):

    # Find the total extinction opacity from photocross sections
    for i in range(n_cross):
      sig_ext[:] += VMR_cross[i,:] * cross[i,l,:]
    
    sig_ext[:] *= nd[:]

    # Find the total rayleigh opacity from Rayleigh species cross sections
    for i in range(n_ray):
      sig_sca[:] += VMR_ray[i,:] * ray[i,l,:]

    sig_sca[:] *= nd[:]

    # Extinction = photocross + Rayleigh
    sig_ext[:] += sig_sca[:]

    # Find scattering albedo
    alb[:] = sig_sca[:]/sig_ext[:]

    # Find the rho*kap and tau grid starting from upper boundary
    tau[-1] = 0.0
    for k in range(nlay-1,-1,-1):
      tau[k] = tau[k+1] + sig_ext[k] * dze[k]

    ## Calculate direct beam for testing
    Idirr = np.zeros(nlev)
    Idirr[:] = Iinc[l] * np.exp(-tau[:]/mu_z)

    seed(l)

    e0dt =  mu_z * Iinc[l]/float(Nph)

    for n in range(Nph):

      # Initialise packet variables (janky python way)
      flag = 0
      id = l*Nph + n
      iscat = 2

      # Initialise junk packet variables (just make sure for correct type)
      cost = 0.0
      nzp = 0.0
      zc = 0
      zp = 0.0
      tau_p = 0.0
      itau = 0.0

      # Initialise photon packet with initial values
      ph = pac(flag, id, cost, nzp, zc, zp, tau_p, itau, iscat)

      # Place photon at top of atmosphere with negative zenith angle
      inc_stellar(ph, nlay, z, mu_z)

      # Scattering loop
      while(ph.flag == 0):
        
        # Sample a random optical depth of the photon
        ph.tau_p = -np.log(random())

        # Move the photon through the grid given by the sampled tau
        tauint_1D_pp(ph, nlay, z, sig_ext, Jdot[l,:])

        # Check status of the photon after moving
        if (ph.flag == 1):
          # Exited top (1), exit loop
          break
        elif (ph.flag == -2):
          # Photon hit surface (-2), test surface albedo 
          if (random() < surf_alb[l]):
            # Lambertian scatter
            scatter_surf(ph, z)
          else:
            # Photon was absorbed by surface, exit loop
            break

        # Photon still in grid, test atmospheric albedo
        if (random() < alb[ph.zc]):
          # Packet get scattered into new direction
          scatter(ph, g[l,ph.zc])
        else:
          # Packet was absorbed, exit loop
          break

    # Scale estimator to dze
    Jdot[l,:] = (e0dt*Jdot[l,:])/dze[:]


  # After all loops, integrate to find k for each cross section species and find integrated mean intensity
  #for k in range(nlay):
  #  J_mean[k] = np.trapz(Jdot[:,l], wl[:]) 
  J_mean[:] = Jdot[0,:]

  return J_mean, Jdot, Idirr

## Mini testing code ##

nlay = 100
nlev = nlay + 1
nwl = 1
n_cross = 1
cross = np.zeros((n_cross,nwl,nlay))
VMR_cross = np.zeros((n_cross, nlay))
n_ray = 1
ray = np.zeros((n_ray,nwl,nlay))
VMR_ray = np.zeros((n_ray, nlay))
g = np.zeros((nwl,nlay))
rho = np.zeros(nlay)
nd = np.zeros(nlay)
Iinc = np.zeros(nwl)
wl = np.zeros(nwl)
surf_alb = np.zeros(nwl)


mu_z = 0.577
k_v = 4e-3
p_up = 1e-9
p_bot = 1000.0

pe = np.logspace(np.log10(p_bot),np.log10(p_up),nlev) * 1e6
dpe = np.zeros(nlay)
pl = np.zeros(nlay)

dpe[:] = pe[1:nlev] - pe[0:nlay]

for i in range(nlay):
  pl[i] = dpe[i] / np.log(pe[i+1]/pe[i])

sb_c = 5.670374419e-5
kb = 1.380649e-16
Tirr = 1000.0
Iinc[:] = sb_c * Tirr**4

grav_const = 1000.0
Rd_gas = 3.5568e7

Tl = np.zeros(nlay)
Tl[:] = 1000.0
z = np.zeros(nlev)

rho[:] = pl[:]/(Rd_gas * Tl[:])
nd[:] = pl[:]/(kb * Tl[:])

VMR_cross[:,:] = 1.0
cross[0,0,:] = k_v * rho[:]/ nd[:]

VMR_ray[:,:] = 1.0
ray[0,0,:] = 0.0 #*cross[0,0,:]

z[0] = 0.0
for i in range(nlay):
  z[i+1] = z[i] + Rd_gas/grav_const * Tl[i] * np.log(pe[i]/pe[i+1])

g[:,:] = 0.0

J_mean_1 = np.zeros(nlay)
Jdot = np.zeros((nwl, nlay))

J_mean_1, Jdot, Idirr = gCMCRT_main(nlay, nwl, wl, n_cross, cross, VMR_cross, n_ray, ray, VMR_ray, g, surf_alb, nd, Iinc, mu_z, z)


VMR_ray[:,:] = 1.0
ray[0,0,:] = 0.5*cross[0,0,:]
cross[0,0,:] = cross[0,0,:] - ray[0,0,:]

J_mean_2 = np.zeros(nlay)
J_mean_2, Jdot, Idirr = gCMCRT_main(nlay, nwl, wl, n_cross, cross, VMR_cross, n_ray, ray, VMR_ray, g, surf_alb, nd, Iinc, mu_z, z)


import matplotlib.pylab as plt

fig = plt.figure()

plt.plot(Idirr, pe/1e6, label='Direct Beam',ls='dashed',c='black')

plt.plot(abs(J_mean_1[:]), pl/1e6, label='mean_1')
plt.plot(abs(J_mean_2[:]), pl/1e6, label='mean_2')

plt.legend()

plt.yscale('log')
plt.xscale('log')

#plt.xlim(1e3,1e8)

plt.gca().invert_yaxis()

plt.show()