################################################################
# Summary: this code calculates equilibrium models baroclinic models of galacic coronae with elliptical equipressure surfaces
# Author: Mattia Sormani
# References: Sormani et al. (2018)
# Date: 05/09/2018
# Brief: This code calculates models of the galactic corona with elliptical equipressure surfaces.
#
# With the default settings, it constructs model 6 of Sormani et al. (2018). Simply running the code by typing in the terminal
#
# >> python rotating.py
#
# should produce figures similar to those in the paper.
#
# ———————————————————————————————————————————————
# Here are a few info:
#
# - The code works by first calculating (P,rho,vphi) at points in the (R,z) plane which are distributed along concentric ellipses, and then constructing linearly interpolating functions between them. If “model” is an instance of the class “Models”, these interpolating functions can be accessed by typing models.P, model.frho, model.fvphi. For example, to print the density at the point (R,z)=(1,3) you have to type
#
# >> Print(model.frho(1,3))
#
# - The ellipses are labelled by the parameter mu, and defined parametrically through it. The relation between the label mu and the length of the minor (major) axis of the ellipses is defined by the functions “calc_b” (“calc_a”). The parameter dt controls how finely spaced you want the points along a given ellipse. The parameters mumin, mumax controls the size of the smallest and largest ellipses, while the parameter dmu controls how finely spaced you want them in the (R,z) plane. If you make dt or dmu smaller, there will be more points in the (R,z) plane to interpolate so precision will be better but the code will be slower.
# - The potential is defined by the function “Calc_Phi”. Change this function if you want to change the potential.
# - The models are defined by two quantities: P_axis(z) and q_axis(z). These quantities are in practice defined through the functions “calc_Paxis” and “calc_a”,”calc_b”. By modifying these functions, you can construct any model whose surfaces of constant pressure are ellipses.
# - The script uses the functions fP,frho,fvphi to interpolate everything on a regular grid in the (R,z) plane and produce the plots. The extent and resolution of this grid is set by the parameters Rmin,Rmax,dR,zmin,zmax,dz.
#
# Have fun!
#
################################################################

import pylab as pl
import numpy as np
import matplotlib.colors as mc
from numpy import sqrt, log, exp, arccosh, sinh, cosh, tanh, pi, cos, sin, arctan2, arccos
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
pl.rcParams['text.usetex'] = True

################################
# Define Phi
################################

def Calc_Phi(R,z):

  # NFW potential
  # [G]    = kpc (100km/s)^2/1e10Msol
  # [rho0] = 1e10Msol/kpc^3
  # [r0]   = kpc
  G = 4.33      
  rho0 = 1e-3   
  r0 = 20.0
  q = 1.0
  rtilde = sqrt(R**2 + z**2/q**2)
  A = 4*pi*G*rho0*r0**2
  B = log(1+rtilde/r0)/(rtilde/r0)
  Phi = -A*B
  C = 1.0/(rtilde**2*(1+rtilde/r0)) - (1.0/rtilde**2)*B
  dPhidR = -A*C*R
  dPhidz = -A*C*z/(q**2)

  return Phi, dPhidR, dPhidz

################################
# Define a(mu), b(mu).
# These determine q_axis(z)
################################

def calc_a(mu):
  a = mu
  dadmu = np.ones(a.shape)
  return a, dadmu

def calc_b(mu):
  L = 20.0
  b = mu * (1.0 - exp(-mu/L))
  dbdmu = np.ones(b.shape) + ((mu/L) - 1.0)*exp(-mu/L)
  return b, dbdmu

#################################
# Define P_axis
#################################

def calc_Paxis(mu):

  # polytropic
  gamma = 5.0/3.0
  P0 = 6e1
  alpha = gamma/(gamma-1.0)
  beta = 1.0/(gamma-1.0)
  C = -3.8
  z, dzdmu = calc_b(mu)
  Phi, dPhidR, dPhidz = Calc_Phi(0,z)
  Paxis = P0*(-Phi+C)**alpha
  dPaxisdmu = P0*alpha*(-Phi+C)**beta*(-dPhidz)*dzdmu

  return Paxis, dPaxisdmu

################################
# define class to manage ellipses
################################

class Ellipse:

    def __init__(self,t,a,dadmu,b,dbdmu,P,dPdmu):
        self.t = t
        self.a = a
        self.b = b
        self.P = P
        self.q = b/a
        self.dadmu = dadmu
        self.dbdmu = dbdmu 
        self.dPdmu = dPdmu
        self.x, self.y = self.calc_xy()
        self.etx, self.ety, self.epx, self.epy = self.calc_et_and_ep()
        self.gradP = self.calc_gradP()

    def calc_xy(self):
      # x and y
      return self.a*cos(t), self.b*sin(t)

    def calc_et_and_ep(self):
      # et = unit vector tangent to ellipse in direction of increasing theta
      # ep = unit vector perpendicular to ellipse directed outwards
      A = sqrt(self.a**2*sin(self.t)**2 + self.b**2*cos(self.t)**2)
      etx, ety =  -self.a*sin(self.t)/A, self.b*cos(self.t)/A
      epx, epy = self.b*cos(self.t)/A, self.a*sin(self.t)/A 
      return etx,ety,epx,epy

    def calc_gradP(self):
      # calculate the distance along y to next orbit, then do y \cdot e_p. This gives the distance.
      dadmu = self.dadmu
      dbdmu = self.dbdmu
      dPdmu = self.dPdmu
      a, b = self.a, self.b
      x, y = self.x, self.y
      gradP = dPdmu * a * b * (b**4*x**2 + a**4*y**2)**(0.5) / (x**2*b**3*dadmu + y**2*a**3*dbdmu)
      return gradP

################################
# given vphi,P,rho on the ellipses, 
# create interpolating functions 
################################

def calc_interpolating_f(ellipses):

  R_pts = []
  z_pts = []

  P_pts    = []
  vphi_pts = []
  rho_pts  = []

  for el in ellipses:

      R, z = el.x, el.y
      P = el.P*np.ones(R.size)
      gradP = el.gradP
      eP_R, eP_z = el.epx, el.epy
      eT_R, eT_z = -el.etx, -el.ety
      dPdR, dPdz = gradP*eP_R, gradP*eP_z

      Phi, dPhidR, dPhidz = Calc_Phi(R,z)

      # calculate vphi
      vphi_squared = R*(dPhidR*eT_R + dPhidz*eT_z)/(eP_z)
      vphi = sqrt(vphi_squared)
      vphi[np.isnan(vphi)] = 0.0

      # calculate rho
      rho = -dPdz/dPhidz
      rho[np.isnan(rho)] = 0.0

      R_pts.append(R)
      z_pts.append(z)
      P_pts.append(P)
      vphi_pts.append(vphi)
      rho_pts.append(rho)

  R_pts = np.asarray(R_pts).flatten()
  z_pts = np.asarray(z_pts).flatten()
  P_pts = np.asarray(P_pts).flatten()
  rho_pts = np.asarray(rho_pts).flatten()
  vphi_pts = np.asarray(vphi_pts).flatten()

  # create interpolating functions
  points = np.vstack((R_pts,z_pts)).T
  frho = LinearNDInterpolator(points,rho_pts,fill_value=0.0)
  fvphi = LinearNDInterpolator(points,vphi_pts,fill_value=0.0)
  fP = LinearNDInterpolator(points,P_pts,fill_value=0.0)

  return frho, fvphi, fP

################################
# define class to manage a corona model
################################

class Model:

    def __init__(self,t,mu,R1d,z1d):

      self.t = t
      self.mu = mu
      self.R1d,  self.z1d  = R1d, z1d
      self.dR,   self.dz   = R1d[1]-R1d[0], z1d[1]-z1d[0]
      self.Rmax, self.zmax = R1d.max(), z1d.max()

      # define properties that characterise ellipses
      self.a_array, self.dadmu_array = calc_a(mu)
      self.b_array, self.dbdmu_array = calc_b(mu)
      self.P_array, self.dPdmu_array = calc_Paxis(mu)

      self.ellipses = [Ellipse(self.t,self.a_array[i],self.dadmu_array[i],self.b_array[i],self.dbdmu_array[i],self.P_array[i],self.dPdmu_array[i]) for i in range(mu.size)]

      # calculate interpolating functions
      self.frho, self.fvphi, self.fP = calc_interpolating_f(self.ellipses)
      
      #####################################
      # use interpolation on a regular grid
      # default basic units are:
      # [v] = 100km/s
      # [R] = kpc
      # [m] = Msol
      # default units of everything else are contructed by combining these, so
      # [time] = kpc/(100km/s)
      # [P]    = Msol (100km/s)^2 kpc^-3
      # [rho]  = Msol kpc^-3
      #####################################

      self.R,self.z  = np.meshgrid(R1d,z1d)
      self.rho  = self.frho(self.R,self.z)
      self.vphi = self.fvphi(self.R,self.z)
      self.P    = self.fP(self.R,self.z)

      # calculate other quantities

      self.gamma = 5.0/3.0 
      self.kb = 1.38e-16         # Boltzmann constant [g cm^2 s^-2 K^-1]
      self.mumean = 0.58         # mean molecular weight
      self.mp = 1.67e-24         # Proton mass [g]
      self.kpc_to_cm = 3.086e21  # 1 kpc in cm
      self.km100_to_cm  = 1e7    # 100 km in cm 
      self.Msol_to_g   = 2e33    # solar mass in g

      self.Phi, self.dPhidR, self.dPhidz = Calc_Phi(self.R,self.z)
      self.Omega = self.vphi/self.R
      self.L     = self.vphi*self.R
      self.Alpha = self.vphi**2/(self.R*self.dPhidR)

      self.cs_squared = self.P/self.rho                                               # sound speed squared (kT/(mu*mp)) [(100km/s)^2]
      self.sigma = log(self.P*self.rho**(-self.gamma))                                # Entropy
      self.T = self.cs_squared*((self.mumean*self.mp)/(self.kb))*self.km100_to_cm**2  # Temperature [K]

      self.Pk = self.P/self.kb * (self.Msol_to_g * self.km100_to_cm**2 / (self.kpc_to_cm)**3)   # P/k [K cm^-3] 
      self.n = self.rho * (self.Msol_to_g / (self.kpc_to_cm)**3) / (self.mumean * self.mp)      # n   [cm^-3]

      # calculate mass of the corona enclosed within spheres of radius R

      self.masses = 2*(2*pi*self.R*self.dR*self.dz)*self.rho # consider both the annuli at +/- z
      self.r_enc = np.linspace(0,300,301)
      self.M_enc = [self.masses[sqrt(self.R**2+self.z**2)<r].sum() for r in self.r_enc]

      # calculate Angular Momentum Distribution (AMD) by binning masses according to value of angular momentum

      self.bins_L = np.linspace(0,101,101)
      self.hist_L, self.bins_L = np.histogram(self.L,bins=self.bins_L,weights=self.masses)

################################
# START MAIN EVENT             #
################################

################################
# create the model by initializing an instance of the class "Model"
################################

tmin,  tmax,  dt  = 0, pi/2, 0.0015
mumin, mumax, dmu = 0, 300, 0.5
Rmin,  Rmax,  dR  = 0, 300, 0.5
zmin,  zmax,  dz  = 0, 300, 0.5

t    = np.arange(-dt/2,tmax+dt,dt)
mu   = np.arange(mumin,mumax+dmu,dmu)
R1d  = np.arange(Rmin,Rmax+dR,dR)
z1d  = np.arange(zmin,zmax+dz,dz)

model = Model(t,mu,R1d,z1d)

################################
# START PLOTTING               #
################################

######################
# plot 2D maps in (R,z) plane
######################

fig = pl.figure(figsize=(14,6))
ax1 = pl.subplot2grid((2,4), (0,0))
ax2 = pl.subplot2grid((2,4), (0,1))
ax3 = pl.subplot2grid((2,4), (0,2))
ax4 = pl.subplot2grid((2,4), (0,3))
ax5 = pl.subplot2grid((2,4), (1,0))
ax6 = pl.subplot2grid((2,4), (1,1))
ax7 = pl.subplot2grid((2,4), (1,2))
ax8 = pl.subplot2grid((2,4), (1,3))

axarr = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]

# plot vphi
levels_plot = np.linspace(0,2.0,101)
levels_cont = [0.01,0.025,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
levels_cbar = np.linspace(0.0,2.0,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[0].contourf(model.R,model.z,model.vphi,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[0].contour(model.R,model.z,model.vphi,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[0],ticks=levels_cbar,format='%g')
cb.ax.set_title(r'$[\rm 100\,km \, s^{-1}]$')
axarr[0].set_title(r'$\bf v_\phi$',fontsize=20)

# plot vphi/(R dPhi/dR)
levels_plot = np.linspace(0,1.0,101)
levels_cont = [1e-6,1e-5,1e-4,1e-3,1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
levels_cbar = np.linspace(0.0,1.0,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[1].contourf(model.R,model.z,model.Alpha,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[1].contour(model.R,model.z,model.Alpha,colors='w',levels=levels_cont)
pl.colorbar(im1,ax=axarr[1],ticks=levels_cbar,format='%g')
axarr[1].set_title(r'$\bf v_\phi^2/(R \partial{\Phi}/\partial{R})$',fontsize=20)

# plot Omega
levels_plot = np.logspace(-5,1.0,101)
levels_cont = np.logspace(-5,1.0,16)
levels_cbar = np.logspace(-5,1.0,7)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[2].contourf(model.R,model.z,model.Omega,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[2].contour(model.R,model.z,model.Omega,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[2],ticks=levels_cbar,format='%g')
cb.ax.set_title(r'$[\rm 100\,km\, s^{-1}\, kpc^{-1}]$')
axarr[2].set_title(r'$\bf \Omega$',fontsize=20)

# plot L
levels_plot = np.linspace(0,50.0,101)
levels_cont = [ 0., 1.0, 2.5, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50.]
levels_cbar = np.linspace(0.0,50.0,6)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[3].contourf(model.R,model.z,model.L,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[3].contour(model.R,model.z,model.L,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[3],ticks=levels_cbar,format='%g')
cb.ax.set_title(r'$[\rm 100\,km \, s^{-1}\, kpc]$')
axarr[3].set_title(r'$\bf l$',fontsize=20)

# plot P
levels_plot = np.logspace(0,4,101)
levels_cont = np.logspace(0,4,5)#[50,140,530] # np.logspace(0,4,11)
levels_cbar = np.logspace(0,4,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[4].contourf(model.R,model.z,model.Pk,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[4].contour(model.R,model.z,model.Pk,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[4],ticks=levels_cbar,format='%.0e')
cb.ax.set_title(r'$[\rm K \, cm^{-3}]$')
axarr[4].set_title(r'$\bf P/k$',fontsize=20)

# plot T
levels_plot = np.logspace(5,6.7,101)
levels_cont = [1e5,5e5,1e6,2e6,3e6,4e6,5e6]
levels_cbar = [1e5,5e5,1e6,2e6,3e6,4e6,5e6]
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[5].contourf(model.R,model.z,model.T,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[5].contour(model.R,model.z,model.T,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[5],ticks=levels_cbar,format='%.0e')
cb.ax.set_title(r'$[\rm K]$')
axarr[5].set_title(r'$\bf T$',fontsize=20)

# plot sigma
levels_plot = np.linspace(-15,0,101)
levels_cont = [-5,-4,-3]
levels_cbar = np.linspace(-15,0,4)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[6].contourf(model.R,model.z,model.sigma,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[6].contour(model.R,model.z,model.sigma,colors='w',levels=levels_cont)
pl.colorbar(im1,ax=axarr[6],ticks=levels_cbar,format='%g')
axarr[6].set_title(r'$\bf\sigma$',fontsize=20)

# plot n
levels_plot = np.logspace(-6,-2,101)
levels_cont = np.logspace(-6,-2,5)
levels_cbar = np.logspace(-6,-2,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[7].contourf(model.R,model.z,model.n,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[7].contour(model.R,model.z,model.n,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[7],ticks=levels_cbar,format='%.0e')
cb.ax.set_title(r'$[\rm cm^{-3}]$')
axarr[7].set_title(r'$\bf n$',fontsize=20)

# formatting
for ax in axarr:
  ax.grid(ls='dashed')
  ax.tick_params(labelsize=18)
  axis_font = {'size':'18'}
  ax.set_aspect('1')
  ax.set_xticks(np.arange(0,int(R1d.max())+1)[::50])
  ax.set_yticks(np.arange(0,int(z1d.max())+1)[::50])
  ax.set_xlim(0,200)
  ax.set_ylim(0,200)

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])

ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax6.set_yticklabels([])
ax7.set_yticklabels([])
ax8.set_yticklabels([])

ax1.set_ylabel(r'$z$',**axis_font)
ax5.set_ylabel(r'$z$',**axis_font)
ax5.set_xlabel(r'$R$',**axis_font)
ax6.set_xlabel(r'$R$',**axis_font)
ax7.set_xlabel(r'$R$',**axis_font)
ax8.set_xlabel(r'$R$',**axis_font)

pl.tight_layout()
pl.subplots_adjust(wspace=-0.1, hspace=0.2)
fig.savefig('equilibria.png',bbox_inches='tight')
pl.close()

######################
# plot 2D maps in (R,z) plane - zoom
######################

fig = pl.figure(figsize=(14,6))
ax1 = pl.subplot2grid((2,4), (0,0))
ax2 = pl.subplot2grid((2,4), (0,1))
ax3 = pl.subplot2grid((2,4), (0,2))
ax4 = pl.subplot2grid((2,4), (0,3))
ax5 = pl.subplot2grid((2,4), (1,0))
ax6 = pl.subplot2grid((2,4), (1,1))
ax7 = pl.subplot2grid((2,4), (1,2))
ax8 = pl.subplot2grid((2,4), (1,3))

axarr = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]

# plot vphi
levels_plot = np.linspace(0,2.0,101)
levels_cont = [0.01,0.025,0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
levels_cbar = np.linspace(0.0,2.0,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[0].contourf(model.R,model.z,model.vphi,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[0].contour(model.R,model.z,model.vphi,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[0],ticks=levels_cbar,format='%g')
cb.ax.set_title(r'$[\rm 100\,km \, s^{-1}]$')
axarr[0].set_title(r'$\bf v_\phi$',fontsize=20)

# plot vphi/(R dPhi/dR)
levels_plot = np.linspace(0,1.0,101)
levels_cont = [1e-6,1e-5,1e-4,1e-3,1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
levels_cbar = np.linspace(0.0,1.0,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[1].contourf(model.R,model.z,model.Alpha,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[1].contour(model.R,model.z,model.Alpha,colors='w',levels=levels_cont)
pl.colorbar(im1,ax=axarr[1],ticks=levels_cbar,format='%g')
axarr[1].set_title(r'$\bf v_\phi^2/(R \partial{\Phi}/\partial{R})$',fontsize=20)

# plot Omega
levels_plot = np.logspace(-5,1.0,101)
levels_cont = np.logspace(-5,1.0,16)
levels_cbar = np.logspace(-5,1.0,7)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[2].contourf(model.R,model.z,model.Omega,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[2].contour(model.R,model.z,model.Omega,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[2],ticks=levels_cbar,format='%g')
cb.ax.set_title(r'$[\rm 100\,km\, s^{-1}\, kpc^{-1}]$')
axarr[2].set_title(r'$\bf \Omega$',fontsize=20)

# plot L
levels_plot = np.linspace(0,50.0,101)
levels_cont = [ 0., 1.0, 2.5, 5., 10., 15., 20., 25., 30., 35., 40., 45., 50.]
levels_cbar = np.linspace(0.0,50.0,6)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[3].contourf(model.R,model.z,model.L,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[3].contour(model.R,model.z,model.L,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[3],ticks=levels_cbar,format='%g')
cb.ax.set_title(r'$[\rm 100\,km \, s^{-1}\, kpc]$')
axarr[3].set_title(r'$\bf l$',fontsize=20)

# plot P
levels_plot = np.logspace(0,4,101)
levels_cont = np.logspace(0,4,5)#[50,140,530] # np.logspace(0,4,11)
levels_cbar = np.logspace(0,4,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[4].contourf(model.R,model.z,model.Pk,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[4].contour(model.R,model.z,model.Pk,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[4],ticks=levels_cbar,format='%.0e')
cb.ax.set_title(r'$[\rm K \, cm^{-3}]$')
axarr[4].set_title(r'$\bf P/k$',fontsize=20)

# plot T
levels_plot = np.logspace(5,6.7,101)
levels_cont = [1e5,5e5,1e6,2e6,3e6,4e6,5e6]
levels_cbar = [1e5,5e5,1e6,2e6,3e6,4e6,5e6]
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[5].contourf(model.R,model.z,model.T,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[5].contour(model.R,model.z,model.T,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[5],ticks=levels_cbar,format='%.0e')
cb.ax.set_title(r'$[\rm K]$')
axarr[5].set_title(r'$\bf T$',fontsize=20)

# plot sigma
levels_plot = np.linspace(-15,0,101)
levels_cont = [-5,-4,-3]
levels_cbar = np.linspace(-15,0,4)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[6].contourf(model.R,model.z,model.sigma,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[6].contour(model.R,model.z,model.sigma,colors='w',levels=levels_cont)
pl.colorbar(im1,ax=axarr[6],ticks=levels_cbar,format='%g')
axarr[6].set_title(r'$\bf\sigma$',fontsize=20)

# plot n
levels_plot = np.logspace(-6,-2,101)
levels_cont = np.logspace(-6,-2,5)
levels_cbar = np.logspace(-6,-2,5)
norm = mc.BoundaryNorm(levels_plot, 256)
im1 = axarr[7].contourf(model.R,model.z,model.n,cmap='viridis',levels=levels_plot,norm=norm)
im2 = axarr[7].contour(model.R,model.z,model.n,colors='w',levels=levels_cont)
cb = pl.colorbar(im1,ax=axarr[7],ticks=levels_cbar,format='%.0e')
cb.ax.set_title(r'$[\rm cm^{-3}]$')
axarr[7].set_title(r'$\bf n$',fontsize=20)

# formatting
for ax in axarr:
  ax.grid(ls='dashed')
  ax.tick_params(labelsize=18)
  axis_font = {'size':'18'}
  ax.set_aspect('1')
  ax.set_xlim(0,30)
  ax.set_ylim(0,30)
  ax.set_xticks(np.linspace(0,30,4))
  ax.set_yticks(np.linspace(0,30,4))

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])

ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax6.set_yticklabels([])
ax7.set_yticklabels([])
ax8.set_yticklabels([])

ax1.set_ylabel(r'$z$',**axis_font)
ax5.set_ylabel(r'$z$',**axis_font)
ax5.set_xlabel(r'$R$',**axis_font)
ax6.set_xlabel(r'$R$',**axis_font)
ax7.set_xlabel(r'$R$',**axis_font)
ax8.set_xlabel(r'$R$',**axis_font)

pl.tight_layout()
pl.subplots_adjust(wspace=-0.1, hspace=0.2)
fig.savefig('equilibria_zoom.png',bbox_inches='tight')
pl.close()

######################
# plot panels with quantities along zaxis + observational data points
######################

def calc_simplified_beta_model(r,n0rc3beta,beta):
  n = n0rc3beta/r**(3.0*beta)
  return n

def plot_obs_n(ax):
  n   = [ [11.0], [31.5,25.5], [100,30], [3.16], [90], [8.5,21,27,31   ], [2.5], [58,8]]
  nmn = [ [15.5], [13, 15   ], [      ], [1   ], [  ], [5.5,1.2,5.1,9.8], [   ], [50,8]  ]
  npl = [ [6.50], [50, 36   ], [100,30], [10  ], [  ], [39,72,39,46    ], [   ], [100,20] ]
  d   = [ [48.2], [73.5,64.7], [15,45 ], [70  ], [20], [20,40,68,118   ], [150], [3,40]]
  dmn = [ [43.2], [59.8,51.2], [      ], [    ], [0 ], [3,10,31,66     ], [50 ], []]
  dpl = [ [53.2], [90.2,81.8], [      ], [    ], [20], [63,76,83,144   ], [250], []]
  labels       = ['Salem et al. (2015)', 'Gatto et al. (2013)', 'Stanimirovic et al. (2002)', 'Sembach et al. (2003)', 'Bregman \& LLoyd-Davies (2007)','Grcevich \& Putman (2009)','Blitz \& Bobishaw (2000)','Hsu et al. (2011)']
  symbols = ['d','^','s','o','*','s','o','v']
  mfcs = ['k','none','k','none','none','none','k','none']
  yes_xerr = [1,1,0,0,1,1,1,0]
  yes_yerr = [1,1,0,1,0,1,0,1]

  for i in [0,1,5,4,3,2,6]:
    n_curr = np.asarray(n[i])*1e-5
    nmn_curr = np.asarray(nmn[i])*1e-5
    npl_curr = np.asarray(npl[i])*1e-5
    d_curr = np.asarray(d[i])
    dmn_curr = np.asarray(dmn[i])
    dpl_curr = np.asarray(dpl[i])
    ax.plot(d_curr,n_curr,symbols[i],color='k',label=labels[i],mfc=mfcs[i])

    if yes_xerr[i]:
      xerr = [d_curr - dmn_curr, dpl_curr - d_curr]
      ax.errorbar(d_curr, n_curr, xerr=xerr, ecolor='k',capsize=1.5,capthick=1.5,elinewidth=1,ls='none')

    if yes_yerr[i]:
      yerr = [n_curr - nmn_curr, npl_curr - n_curr]
      ax.errorbar(d_curr, n_curr, yerr=yerr, ecolor='k',capsize=1.5,capthick=1.5,elinewidth=1,ls='none')

  # now add shaded area corresponding to models from Miller&Bregman2013,2015
  f = 2/0.3 # factor of 2 is to convert from n_e to n, factor of 0.3 is because we assume metallicity Z=0.3 Z_sol
  n0rc3beta_MB13, n0rc3beta_MB13_min, n0rc3beta_MB13_max = f*4.8e-2, f*1.1e-2, f*13.3e-2
  beta_MB13, beta_MB13_min, beta_MB13_max = 0.71, 0.51, 0.88
  n0rc3beta_MB15, n0rc3beta_MB15_min, n0rc3beta_MB15_max = f*1.35e-2, f*1.11e-2, f*1.59e-2
  beta_MB15, beta_MB15_min, beta_MB15_max = 0.5, 0.47, 0.53
  r = np.linspace(0,300,901)
  A,B,C,D = calc_simplified_beta_model(r,n0rc3beta_MB13_min,beta_MB13_min), calc_simplified_beta_model(r,n0rc3beta_MB13_min,beta_MB13_max), calc_simplified_beta_model(r,n0rc3beta_MB13_max,beta_MB13_min), calc_simplified_beta_model(r,n0rc3beta_MB13_max,beta_MB13_max)
  E,F,G,H = calc_simplified_beta_model(r,n0rc3beta_MB15_min,beta_MB15_min), calc_simplified_beta_model(r,n0rc3beta_MB15_min,beta_MB15_max), calc_simplified_beta_model(r,n0rc3beta_MB15_max,beta_MB15_min), calc_simplified_beta_model(r,n0rc3beta_MB15_max,beta_MB15_max)
  ax.fill_between(r,np.min(np.vstack((A,B,C,D)),axis=0),np.max(np.vstack((A,B,C,D)),axis=0),color='k',alpha=0.1,label='Miller \& Bregman (2013)')
  ax.fill_between(r,np.min(np.vstack((E,F,G,H)),axis=0),np.max(np.vstack((E,F,G,H)),axis=0),color='r',alpha=0.1,label='Miller \& Bregman (2015)')

  ax.legend(fontsize=7.5,loc='lower left',framealpha=1.0)

# function that plots the observed values of Pressure
def plot_obs_P(ax):
  d   = [ [10,50,100 ], [15,45   ], [3,40  ]]
  P   = [ [530,140,50], [1000,300], [580,84]]
  labels       = ['Fox et al. (2005)', 'Stanimirovic et al. (2002)', 'Hsu et al. (2011)']
  symbols = ['d','s','^']
  mfcs = ['k','k','none']

  for i in [2,0,1]:
    ax.plot(d[i],P[i],symbols[i],color='k',label=labels[i],mfc=mfcs[i])

  ax.legend(fontsize=8,loc='lower left',framealpha=1.0)

fig = pl.figure(figsize=(10,6))
ax1 = pl.subplot2grid((2,2), (0,0))
ax2 = pl.subplot2grid((2,2), (0,1))
ax3 = pl.subplot2grid((2,2), (1,0))
ax4 = pl.subplot2grid((2,2), (1,1))

axarr = [ax1,ax2,ax3,ax4]

# plot Paxis
axarr[0].plot(model.z1d,model.Pk[:,0],color='k')
axarr[0].set_ylabel(r'$P_{\rm axis}/k \ \rm [K cm^{-3}]$',**axis_font)
axarr[0].set_ylim(1e0,1e5)
plot_obs_P(axarr[0])

# plot qaxis
axarr[1].plot(model.b_array,model.b_array/model.a_array,color='k')
axarr[1].set_ylabel(r'$q_{\rm axis}$',**axis_font)
axarr[1].set_ylim(1e-3,1.1)

# plot naxis
axarr[2].plot(model.z1d,model.n[:,0],color='k')
axarr[2].set_ylabel(r'$n_{\rm axis}\ \rm [cm^{-3}]$',**axis_font)
axarr[2].set_ylim(1e-9,1e-2)
plot_obs_n(axarr[2])

# plot Taxis
axarr[3].plot(model.z1d,model.T[:,0],color='k')
axarr[3].set_ylabel(r'$T_{\rm axis}\ \rm [K]$',**axis_font)
axarr[3].set_ylim(1e5,1e7)

# formatting
for ax in axarr:
  ax.tick_params(labelsize=18)
  axis_font = {'size':'22'}
  ax.set_xlim(1,300)
  ax.set_xscale('log')
  ax.grid(ls='dashed')

axarr[0].set_yscale('log')
axarr[2].set_yscale('log')
axarr[3].set_yscale('log')

axarr[0].set_yticks([1e1,1e2,1e3,1e4,1e5,1e6])
axarr[2].set_yticks([1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])

ax3.set_xlabel(r'$z_{\rm axis} \ \rm [kpc]$',**axis_font)
ax4.set_xlabel(r'$z_{\rm axis} \ \rm [kpc]$',**axis_font)

pl.tight_layout()
fig.savefig('panels.pdf',bbox_inches='tight')
pl.close()

##################
# plot M(r) enclosed within spheres
##################

fig, ax = pl.subplots(1,1,figsize=(8,6))
ax.plot(model.r_enc,model.M_enc,color='k')
ax.grid(ls='dashed')
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(labelsize=16)
axis_font = {'size':'22'}
ax.set_xlabel(r'$r \ \rm [kpc]$',**axis_font)
ax.set_ylabel(r'$M_{\rm enclosed} \ [M_\odot]$',**axis_font)
ax.set_xlim(1,300)
ax.set_ylim(1e3,1e12)
ax.set_yticks(np.logspace(3,12,10))
pl.tight_layout()
fig.savefig('Menc.pdf',bbox_inches='tight')
pl.close()

##################
# plot AMD
##################

fig, ax = pl.subplots(1,1,figsize=(8,6))
ax.bar(model.bins_L[:-1], model.hist_L,width=model.bins_L[1]-model.bins_L[0])
ax.grid()
ax.tick_params(labelsize=14)
axis_font = {'size':'20'}
ax.set_xlabel(r'$l \ \rm [100 km s^{-1} kpc]$',**axis_font)
ax.set_ylabel(r'$\rm mass \ [M_\odot]$',**axis_font)
pl.tight_layout()
fig.savefig('AMD.pdf',bbox_inches='tight')
pl.close()

##################
# plot vphi at different heights above the plane
##################

fig, ax = pl.subplots(1,1,figsize=(8,6))
for i in [0,10,20,30,50,100,200]:
  zi = model.z[i][0]
  ax.plot(model.R[i],model.vphi[i],label=r'$z=%.0f$'%(zi))
ax.grid(ls='dashed')
ax.tick_params(labelsize=20)
axis_font = {'size':'24'}
ax.set_xlabel(r'$R \ [{\rm kpc}]$',**axis_font)
ax.set_ylabel(r'$v_\phi\  {\rm [100\,km/s]}$',**axis_font)
ax.set_xlim(0,100)
ax.set_ylim(0,3)
ax.legend(loc='upper right',framealpha=1.0,fontsize=18)
pl.tight_layout()
fig.savefig('vphi.pdf',bbox_inches='tight')
pl.close()