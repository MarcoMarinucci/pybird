from pybird.module import *
import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from scipy.special import expi
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy import interpolate
import time

class GreenFunction(object):
    """A class to compute cosmological Green's functions and growth factors.
    
    The GreenFunction class handles calculations related to linear perturbation theory,
    including growth factors, growth rates, and Green's functions for different
    cosmological models (ΛCDM, wCDM, quintessence). It provides the time-dependent
    functions needed for perturbation theory calculations.
    
    Attributes:
        Omega0_m (float): Present-day matter density parameter.
        H0 (float): Hubble constant at present day. (used for f(R) model)
        w (float): Dark energy equation of state parameter.
        wa (float): Dark energy equation of state time variation parameter (for w0wa model).
        Omega0_k (float): Present-day curvature density parameter.
        Omega_rc (float): Crossover scale for nDGP model.
        fR0 (float): f(R_0) model parameter.
        alphaM (float): EFT of DE parameter alpha_M.
        alphaT (float): EFT of DE parameter alpha_T.
        alphaB (float): EFT of DE parameter alpha_B.
        eta (float): EFT of DE parameter eta.
        background (str): Background expansion model ('lcdm', 'w0wa', 'evolve_Mp').
        model (str): Cosmological model ('lcdm', 'quintessence', 'nDGP', 'EFTofDE', 'bootstrap', 'fR').
        timedep (str): Time dependence of EFT parameters ('propto_omega', 'propto_scale', 'quasi_static_alphas_power_law', 'constant_alphas').
        vectorize (bool): Whether to vectorize calculations for arrays of parameters.
        OmegaL_by_Omega_m (float): Ratio of dark energy to matter density.
        wcdm (bool): Whether using wCDM cosmology.
        epsrel (float): Relative tolerance for numerical integrations.
    
    Methods:
        C(): Compute the time-dependent function C(a) for quintessence models.
        H(): Compute the conformal Hubble parameter H(a).
        H3(): Compute H(a)^-3.
        Omega_m(): Compute the time-dependent matter density parameter Ω_m(a).
        D(): Compute the linear growth factor D(a).
        DD(): Compute the derivative of the growth factor D'(a).
        fplus(): Compute the growth rate f+(a) = aD'(a)/D(a).
        Dminus(): Compute the decay factor D-(a).
        DDminus(): Compute the derivative of the decay factor D-'(a).
        fminus(): Compute the decay rate f-(a) = aD-'(a)/D-(a).
        W(): Compute the Wronskian W(a) = D'(a)D-(a) - D(a)D-'(a).
        
        G1d(), G2d(), G1t(), G2t(): Compute Green's functions for perturbation theory.
        I1d(), I2d(), I1t(), I2t(): Compute second order coefficients.
        mG1d(), mG2d(), mG1t(), mG2t(): Compute second order time integrals.
        G(): Compute quintessence time function.
        IU1d(), IU2d(), IU1t(), IU2t(): Compute third order coefficients.
        IV11d(), IV12d(), IV21d(), IV22d(): Compute third order coefficients.
        IV11t(), IV12t(), IV21t(), IV22t(): Compute third order coefficients.
        mU1d(), mU2d(), mU1t(), mU2t(): Compute third order time integrals.
        mV11d(), mV12d(), mV21d(), mV22d(): Compute third order time integrals.
        mV11t(), mV12t(), mV21t(), mV22t(): Compute third order time integrals.
        Y(): Compute third order time function.
    """

    def __init__(self, Omega0_m, H0 = 67.7, w=-1., wa = 0., Omega0_k=0., Omega_rc=None, fR0 = 0., alphaM=0., alphaT=0.,alphaB=0., eta = 1., background='lcdm', model='lcdm', timedep = 'propto_omega', vectorize=False):
        options_background = ['lcdm', 'w0wa','evolve_Mp'] #MM: notice that evolving alphaM and varying w is still not implemented, don't know if we want to do this
        options_model = ['lcdm','quintessence', 'nDGP', 'EFTofDE', 'bootstrap', 'fR']
        options_timedep = ['propto_omega', 'propto_scale', 'quasi_static_alphas_power_law', 'constant_alphas']
        self.vectorize = vectorize
        self.Omega0_m = Omega0_m
        if self.vectorize:
            self.Omega0_m = np.array(Omega0_m)
            self.Omega0_k = np.array(Omega0_k)
            self.w = np.array(w) if w is not None else None
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m-Omega0_k)/self.Omega0_m
        self.wcdm = False
        self.quintessence = False
        self.nDGP = False
        self.w0wa = False
        self.EFTofDE = False
        self.boot = False
        self.fR = False
        self.LCDM = True
        self.background = background
        self.model = model
        self.timedep = timedep
        if self.background in options_background:
            if self.background == 'w0wa':
                self.w0wa = True
                self.w0 = w
                self.wa = wa
                self.LCDM = False
        else: raise Exception("Choose the background expansion among [lcdm, w0wa, evolve_Mp]")
        if self.model in options_model:
            if self.model == 'quintessence':
                self.quintessence = True
                self.w0 = w
                self.w = np.array(w) if self.vectorize else w # C(), H() and the quintessence integrals use self.w
                # LCDM stays True: quintessence goes through the integral expressions (H, C with w), as on master
            elif self.model == 'nDGP':
                self.nDGP = True
                self.Omega_rc = Omega_rc
                self.LCDM = False
            elif self.model == 'EFTofDE':
                self.EFTofDE = True
                if alphaB == None:
                    self.alphaB = 0.
                else:
                    self.alphaB = alphaB
                if alphaT == None:
                    self.alphaT = 0.
                else:
                    self.alphaT = alphaT
                self.eta = eta
                if alphaM == None:
                    self.alphaM = 0.
                else:
                    self.alphaM = alphaM
                self.LCDM = False # I need this for the kernel's functions
                if self.timedep in options_timedep: pass
                else: raise Exception('You asked for the EFTofDE model, but you should choose the time dependence\n\
                                      of the alpha_i parameters among [propto_omega, propto_scale, quasi_static_alphas_power_law, constant_alphas]')
            elif self.model == 'fR':
                self.fR = True
                self.fR0 = fR0
                self.LCDM = True #I'm setting this to be True for the Green's functions are calculated using LCDM
            elif self.model == 'lcdm':
                if self.background == 'w0wa':
                    self.LCDM = False
                else:
                    pass
            elif self.model == 'bootstrap':
                self.LCDM = True #this is because we are using a model agnostic approach and we use the epsilon prescription
        else: raise Exception('Choose the model among [lcdm, quintessence, nDGP, EFTofDE, bootstrap, fR]')

        self.epsrel = 1e-4
        if self.LCDM == False:
            x_arr = np.log(np.logspace(-7,0.5,100,base=np.e))
            f_num = self.compute_f()
            self.f_num = interpolate.interp1d(x_arr, f_num, fill_value='extrapolate')
            boot2 = self.compute_ag_dg2()
            self.ag2_num = interpolate.interp1d(x_arr[:-1], boot2[0], fill_value='extrapolate')
            self.dg2_num = interpolate.interp1d(x_arr[:-1], boot2[1], fill_value='extrapolate')
            boot3 = self.compute_ag_dg3()
            self.ag3_num = interpolate.interp1d(x_arr[:-2], boot3[0], fill_value='extrapolate')
            self.dg3_num = interpolate.interp1d(x_arr[:-2], boot3[1], fill_value='extrapolate')
    ##################################################################
    # Scale-dependent growth block
    ##################################################################
    ##################################################################
    def D_scale(self, avals, kvals):
        f = np.zeros([len(avals),len(kvals)])

        d_ini = self.D_LCDM(avals[0])

        start = time.time()
        for i, k in enumerate(kvals):
            for j, a in enumerate(avals):
                #f[j,i] = d_ini*np.exp(-quad(lambda ap: self.f_scale(np.exp(ap),k), np.log(a), 0., epsrel=self.epsrel)[0])
                #f[j,i] = d_ini*np.exp(quad(lambda ap: self.f_scale(ap,k)/ap, avals[0], a, epsrel=self.epsrel)[0])
                f[j,i] = d_ini*np.exp(quad(lambda ap: self.f_scale(np.exp(ap),k), np.log(avals[0]), np.log(a), epsrel=self.epsrel)[0])
                #f[j,i] = d_ini*np.exp(quad(lambda x: self.f_scale(np.exp(x),k), np.log(avals[0]), np.log(a), epsrel=self.epsrel)[0])
        finish = time.time()
        print("(D) Time needed for {n} wavenumbers: {t} sec.".format(n=len(kvals), t=finish-start))
        return f

    def compute_f_scale(self, avals, kvals):
        f = np.zeros([len(avals),len(kvals)])

        gamma = 0.55

        start = time.time()
        for i, k in enumerate(kvals):
            sol = solve_ivp(self.F_source, [avals[0], avals[-1]], [(self.OM_m(avals[0]))**gamma], t_eval=avals, method="RK45", args={k}, rtol = 1e-8)
            f[:,i] = sol.y

         # results = Parallel(n_jobs = 1)(delayed(p_zone)(f, a_eval, k_arr, f_0, i) for i in range(len(k_arr)))

        finish = time.time()
        print("Time needed for {n} wavenumbers: {t} sec.".format(n=len(kvals), t=finish-start))

        return f


    def OM_m(self, a):
        return self.Omega0_m * a**(-3.)/self.E(a)**2

    def Pi(self, k, a):
        #return (k/a)**2 + self.H0**2 *(self.Omega0_m+4.*a**3 *self.Omega0_m*self.OmegaL_by_Omega_m)**3 / (2.*self.fR0*a**9 *(3.*self.Omega0_m - 4.)**2)
        return (k/a)**2 + self.H0**2 *(self.Omega0_m+4.*a**3 *self.Omega0_m*self.OmegaL_by_Omega_m)**3 / (self.fR0*a**9 *(3.*self.Omega0_m - 4.)**2)

    def Mu(self, k, a):
        return 1.+pow(k/a,2.)/ (3.* self.Pi(k,a))

    def z_a(self, a):
        return 1./(a) - 1.

    def a_z(self, z):
        return 1./(1.+z)

    def E(self, a):
        return np.sqrt(self.Omega0_m/a**3 + self.Omega0_m*self.OmegaL_by_Omega_m)

    def F_source(self, t, f, k):
        return (-f**2 - (2. - 1.5*self.OM_m(t))*f + 1.5*self.OM_m(t)*self.Mu(k, t)) / t

    ########################################################################################
    # End of scale-dependent growth block
    ########################################################################################

    def C(self, a):
        if self.quintessence: return 1. + (1.+self.w) * self.OmegaL_by_Omega_m * a**(-3.*self.w)
        else: return 1.

    def w_w0wa(self, a):
        if self.background == 'lcdm':
            return -1.
        #elif self.background == 'wcdm':
        #    return self.w0
        elif self.background == 'w0wa':
            return self.w0 + self.wa*(1 - a)
        else:
            raise Exception('You have selected an expansion model which is not among lcdm or w0wa')

    def H(self, a):
        """Conformal Hubble"""
        if self.quintessence: return ( self.Omega0_m/a + (1.-self.Omega0_m)*a**2 * a**(-3.*(1.+self.w)) )**.5
        if self.w0wa: return ( self.Omega0_m/a + (1.-self.Omega0_m)*a**2 * a**(-3.*(1.+self.w0 + self.wa))*np.e**(-3*(1-a)*self.wa) )**.5
        else: return (self.Omega0_m/a + (1.-self.Omega0_m)*a**2)**.5

    def H3(self, a):
        return self.C(a)/self.H(a)**3

    def Omega_m(self, a):
        return self.Omega0_m / (self.H(a)**2 * a)
    
    #MM part for exact time with MG. This part is not vectorized yet.

    def H_NC(self, a):
        """Non-Conformal Hubble, H0=1"""
        if self.w0wa:
            return (self.Omega0_m/a/a/a + (1.-self.Omega0_m)*a**(-3*(1 + self.w0 + self.wa))*np.e**(-3*(1-a)*self.wa))**0.5 #expansion in w0-wa
        else:
            return (self.Omega0_m/a/a/a + (1.-self.Omega0_m))**0.5 #expansion fixed to LCDM

    def dHda_NC(self, a):
        """Derivative of non-Conformal Hubble w.r.t. a, H0=1"""
        return 0.5*(-3.*self.Omega0_m/a/a/a/a)*(self.Omega0_m/a/a/a + (1.-self.Omega0_m))**(-0.5) #expansion fixed to LCDM


    def H_x(self,x):
        """H using the time variable x = log(a)"""
        """Useful for numerical estimation of D and f"""
        if self.w0wa:
            return (self.Omega0_m*np.e**(-3.*x) + (1. - self.Omega0_m)*np.e**(-3*x*(1 + self.w0 + self.wa))*np.e**(-3*(1 - np.e**x)*self.wa))**0.5
        else:
            return (self.Omega0_m*np.e**(-3.*x) + 1. - self.Omega0_m)**0.5 #background expansion fixed to LCDM


    def Om_x(self,x):
        """Matter evolution using the time variable x = log(a)"""
        if self.background == 'evolve_Mp':
            if np.isclose(self.alphaM, 0., atol=1e-7):
                return (self.Omega0_m)*(np.e**(-3*x))/(self.H_x(x)**2.)
            else:
                if np.isclose(self.eta,3., atol=0.001):
                    num_loc = (3*self.Omega0_m)
                    den_loc = (np.e**((1/3)*np.e**(3*x)*self.alphaM)*(3*np.e**(3*x) - \
                        np.e**(3*x)*self.alphaM*self.Omega0_m*(3/(np.e**(self.alphaM/3)*self.alphaM) + expi(-(self.alphaM/3))) + \
                        np.e**(3*x)*self.alphaM*self.Omega0_m*((3*np.e**(-3*x - (1/3)*np.e**(3*x)*self.alphaM))/self.alphaM + \
                        expi((-(1/3))*np.e**(3*x)*self.alphaM))))

                elif np.isclose(self.eta,3./2.  , atol=0.001):
                    num_loc = (3*self.Omega0_m)/np.e**((2/3)*(np.e**x)**(3/2)*self.alphaM)
                    den_loc = (2*((3*np.e**(3*x))/2 - (4/3)* np.e**(3*x)*self.alphaM**2*self.Omega0_m*(-((27*(-(self.alphaM/ \
                            3) + (2*self.alphaM**2)/9))/(np.e**((2*self.alphaM)/3)*(8*self.alphaM**3))) - \
                            (1/2)*expi(-((2*self.alphaM)/3))) + (4/3)*np.e**(3*x)*self.alphaM**2*self.Omega0_m*(-((27*((-(1/3))*(np.e**x)**(3/ \
                            2)*self.alphaM + (2/9)*np.e**(3*x)*self.alphaM**2))/(np.e**((2/3)*(np.e**x)**(3/2)*self.alphaM)*(8*(np.e**x)**(9/ \
                            2)*self.alphaM**3))) - (1/2)*expi((-(2/3))*(np.e**x)**(3/2)*self.alphaM))))

                else:
                    raise Exception('You have selected a value for eta which is not implemented when Mp is varied!!')
                return num_loc/den_loc
        else:
            return (self.Omega0_m)*(np.e**(-3*x))/(self.H_x(x)**2.)

    def dlogHdx(self,x):
        """Log Derivative of H w.r.t. the time variable x = log(a)"""
        """Useful for numerical estimation of D and f"""
        if self.w0wa:
            return - 3/2*(self.Om_x(x) + (1 + self.w_w0wa(np.e**x))*(1 - self.Om_x(x)))
        else:
            return -3/2*self.Om_x(x) #background expansion fixed to LCDM


    def alpha_fun(self, a, alpha0):
        if self.timedep == 'propto_omega':
            return alpha0*(1 - self.Omega_m(a))/(1-self.Omega0_m)
        elif self.timedep == 'propto_scale':
            return alpha0*a
        elif self.timedep == 'quasi_static_alphas_power_law':
            return alpha0*a**(self.eta)
        elif self.timedep == 'constant_alphas':
            return alpha0
    def alpha_fun_x(self, x, alpha0):
        if self.timedep == 'propto_omega':
            return alpha0*(1 - self.Om_x(x) )/(1-self.Omega0_m)
        elif self.timedep == 'propto_scale':
            return alpha0*(np.e**(x))
        elif self.timedep == 'quasi_static_alphas_power_law':
            return alpha0*(np.e**(self.eta*x))
        elif self.timedep == 'constant_alphas':
            return alpha0
    def D_alpha(self, a, alpha0):
        '''Derivative of the alpha function'''
        if self.timedep == 'propto_omega':
            return -3.*self.w_w0wa(a)*self.Omega_m(a)*self.alpha_fun(a,alpha0)
        elif self.timedep == 'propto_scale':
            return 1.*self.alpha_fun(a,alpha0)
        elif self.timedep == 'quasi_static_alphas_power_law':
            return self.eta*self.alpha_fun(a,alpha0)
        elif self.timedep == 'constant_alphas':
            return 0

    def M2(self,a):
        #coded only for ~a^eta case
        return np.exp(self.alpha_fun(a,self.alphaM)/self.eta)
    def xi_fun(self, a):
        return self.alpha_fun(a, self.alphaB)*(1 + self.alpha_fun(a, self.alphaT)) + self.alpha_fun(a, self.alphaT) -  self.alpha_fun(a, self.alphaM)
    def nu_func(self, a):
        return - ((1 + self.alpha_fun(a,self.alphaB))*(self.xi_fun(a) + self.dlogHdx(np.log(a))) + self.D_alpha(a,self.alphaB) + 3/2*(self.Om_x(np.log(a))))

    def mu_x(self,x):
        """mu function of nDGP/EFTofDE, time variable x = log(a)"""
        if self.nDGP: return 1 + 1/3*np.sqrt(self.Omega_rc)/(np.sqrt(self.Omega_rc) + self.H_x(x)*(1 + self.dlogHdx(x)/3))
        #elif self.EFTofDE: return 1 - (self.alphaB_fun_x(x))/(1 + self.alphaB_fun_x(x) + 3/2*self.Om_x(x))
        elif self.EFTofDE: return 1 + self.alpha_fun_x(x,self.alphaT) + (self.xi_fun(np.e**x)**2)/(self.nu_func(np.e**x))
        else: return 1.

    def mu_psi_x(self,x):
        """mu_psi function of EFTofDE, time variable x = log(a)"""
        #if self.EFTofDE: return 1 - (self.alphaB_fun_x(x))/(1 + self.alphaB_fun_x(x) + 3/2*self.Om_x(x))
        if self.EFTofDE: return 1 + (self.alpha_fun_x(x, self.alphaB)*self.xi_fun(np.e**x))/(self.nu_func(np.e**x))
        else: return 0.

    def mu_chi_x(self,x):
        """mu_chi function of EFTofDE, time variable x = log(a)"""
        #if self.EFTofDE: return - 1/(1 + self.alphaB_fun_x(x) + 3/2*self.Om_x(x))
        if self.EFTofDE: return (self.xi_fun(np.e**x))/(self.nu_func(np.e**x))
        else: return 0.

    def mu(self, a):
        if self.nDGP: return 1 + 1/3 * np.sqrt(self.Omega_rc)/ (np.sqrt(self.Omega_rc) + self.H_NC(a)*(1.+a*self.dHda_NC(a)/3./self.H_NC(a)))#return 1.+1./(3.*self.beta(a))
        #elif self.EFTofDE: return 1 - (self.alphaB_fun(a))/(1 + self.alphaB_fun(a) + 3/2*self.Omega_m(a))
        elif self.EFTofDE: return 1 + self.alpha_fun(a,self.alphaT) + (self.xi_fun(a)**2)/(self.nu_func(a))
        else: return 1.

    def mu_psi(self,a):
        """mu_psi function of nDGP, time variable a"""
        if self.EFTofDE: return 1 + (self.alpha_fun(a, self.alphaB)*self.xi_fun(a))/(self.nu_func(a))
        else: return 0.

    def mu_chi(self,a):
        """mu_chi function of nDGP, time variable a"""
        #if self.EFTofDE: return - 1/(1 + self.alphaB_fun(a) + 3/2*self.Omega_m(a))
        if self.EFTofDE: return (self.xi_fun(a))/(self.nu_func(a))
        else: return 0.

    def den(self,a):
        return np.sqrt(self.Omega_rc) + self.H_NC(a)*(1.+a*self.dHda_NC(a)/3./self.H_NC(a))

    def C3(self,a):
        if self.EFTofDE:
            return -self.alpha_fun(a,self.alphaT)
        else: return 0.
    def C4(self,a):
        if self.EFTofDE:
            return -4*self.alpha_fun(a,self.alphaB) + 2*self.alpha_fun(a,self.alphaM) - 3*self.alpha_fun(a,self.alphaT)
        else: return 0.

    def mu2(self, a):
        #if self.nDGP: return (-0.5*self.H_NC(a)**2.*(1./(3.*self.beta(a)))**3./self.Omega_rc)
        #this is what we call mu_{\phi, 2} in the EFTofDE notation
        if self.nDGP: return -0.5*self.H_NC(a)**2.*np.sqrt(self.Omega_rc)/(3.*self.den(a))**3.
        elif self.EFTofDE: return (self.mu_chi(a)/4)*(3*self.mu_chi(a)*self.mu_psi(a)*self.C3(a) + self.mu_chi(a)**2*self.C4(a))
        else: return 0.

    def mu22(self, a):
        #if self.nDGP: return (0.5*self.H_NC(a)**4.*(1./(3.*self.beta(a)))**5./self.Omega_rc/self.Omega_rc)
        #this is what we call mu_{\phi, 22} in the EFTofDE notation
        if self.nDGP: return 0.5*self.H_NC(a)**4.*(self.Omega_rc**(3./2.))/(3.*self.den(a))**5.
        elif self.EFTofDE: return (1/(8*self.nu_func(a)))*((3*self.mu_psi(a) - 1)*self.C3(a)*self.mu_chi(a) + self.C4(a)*self.mu_chi(a)**2)**2
        else: return 0.

    def rm_f(self, f, x):
        '''Right member of the differential equation for f'''
        dfdx = -(2 + f)*f - self.dlogHdx(x)*f + 3/2*self.mu_x(x)*self.Om_x(x)
        return  dfdx
    def compute_f(self):
        xvals = np.log(np.logspace(-7,0.5,100,base=np.e))
        f0 = (-1 + np.sqrt(25 + 24*(self.mu_x(xvals[0]) -1)))/4
        f_plus = odeint(self.rm_f,f0,xvals)[:,0]
        return f_plus

    def rm_ag_dg2(self,w,x):
        '''Right members of the differential equations for a_gamma^2 and d_gamma^2'''
        a1, d1 = w
        # mu multiplies the full delta in the Poisson equation, delta_2 included (as in rm_ag_dg3 and in the scale-dependent rm_ag_dg2_vec)
        rm = [self.f_num(x)*(2. - 2.*a1 + d1),
        self.f_num(x)*(- d1 + 3./2. *self.mu_x(x)*self.Om_x(x)/(self.f_num(x)**2.)*(a1 - d1) + 2.*self.mu2(np.e**x)/(self.f_num(x)**2.)*(3./2.*self.Om_x(x))**2.)]
        return rm
    def compute_ag_dg2(self):
        xvals = np.log(np.logspace(-7,0.5,100,base=np.e))
        a1_in = 2*5/7 #EdS initial conditions
        d1_in = 2*3/7 #EdS initial conditions
        in0 = [a1_in, d1_in] #EdS initial conditions
        wsol = odeint(self.rm_ag_dg2, in0, xvals[:-1])
        agsol = wsol[:,0]
        dgsol = wsol[:,1]
        return agsol, dgsol

    def rm_ag_dg3(self,w,x):
        '''Right members of the differential equations for a_gamma^2 and d_gamma^2'''
        a3, d3 = w
        rm = [self.f_num(x)*(2*self.dg2_num(x) - 3*a3 + d3),
         self.f_num(x)*(- 2*d3 + 3/2 *self.Om_x(x)/(self.f_num(x)**2)*(a3 -d3)*self.mu_x(x) + 2*(3/2*self.Om_x(x))**2*(3/2*self.Om_x(x)/self.f_num(x)**2*self.mu22(np.e**x) + self.ag2_num(x)*self.mu2(np.e**x)/self.f_num(x)**2))]
        return rm
    def compute_ag_dg3(self):
        xvals = np.log(np.logspace(-7,0.5,100,base=np.e))
        a3_in = 2/3 #EdS initial conditions
        d3_in = 2/7 #EdS initial conditions
        in0 = [a3_in, d3_in] #EdS initial conditions
        wsol = odeint(self.rm_ag_dg3, in0, xvals[:-2])
        agsol = wsol[:,0]
        dgsol = wsol[:,1]
        return agsol, dgsol



    def compute_primes_MG(self, Y, x):
        """Second order differential equation for growth factor D extended from Eq.(A.1) of 2005.04805"""
        """The equation is solved in the time variable x = log(a)"""
        D, DD = Y
        if self.w0wa:
            DDD = [DD, - (2. + self.dlogHdx(x))*DD + 3./2.*self.Om_x(x)*D]
        if self.nDGP:
            DDD = [DD,- (2. + self.dlogHdx(x))*DD + 3./2.*self.mu_x(x)*self.Om_x(x)*D]
        if self.EFTofDE:
            DDD = [DD,- (2. + self.dlogHdx(x))*DD + 3./2.*self.mu_x(x)*self.Om_x(x)*D]
        return DDD


    def D_DD_MG_num(self,a):
        """Solve for growth and decay mode in general scale-independent cosmologies"""
        """Only time-dependent growth factor"""
        """Time variable x = log(a)"""
        xin = -7.
        xfin = +7.
        if self.w0wa or self.EFTofDE:
            xfin = 4.
        xpoints = 1000
        #if a < np.e**xin:
        #    print('Need to decrease a_ini from ', np.e**xin, ' to below ', a)
        delta_x = (xfin - xin)/xpoints
        xss = np.arange(xin,xfin,delta_x) # x's for the growth mode solution
        #xss_inv = xss[::-1] # inverted a's for the decay mode solution

        #Initial conoditions
        D0plus = [np.e**xin,np.e**xin]
        #Numerical solutions
        ans_plus = odeint(self.compute_primes_MG,D0plus,xss,mxstep = 4000)
        Dplus = ans_plus[:,0] #EARLY time initial conditions (EdS approx. is sufficient) for the growing mode
        DDplus = ans_plus[:,1] #LATE time initial conditions (EdS approx. is sufficient) for the decay mode


        #Dp = interpolate.interp1d(xss,Dplus,kind = 'cubic')
        #DDp = interpolate.interp1d(xss,DDplus,kind = 'cubic')
        return Dplus,DDplus#Dp(np.log(a)),DDp(np.log(a))

    def D_DD_minus_MG_num(self,a):
        """Solve for decay mode in general scale-independent cosmologies"""
        """Only time-dependent growth factor"""
        """Time variable x = log(a)"""
        xin = -7.
        xfin = +7.
        if self.w0wa or self.EFTofDE:
            xfin = 4.
        xpoints = 1000
        #if a < np.e**xin:
        #    print('Need to decrease a_ini from ', np.e**xin, ' to below ', a)
        delta_x = (xfin - xin)/xpoints
        xss = np.arange(xin,xfin,delta_x) # x's for the growth mode solution
        xss_inv = xss[::-1] # inverted a's for the decay mode solution

        #Initial conditions
        D0minus = [np.e**(-2*xfin),-2*np.e**(-2*xfin)]
        #Numerical solutions
        ans_minus = odeint(self.compute_primes_MG,D0minus,xss_inv,mxstep = 4000)
        Dminus1 = ans_minus[:,0][::-1]
        DDminus = ans_minus[:,1][::-1]
        Dminus = Dminus1/(Dminus1[0]/(np.e**(-3*xin/2)))
        DDminus = DDminus/(Dminus1[0]/(np.e**(-3*xin/2)))
        #Dm = interpolate.interp1d(xss,Dminus,kind = 'cubic')
        #DDm = interpolate.interp1d(xss,DDminus,kind = 'cubic')
        return Dminus,DDminus#Dm(np.log(a)),DDm(np.log(a))

    def D(self, a):
        """Growth factor"""
        # if self.wcdm: return a*hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-(a**(-3*self.w))*self.OmegaL_by_Omega_m)
        # else:
        #     if self.vectorize:
        #         I = quad_vec(self.H3, 0, a, epsrel=self.epsrel)[0]
        #         return 5 * self.Omega0_m * I * self.H(a) / (2.*a)
        #     else:
        #         I = quad(self.H3, 0, a, epsrel=self.epsrel)[0]
        #     return 5 * self.Omega0_m * I * self.H(a) / (2.*a)
        if self.LCDM == False:
            #return quad(lambda ap: self.f_num(np.log(ap))/ap, 1e-4,a, epsrel=self.epsrel)[0]
            return np.e**-4*np.e**quad(lambda x: self.f_num(x), -4,np.log(a), epsrel=self.epsrel)[0]
        else:
            I = quad(self.H3, 0, a, epsrel=self.epsrel)[0]
            return 5 * self.Omega0_m * I * self.H(a) / (2.*a)
    
    def D_LCDM(self, a):
        """Growth factor in LCDM"""
        #return a*hyp2f1((-2.)/(-2.),-1/(-3),1-(5/(-6)),-(a**(3))*self.OmegaL_by_Omega_m)
        return a*hyp2f1(1.,1./3.,11/6,-(a**3)*self.OmegaL_by_Omega_m)

    def DD(self, a):
        """Derivative of growth factor"""
        if self.nDGP: return self.DDp_nDGP(np.log(a))/a#/self.Dp_nDGP(np.log(1))
        elif self.w0wa: return self.DDp_w0wa(np.log(a))/a#/self.Dp_nDGP(np.log(1))
        elif self.wcdm: return (-(a**(-3.*self.w))*self.OmegaL_by_Omega_m*((3*(self.w-1))/(6.*self.w-5.))*hyp2f1(1.5-0.5*(1/self.w),1-(1/(3.*self.w)),2-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)+hyp2f1((self.w-1)/(2.*self.w),-1/(3.*self.w),1-(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m))#/(hyp2f1((self.w-1)/(2*self.w),-1/(3*self.w),1-(5/(6*self.w)),-self.OmegaL_by_Omega_m))
        elif self.EFTofDE: return self.DDp_EFTofDE(np.log(a))/a
        else: return (2.5-(1.5*self.D(a)/a)) * self.Omega_m(a) * self.C(a)

    def fplus(self, a):
        """Growth rate"""
        if self.LCDM:
            return a * self.DD(a) / self.D(a)
        else:
            return float(self.f_num(np.log(a)))

    def Dminus(self, a):
        """Decay factor"""
        if self.nDGP: return self.Dm_nDGP(np.log(a))#/self.Dm_nDGP(np.log(1))
        elif self.w0wa: return self.Dm_w0wa(np.log(a))#/self.Dm_nDGP(np.log(1))
        elif self.wcdm: return a**(-3/2.)*hyp2f1(1/(2.*self.w),(1/2.)+(1/(3.*self.w)),1+(5/(6.*self.w)),-(a**(-3.*self.w))*self.OmegaL_by_Omega_m)
        elif self.EFTofDE: return self.Dm_EFTofDE(np.log(a))
        else: return self.H(a) / (a*self.Omega0_m**.5)

    def DDminus(self, a):
        """Derivative of decay factor"""
        if self.nDGP: return self.DDm_nDGP(np.log(a))/a#/self.Dm_nDGP(np.log(1))
        elif self.w0wa: return self.DDm_w0wa(np.log(a))/a#/self.Dm_nDGP(np.log(1))
        elif self.wcdm: return ((-1+3.*self.w)*hyp2f1(0.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m))-(2+3.*self.w)*hyp2f1(1.5+1/(3.*self.w),1/(2.*self.w),1+5/(6.*self.w),-(a**(-3.*self.w))*(self.OmegaL_by_Omega_m)))/(2*(a**(5/2.)))
        elif self.EFTofDE: return self.DDm_EFTofDE(np.log(a))/a
        else: return -1.5 * self.Omega_m(a) * self.Dminus(a) / a * self.C(a)

    def fminus(self, a):
        """Decay rate"""
        return a * self.DDminus(a) / self.Dminus(a)

    def W(self, a):
        """Wronskian"""
        return self.DDminus(a) * self.D(a) - self.DD(a) * self.Dminus(a)

    #greens functions
    def G1d(self, a, ai):
        return(self.DDminus(ai)*self.D(a)-self.DD(ai)*self.Dminus(a))/(ai*self.W(ai))
    def G2d(self, a, ai):
        return self.fplus(ai)*(self.Dminus(a)*self.D(ai)-self.D(a)*self.Dminus(ai))/(ai*ai*self.W(ai))
    def G1t(self, a, ai):
        return a*(self.DDminus(ai)*self.DD(a)-self.DD(ai)*self.DDminus(a))/(self.fplus(a)*ai*self.W(ai))
    def G2t(self, a, ai):
        return a*self.fplus(ai)*(self.DDminus(a)*self.D(ai)-self.DD(a)*self.Dminus(ai))/(self.fplus(a)*ai*ai*self.W(ai))

    # second order coefficients
    def I1d(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G1d(a,ai)/self.D(a)**2 / self.C(ai)
    def I2d(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2d(a,ai)/self.D(a)**2 / self.C(ai)
    def I1t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G1t(a,ai)/self.D(a)**2 / self.C(ai)
    def I2t(self, ai, a):
        return self.fplus(ai)*self.D(ai)**2*self.G2t(a,ai)/self.D(a)**2 / self.C(ai)

    # second order time integrals
    def mG1d(self, a):
        # if self.vectorize:
        #     return quad_vec(self.I1d,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        # else:
        #     return quad(self.I1d,0,a,args=(a,), epsrel=self.epsrel)[0]
        if self.LCDM == False:
            return self.ag2_num(np.log(a))/2
        else:
            return quad(self.I1d,0.,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2d(self, a):
        # if self.vectorize:
        #     return quad_vec(self.I2d,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        # else:
        #     return quad(self.I2d,0,a,args=(a,), epsrel=self.epsrel)[0]
        if self.LCDM == False:
            return 1 - self.mG1d(a)
        else:
            return quad(self.I2d,0.,a,args=(a,), epsrel=self.epsrel)[0]
    def mG1t(self, a):
        # if self.vectorize:
        #     return quad_vec(self.I1t,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        # else:
        #     return quad(self.I1t,0,a,args=(a,), epsrel=self.epsrel)[0]
        if self.LCDM == False:
            return self.dg2_num(np.log(a))/2
        else:
            return quad(self.I1t,0.,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2t(self, a):
        # if self.vectorize:
        #     return quad_vec(self.I2t,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        # else:
        #     return quad(self.I2t,0,a,args=(a,), epsrel=self.epsrel)[0]
        if self.LCDM == False:
            return 1 - self.mG1t(a)
        else:
            return quad(self.I2t,0.,a,args=(a,), epsrel=self.epsrel)[0]

    # quintessence time function
    def G(self, a):
        return self.mG1d(a) + self.mG2d(a)

    # third order coefficients
    def IU1d(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IU2d(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IU1t(self, ai, a):
        return self.fplus(ai)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IU2t(self, ai, a):
        return self.fplus(ai)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)

    def IV11d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV12d(self, ai, a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV21d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV22d(self, ai, a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)

    def IV11t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV12t(self, ai,a):
        return self.fplus(ai)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV21t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
    def IV22t(self, ai,a):
        return self.fplus(ai)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(ai)/self.D(a))**3 / self.C(ai)
   
    # third order time integrals
    def mU1d(self, a):
        if self.vectorize:
            return quad_vec(self.IU1d,0,a,args=(a,), epsrel=self.epsrel)[0]

        else:
            return quad(self.IU1d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2d(self, a):
        if self.vectorize:
            return quad_vec(self.IU2d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU1t(self, a):
        if self.vectorize:
            return quad_vec(self.IU1t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU1t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2t(self, a):
        if self.vectorize:
            return quad_vec(self.IU2t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2t,0,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11d(self, a):
        if self.vectorize:
            return quad_vec(self.IV11d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12d(self, a):
        if self.vectorize:
            return quad_vec(self.IV12d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV12d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21d(self, a):
        if self.vectorize:
            return quad_vec(self.IV21d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21d,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22d(self, a):
        if self.vectorize:
            return quad_vec(self.IV22d,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22d,0,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11t(self, a):
        if self.vectorize:
            return quad_vec(self.IV11t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12t(self, a):
        # if self.vectorize:
        #     return quad_vec(self.IV12t,0,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-5)[0]
        # else:
        #     return quad(self.IV12t,0,a,args=(a,), epsrel=self.epsrel)[0]
        if self.LCDM==False:
            return self.ag2_num(np.log(a))/2 - self.dg3_num(np.log(a))/4 - 1/2
        else:
            return quad(self.IV12t,0.,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21t(self, a):
        if self.vectorize:
            return quad_vec(self.IV21t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21t,0,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22t(self, a):
        if self.vectorize:
            return quad_vec(self.IV22t,0,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22t,0,a,args=(a,), epsrel=self.epsrel)[0]

    def Y(self, a):
        # if self.quintessence: return -3/14.*self.G(a)**2 + self.mV11d(a) + self.mV12d(a)
        # else: return -3/14. + self.mV11d(a) + self.mV12d(a)
        if self.LCDM == False:
            return self.ag2_num(np.log(a))/2 - 5/7
        elif self.quintessence: return -3/14.*self.G(a)**2 + self.mV11d(a) + self.mV12d(a)
        else:
            return -3/14. + self.mV11d(a) + self.mV12d(a)