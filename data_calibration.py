#notes: _____________________________________________________________________
#https://arxiv.org/pdf/2111.06519
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#https://arxiv.org/html/2602.03638v1#S3
#https://iopscience.iop.org/article/10.3847/1538-4357/adf222/pdf

#BC_g = m_bol − m_g
#so m_bol = m_g + BC_g(g−r)
#provide BC (g-r)

# 1. measure apparent mag in g and r
# 2. computer g-r, plug into martinez polynomial
# 3. get m_bol

#to do:
#check extinction coef, same w format of extinction correction (SDSS variables)
#figure out ZTF fid int or str
#add in curvefit
#input target.py
#write main pipeline for each targets
#____________________________________________________________________________

import numpy as np
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18
import astropy.units as u

#constants
M_BOL_SUN = 4.74
L_SUN = 3.828e33

#ZTF extinction coefficients find which version of these we are using
#R_g =  
#R_r = 
#R_i = 

#nickel/cobalt decay time scales
TAU_NI = 8.8
TAU_CO = 113.6

PLATEAU_SEARCH_START = 60
PLATEAU_SEARCH_END   = 150

#martinez table 4 g-r BC polynomial coeffs
#BC = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4, x = g-r

MARTINEZ_BC_GR = {
    'plateau': {'coeffs': [-0.219,  0.813,  -2.194,  1.205,  -0.305],
                'range':  (0.01, 2.17),
            },
    'tail': {'coeffs': [-9.994, 21.507, -15.343,  3.273,  None],
             'range':  (0.78, 2.07),
            },
}

# Goldberg eq 22 explosion energy coefficients
GOLDBERG_EXP = {
    'coeffs': [-0.728, 2.148, -0.280, 2.091, -1.632]
}

#distance modulus in Mpc
def distance_modulus(z):
    d_L = Planck18.luminosity_distance(z).to(u.Mpc).value
    mu  = 5.0 * np.log10(d_L) + 25.0
    return mu, d_L

#corrects time dilation
def compute_phase_rest(mjd, t_explosion, z):
    return (mjd - t_explosion) / (1 + z) 

#bolometric corrections
def assign_phase_label(phase_rest_days, tp):

    phase_rest_days = np.asarray(phase_rest_days)
    labels = np.full(phase_rest_days.shape, 'plateau', dtype=object)
    labels[phase_rest_days > tp] = 'tail'
    return labels

def BC_martinez2022(g_minus_r, phase='plateau'):
    #M_bol = M_g + BC(g-r)
    #g_minus_r: float or array
    #phase: 'plateau'/'tail'
    #sigma: calibration scatter for this phase (magnitudes)
    
    entry = MARTINEZ_BC_GR[phase]
    c = entry['coeffs']

    x = np.atleast_1d(np.asarray(g_minus_r, dtype=float)) #

    BC = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
    if c[4] is not None:
        BC += c[4]*x**4

    return BC

#loops through content in the rows and plugs in the g-r val to the polynomial
#adds BC to apparrent mag of g filter and stores
def apply_bolometric_correction(obs_df, tp):

    df = obs_df.copy()
    df['phase_label'] = assign_phase_label(df['phase_rest'].values, tp)

    M_bol_list, BC_list = [], []

    for _, row in df.iterrows():
        BC = BC_martinez2022(row['color_gr'], phase=row['phase_label'])
        M_bol_list.append(float(row['M_g'] + BC))
        BC_list.append(float(BC))

    df['BC'] = BC_list
    df['M_bol'] = M_bol_list
    return df


#bolometric luminosity
def Mbol_to_luminosity(M_bol):
    return L_SUN * 10**(0.4 * (M_BOL_SUN - M_bol))

def Mbol_to_luminosity_with_err(M_bol, M_bol_err):
    L = Mbol_to_luminosity(M_bol)
    L_err = L * 0.4 * np.log(10) * M_bol_err
    return L, L_err


#from hw 2
def nickel_mass(L_bol, t_rest_days):
    eq1 = 3.9 * np.exp(-t_rest_days / TAU_NI)
    eq2 = 0.678 * (np.exp(-t_rest_days / TAU_CO) - np.exp(-t_rest_days / TAU_NI))
    return L_bol / (2e43 * (eq1 + eq2))

#for each target
#ensure that certain details are right within this block
def process_target(target, t_explosion):
    sn_name = target.oid
    z = target.redshift
    ra, dec = target.coordinates

    # build dataframe from target observations
    obs_df = pandas.DataFrame({
        'fid': target.fid,
        'mjd': target.mjd,
        'mag': target.mag,
    })

    #split into g and r bands, idk what this value is if its int or str

    #g_df = obs_df[obs_df['fid'] == ?][['mjd', 'mag']].rename(columns={'mag': 'mag_g'})
    #r_df = obs_df[obs_df['fid'] == ?][['mjd', 'mag']].rename(columns={'mag': 'mag_r'})

    #extinction correction, check if using unit or AoverE unit

    #distance modulus
    mu, d_L = distance_modulus(z)

    #merge g and r on nearest mjd
    merged = pandas.merge_asof(
        # g_df.sort_values('mjd'),
        # r_df.sort_values('mjd'),
        on='mjd',
        tolerance= 2.0, #only match if within 2 days as said in class
        direction='nearest').dropna(subset=['mag_g', 'mag_r'])

    #computes phase, color, absolute magnitude
    merged['phase_rest'] = compute_phase_rest(merged['mjd'].values, t_explosion, z)
    merged['color_gr'] = merged['mag_g'] - merged['mag_r']
    merged['M_g'] = merged['mag_g'] - mu

    #need to find tp using days for each observation before BC corrections since L isnt found yet
    days_temp = merged['phase_rest'].values
    mag_temp = -merged['mag_g'].values #negative since mimics luminosity instead, brighter is smaller num basically
    slope = np.diff(mag_temp) / np.diff(days_temp) #hw 2, first derivitive test
    days_slope = (days_temp[:-1] + days_temp[1:]) / 2 #midpoint
    slope_smooth = gaussian_filter1d(slope, sigma=2)

    mask = (days_slope >= PLATEAU_SEARCH_START) & (days_slope <= PLATEAU_SEARCH_END)
    local_min = np.argmin(slope_smooth[mask]) #finding min of FDT
    global_min = np.where(mask)[0][local_min]
    tp = days_slope[global_min]
    print(f"tp: {tp:.1f}")

    #most of the rest of this is closely tied from hw 2
    #apply bc
    result_df = apply_bolometric_correction(merged, tp)
    result_df['L_bol'] = Mbol_to_luminosity(result_df['M_bol'].values)

    days = result_df['phase_rest'].values
    luminosity = result_df['L_bol'].values
    interp_func = interp1d(days, luminosity, kind='cubic')

    #L50
    L50 = interp_func(50.0)
    print(f"L50 = {L50:.3e} erg/s")

    #recompute slope from bolometric luminosity for plot
    slope_bol = np.diff(luminosity) / np.diff(days)
    days_slope_bol = (days[:-1] + days[1:]) / 2
    slope_smooth_bol = gaussian_filter1d(slope_bol, sigma=2)

    mask_bol = (days_slope_bol >= PLATEAU_SEARCH_START) & (days_slope_bol <= PLATEAU_SEARCH_END)
    slope_at_tp = float(slope_smooth_bol[mask_bol][np.argmin(slope_smooth_bol[mask_bol])]) if mask_bol.sum() > 0 else np.nan

    #nickel mass
    t_tail = min(150.0, days.max())
    luminosity_at_tail = float(interp_func(t_tail))
    M_Ni = nickel_mass(luminosity_at_tail, t_tail)
    print(f"M_Ni = {M_Ni:.4e} M_sun")

    #explosion energy
    L_42 = L50 / 1e42
    t_p2 = tp / 100.0
    R_500 = 1 #500 R_sun
    a_E, b_E, c_E, d_E, e_E = GOLDBERG_EXP['coeffs']
    log_E_51 = (a_E + b_E*np.log10(L_42) + c_E*np.log10(M_Ni/0.1) + d_E*np.log10(t_p2) + e_E*np.log10(R_500))
    E_51 = 10**log_E_51
    E_exp = E_51 * 1e51
    print(f"E_exp = {E_51:.3e} * 10^51 erg")



#main w targets
