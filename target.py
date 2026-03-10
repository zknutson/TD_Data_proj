from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple
import pandas as pd
from alerce.core import Alerce
from pandas import DataFrame
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from astropy.cosmology import Planck18
import numpy as np
import astropy.units as u

Observation = NamedTuple('Observation', [
    ('fid', int),
    ('mjd', float),
    ('mag', float),
    ('abs_mag', float),
    ('obs_mag', float),
    ('obs_abs_mag', float),
    ("A_SFD", object),
])

def distance_modulus(z):
    d_L = Planck18.luminosity_distance(z).to(u.Mpc).value
    mu  = 5.0 * np.log10(d_L) + 25.0
    return mu, d_L

@dataclass
class Target:
    def __init__(self, oid, redshift, ra, dec):
        self.oid: str = oid
        self.redshift: float = redshift
        self.coordinates: tuple[str, str] = (ra, dec)
        alerce = Alerce()
        print(f"Querying lightcurve for {oid}...")
        try:
            observations = alerce.query_lightcurve(
                oid=oid,
                format="pandas"
            )
        except Exception as e:
            print(f"Error querying lightcurve for {oid}: {e}")
            self.fid = []
            self.mjd = []
            self.mag = []
            return
        detections = DataFrame(observations["detections"][0])
        self.fid: list[int] = detections["fid"]
        self.mjd: list[float] = detections["mjd"]
        self.obs_mag: list[float] = detections["magpsf"]
        dist_mod = distance_modulus(redshift)
        self.obs_abs_mag: list[float] = [obs_mag - dist_mod[0] for obs_mag in self.obs_mag]
        self.mag = [obs_mag - ext for obs_mag, ext in zip(self.obs_mag, self.extinction)]
        self.abs_mag = [obs_abs_mag - ext for obs_abs_mag, ext in zip(self.obs_abs_mag, self.extinction)]

    @cached_property
    def extinction(self):
        import astropy.coordinates as coord
        import astropy.units as u
        coords = coord.SkyCoord(self.coordinates[0], self.coordinates[1], frame='fk4',  unit=(u.hourangle, u.deg))
        table = IrsaDust.get_extinction_table(coords)  
        extinction_values = []
        for fid in self.fid:
            if fid == 1:
                #get the row where the str8 column is "DSS-II g"
                entry = table[:][table['Filter_name'] == "SDSS g"]
            elif fid == 2:
                entry = table[:][table['Filter_name'] == "SDSS r"]
            elif fid == 3:
                entry = table[:][table['Filter_name'] == "SDSS i"]
            extinction_values.append(entry[0]["A_SFD"])
        return extinction_values
    
    def grab_slice_df(self, idxs):
        return DataFrame({
            "fid": [self.fid[i] for i in idxs],
            "mjd": [self.mjd[i] for i in idxs],
            "obs_mag": [self.obs_mag[i] for i in idxs],
            "obs_abs_mag": [self.obs_abs_mag[i] for i in idxs],
            "abs_mag": [self.abs_mag[i] for i in idxs],
            "mag": [self.mag[i] for i in idxs],
            "A_SFD": [self.extinction[i] for i in idxs]
        })
    
    def __getitem__(self, idx):
        if type(idx) == int:
            return self.grab_slice_df([idx])

        if type(idx) == slice:
            start, stop, step = idx.indices(len(self.fid))
            idxs = list(range(start, stop, step))
            return self.grab_slice_df(idxs)
        
        if type(idx) == list or type(idx) == np.ndarray:
            return self.grab_slice_df(idx)
        
        if type(idx) == str:
            if idx == "g":
                return self.g_df
            elif idx == "r":
                return self.r_df
            elif idx == "i":
                return self.i_df

    @cached_property
    def g_df(self):
        idxs = [i for i, fid in enumerate(self.fid) if fid == 1]
        return self.grab_slice_df(idxs)
    
    @cached_property
    def r_df(self):
        idxs = [i for i, fid in enumerate(self.fid) if fid == 2]
        return self.grab_slice_df(idxs)

    @cached_property
    def i_df(self):
        idxs = [i for i, fid in enumerate(self.fid) if fid == 3]
        return self.grab_slice_df(idxs)
    
    def gr_df(self, tolerance: float = 2):
        for g_mjd in self.g_df["mjd"]:
            nearest_delta_mjd = min(abs(r_mjd - g_mjd) for r_mjd in self.r_df["mjd"])
            if nearest_delta_mjd > tolerance:
                continue
            

        print(gr_df)
        gr_df["delta_mjd"] = abs(gr_df["mjd_g"] - gr_df["mjd_r"])
        gr_df = gr_df[gr_df["delta_mjd"] <= tolerance]
        return gr_df

def targets_from_TNS_csv(filename, stop_early=0):
    readingtable=pd.read_csv(filename)
    groups= readingtable["Disc. Internal Name"]
    groups_nonan= groups.dropna()
    listofZTFs= []
    listofRedshifts= readingtable["Redshift"].tolist()
    listofRAs= readingtable["RA"].tolist()
    listofDecs= readingtable["DEC"].tolist()

    for i,row in enumerate(groups_nonan.tolist()):
        if "ZTF" in row:
            listofZTFs.append(row)
    
    if stop_early:
        listofZTFs = listofZTFs[:stop_early]

    targets = []
    for oid, redshift, ra, dec in zip(listofZTFs, listofRedshifts, listofRAs, listofDecs):
        target = Target(oid, redshift, ra, dec)
        targets.append(target)
    
    return targets