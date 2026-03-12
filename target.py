from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from alerce.core import Alerce
from pandas import DataFrame
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from astropy.cosmology import Planck18
import numpy as np
import astropy.units as u
from tqdm import tqdm

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
        retry_count = 5
        for attempt in range(retry_count):
            try:
                observations = alerce.query_lightcurve(
                    oid=oid,
                    format="pandas"
                )
                break  # Exit the loop if the query was successful
            except Exception as e:
                print(f"Error querying lightcurve for {oid} (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
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
    
    @cached_property
    def color_gr_df(self):
        gr_df = DataFrame({
            "mjd": [],
            "color_gr": [],
            "abs_mag_g": [],
        })
        for g_obs in self.g_df.itertuples():
            for r_obs in self.r_df.itertuples():
                if abs(g_obs.mjd - r_obs.mjd) < 5:
                    gr_df.loc[len(gr_df)] = {
                        "mjd": g_obs.mjd,
                        "color_gr": g_obs.obs_mag - r_obs.obs_mag,
                        "abs_mag_g": g_obs.obs_abs_mag
                    }
                    break
        return gr_df

def _extract_tns_rows(filename, stop_early=0, start_at_end=False):
    readingtable = pd.read_csv(filename)
    filtered = readingtable[
        readingtable["Disc. Internal Name"].fillna("").str.contains("ZTF")
    ]

    if start_at_end:
        filtered = filtered.iloc[::-1]
    if stop_early:
        filtered = filtered.iloc[:stop_early]

    return list(
        filtered[["Disc. Internal Name", "Redshift", "RA", "DEC"]]
        .itertuples(index=False, name=None)
    )


def targets_from_TNS_csv(
    filename,
    stop_early=0,
    start_at_end=False,
    use_threads=False,
    max_workers=None,
    show_progress=True,
):
    rows = _extract_tns_rows(filename, stop_early=stop_early, start_at_end=start_at_end)

    if not use_threads:
        return [Target(oid, redshift, ra, dec) for oid, redshift, ra, dec in rows]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(Target, oid, redshift, ra, dec): i
            for i, (oid, redshift, ra, dec) in enumerate(rows)
        }
        results = [None] * len(rows)

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Loading targets", unit="target")

        for future in iterator:
            idx = futures[future]
            results[idx] = future.result()

        return results


async def targets_from_TNS_csv_async(
    filename,
    stop_early=0,
    start_at_end=False,
    max_workers=None,
    show_progress=True,
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: targets_from_TNS_csv(
            filename,
            stop_early=stop_early,
            start_at_end=start_at_end,
            use_threads=True,
            max_workers=max_workers,
            show_progress=show_progress,
        ),
    )