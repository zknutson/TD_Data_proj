from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple
import pandas as pd
from alerce.core import Alerce
from pandas import DataFrame
from astroquery.ipac.irsa.irsa_dust import IrsaDust

Observation = NamedTuple('Observation', [
    ('redshift', float),
    ('fid', int),
    ('mjd', float),
    ('mag', float),
    ("A_SFD", object),
])

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
        self.mag: list[float] = detections["magpsf"]

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
    
    def __getitem__(self, idx):
        return Observation(
            self.redshift,
            self.fid[idx],
            self.mjd[idx],
            self.mag[idx],
            self.extinction[idx]
        )

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