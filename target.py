from dataclasses import dataclass
import pandas as pd
from alerce.core import Alerce
from pandas import DataFrame
from astroquery.ipac.irsa.irsa_dust import IrsaDust

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

    @property
    def extinction(self):
        import astropy.coordinates as coord
        import astropy.units as u
        coords = coord.SkyCoord(self.coordinates[0], self.coordinates[1], frame='fk4',  unit=(u.hourangle, u.deg))
        table = IrsaDust.get_extinction_table(coords)  
        return table

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