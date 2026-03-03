from dataclasses import dataclass
from turtle import pd
import alerce
from pandas import DataFrame
from astroquery.ipac.irsa.irsa_dust import IrsaDust

@dataclass
class Target:
    def __init__(self, oid, redshift, ra, dec):
        self.oid: str = oid
        self.redshift: float = redshift
        self.coordinates: tuple[float, float] = (ra, dec)
        observations = alerce.query_lightcurve(
            oid=oid,
            format="pandas"
        )
        detections = DataFrame(observations["detections"][0])
        self.fid: list[int] = detections["fid"]
        self.mjd: list[float] = detections["mjd"]
        self.mag: list[float] = detections["magpsf"]

    @property
    def extinction(self):
        result = IrsaDust.get_query_table(
            self.coordinates,
            radius=0.1,
            catalog="dr5"
        )

def targets_from_TNS_csv(filename):
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
    
    targets = []
    for oid, redshift, ra, dec in zip(listofZTFs, listofRedshifts, listofRAs, listofDecs):
        target = Target(oid, redshift, ra, dec)
        targets.append(target)
    
    return targets