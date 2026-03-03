from dataclasses import dataclass
import alerce
from pandas import DataFrame


@dataclass
class Target:
    oid: str
    redshift: float
    fid: list[int]
    mjd: list[float]
    mag: list[float]

    @classmethod
    def from_oid_rsft(cls, oid, redshift):
        observations = alerce.query_lightcurve(
            oid=oid,
            format="pandas"
        )
        detections = DataFrame(observations["detections"][0])
        fid = detections["fid"]
        mjd = detections["mjd"]
        mag = detections["magpsf"]
        return cls(oid, redshift, fid, mjd, mag)
    