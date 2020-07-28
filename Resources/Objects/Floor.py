from time import time
from typing import Dict, List, Tuple
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Algorithms.svm.svm import svm_model
from Resources.Objects.Matrices.NormalizedDistribution import sort_matrices, NormalizedMatrix
from Resources.Objects.TestData import create_from_db, Sample
from Resources.Objects.Zone import Zone


class Floor:
    def __init__(self, floor_id: str, access_points: List[AccessPoint], grid_points: List[GridPoint],
                 zones: List[Zone], data: List[Sample] = None, matrix: NormalizedMatrix = None,
                 model: svm_model = None):
        self.floor_id: str = floor_id
        self.access_points: List[AccessPoint] = access_points
        self.zones: List[Zone] = zones
        self.grid_points: List[GridPoint] = grid_points
        self.matrix: NormalizedMatrix = matrix
        self.model: svm_model = model
        self.data: List[Sample] = data

    @property
    def total_zone(self) -> int:
        return len(self.zones)

    @property
    def ap_list(self) -> list:
        return [d.id for d in self.access_points]

    @property
    def get_centroids(self):
        centroids = [z.points for z in self.zones]
        return centroids

    @classmethod
    def create_floor_list(cls, floor_list: list):
        floors = list()
        for f in floor_list:
            floor_id = f['floor_id']
            aps = AccessPoint.create_db_point_list(list(f['Access_Points']))
            gps = GridPoint.create_point_list_db(f['Grid_Points'], aps)
            zones = Zone.create_point_list(list(f['Zones']), gps)
            start_time = time()
            data = create_from_db(aps, zones, list(f['Data']))
            end_time = time()
            print("Sample was created: {}s.".format(end_time - start_time))
            floors.append(Floor(floor_id=floor_id, access_points=aps, grid_points=gps, zones=zones, data=data))

        return floors
