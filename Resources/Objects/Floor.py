from time import time
from typing import Dict, List, Tuple
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Algorithms.svm.svm import svm_model
from Resources.Objects.Matrices.NormalizedDistribution import sort_matrices, NormalizedMatrix
from Resources.Objects.TestData import create_from_db, Sample, create_from_db_new
from Resources.Objects.Zone import Zone
from Algorithms.dijkstra import Graph

testActuals = [[0.3, 0.3], [0.2, 1.0], [0.4, 1.7], [1.1, 1.1], [1.8, 0.3],
               [2.2, 0.1], [2.2, 1.1], [2.9, 1.8], [3, 0.8], [3.6, 1.4],
               [4.3, 1.5], [4.2, 2.2], [2.3, 2.2], [3, 2.8], [2.2, 3],
               [2.8, 3.4], [2.6, 4.2], [3.3, 3.9], [3.3, 4.7], [2.1, 5],
               [2.5, 5.6], [2.5, 6.5], [3, 6.9], [2.5, 7.8], [2, 7.1],
               [1.2, 7.8], [0.9, 7], [0.2, 7.7], [0.2, 6.3], [1.5, 6.3],
               [1.6, 5.9], [0.5, 5.9], [1.1, 5], [0.2, 5], [1, 4.2],
               [1.6, 4], [1.1, 3.4], [0.2, 4], [0.2, 3], [3.4, 5.6],
               [3.6, 6.8], [3.6, 7.8], [4.6, 7.8], [5.5, 7.8], [4.3, 7],
               [5.5, 7], [5, 6.5], [4.1, 5.9], [5.1, 5.9], [5.5, 5.2],
               [4.5, 5.2], [5, 4.9], [4.3, 3.9], [5.6, 3.8], [4.8, 3.2],
               [5.5, 2.2], [5.6, 1.3], [4.8, 1], [5.9, 0.8], [5.3, 0.2]]

testActuals1 = [[[0.3, 0.3], [1.8, 0.3], [1.1, 1.1], [0.4, 1.7], [1.85, 1.7]],
                [[0.2, 3.1], [0.2, 3.9], [1.05, 3.6], [1.8, 3.2], [1.8, 3.9]],
                [[1.2, 4.1], [0.1, 5], [1.8, 4.8], [0.2, 5.8], [1.6, 5.8]],
                [[0.95, 6.2], [0.5, 6.95], [1.3, 6.9], [0.4, 7.6], [1.5, 7.6]],
                [[0.5, 8.6], [1.1, 8.8], [0.2, 9.7], [1.1, 9.8], [1.8, 9.7]],
                [[3, 1.95], [3.9, 1.9], [3.9, 1.4], [3.5, 0.2], [2.4, 0.2]],
                [[2.8, 2.3], [3.4, 3.8], [2.2, 3], [2.3, 3.8], [3.4, 3]],
                [[2.2, 4.3], [3.05, 4.3], [3.7, 4.9], [2.6, 5.5], [3.8, 5.6]],
                [[2.1, 6.2], [3.2, 6.2], [3.9, 7], [3.2, 7.7], [2.2, 7.4]],
                [[2.3, 8.9], [2.4, 9.6], [3.2, 8.6], [2.6, 8.2], [3.4, 9.8]],
                [[5.9, 0.3], [5.1, 0.9], [5.9, 1.4], [5.1, 1.8], [4.5, 1.5]],
                [[4.6, 2.2], [5.8, 2.3], [5.5, 3], [5.1, 3.3], [5.9, 3.8]],
                [[4.9, 4.3], [4.9, 5], [5.5, 4.5], [4.4, 5.6], [5.5, 5.4]],
                [[4.4, 6.5], [5.8, 7.5], [5.4, 6.6], [4.6, 7.6], [5.05, 7.2]],
                [[4.2, 9.1], [4.8, 9.8], [5.1, 8.9], [4.95, 8.2], [5.6, 9.5]]]


class Floor:
    def __init__(self, floor_id: str, access_points: List[AccessPoint], grid_points: List[GridPoint],
                 zones: List[Zone] = None, data: List[Sample] = None, random_data: List[Sample] = None,
                 matrix: NormalizedMatrix = None,
                 model: svm_model = None):
        self.floor_id: str = floor_id
        self.access_points: List[AccessPoint] = access_points
        self.zones: List[Zone] = zones
        self.grid_points: List[GridPoint] = grid_points
        self.matrix: NormalizedMatrix = matrix
        self.model: svm_model = model
        self.data: List[Sample] = data
        self.random_data: List[Sample] = random_data

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

    @property
    def get_grid_points_only(self):
        cps = self.get_centroids
        gp_only = cps[0].CornerPoints
        for cp in cps:
            for cnp in cp.CornerPoints:
                if cnp in gp_only:
                    pass
                else:
                    gp_only.append(cnp)

        return gp_only

    @property
    def get_graph(self):
        g = Graph()
        edge_list = []
        for z in self.zones:
            g.add_vertex(z.zone_num)
            distances = z.distance
            for key, value in distances.items():
                edge_list.append((z.zone_num, key, value))
        for edge in edge_list:
            g.add_edge(edge[0], edge[1], edge[2])
        return g

    def find_zone(self, name):
        for zone in self.zones:
            if zone.zone_num == name:
                return zone

    def find_gp(self, id):
        for gp in self.grid_points:
            if gp.num == id:
                return gp

    @classmethod
    def create_floor_list(cls, floor_list: list, access_points: List[AccessPoint]):
        floors = list()
        for f in floor_list:
            floor_id = f['floor_id']
            aps = [ap for ap in access_points if ap.id in list(f['Access_Points'])]
            # gps = GridPoint.create_point_list_db(f['Grid_Points'], aps)
            gps = GridPoint.create_point_list_db_new(f['Grid_Points'], aps)
            zones = Zone.create_point_list(list(f['Zones']), gps)
            start_time = time()
            data = create_from_db(access_points, zones, list(f['Data']))
            random_data = create_from_db(access_points, zones, list(f['Random_Data']), testActuals1)
            end_time = time()
            print("Sample was created: {}s.".format(end_time - start_time))
            floors.append(Floor(floor_id=floor_id, access_points=aps, grid_points=gps, zones=zones, data=data,
                                random_data=random_data))

        return floors

    @classmethod
    def create_floor_list_new(cls, floor_list: list, access_points: List[AccessPoint]):
        floors = list()
        for f in floor_list:
            floor_id = f['floor_id']
            aps = [ap for ap in access_points if ap.id in list(f['Access_Points'])]
            gps = GridPoint.create_point_list_db_new(f['Grid_Points'], aps)
            start_time = time()
            data = create_from_db_new(access_points, gps, list(f['Data']))
            random_data = create_from_db_new(access_points, gps, list(f['Random_Data']), testActuals)
            end_time = time()
            print("Sample was created: {}s.".format(end_time - start_time))
            floors.append(Floor(floor_id=floor_id, access_points=aps, grid_points=gps, data=data,
                                random_data=random_data))

        return floors
