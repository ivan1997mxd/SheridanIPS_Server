from typing import Dict, List, Tuple

from Resources.Objects.Floor import Floor
from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Resources.Objects.Points.Centroid import Centroid
from Resources.Objects.Matrices.NormalizedDistribution import sort_matrices, NormalizedMatrix


# class Zone:
#     def __init__(self, zone_num: str, room_id: str, points: Centroid):
#         self.zone_num: str = zone_num
#         self.room_id: str = room_id
#         self.points: Centroid = points
#
#     @property
#     def num(self) -> int:
#         num = int(self.zone_num[-1])
#         return num
#
#     @classmethod
#     def create_point_list(cls, zone_list: list, gp_list: list):
#         zones = list()
#         for z in zone_list:
#             zone_num = z["zone_Num"]
#             room_id = z['room_id']
#             points = Centroid.create_point(z['points'], gp_list)
#             zones.append(Zone(zone_num=zone_num, room_id=room_id, points=points))
#         return zones
#
#     def contains(self, point: Tuple[float, float]) -> bool:
#         x = point[0]
#         y = point[1]
#         if self.points.TopLeft.x <= x <= self.points.BottomRight.x:
#             if self.points.TopLeft.y <= y <= self.points.BottomRight.y:
#                 return True
#         return False
#
#
# class Floor:
#     def __init__(self, floor_id: str, access_points: List[AccessPoint], grid_points: List[GridPoint],
#                  zones: List[Zone], matrix: NormalizedMatrix = None):
#         self.floor_id: str = floor_id
#         self.access_points: List[AccessPoint] = access_points
#         self.zones: List[Zone] = zones
#         self.grid_points: List[GridPoint] = grid_points
#         self.matrix: NormalizedMatrix = matrix
#
#     @property
#     def total_zone(self) -> int:
#         return len(self.zones)
#
#     @property
#     def ap_list(self) -> list:
#         return [d.id for d in self.access_points]
#
#     @property
#     def get_centroids(self):
#         centroids = [z.points for z in self.zones]
#         return centroids
#
#     @classmethod
#     def create_floor_list(cls, floor_list: list):
#         floors = list()
#         for f in floor_list:
#             floor_id = f['floor_id']
#             aps = AccessPoint.create_db_point_list(list(f['Access_Points']))
#             gps = GridPoint.create_point_list_db(f['Grid_Points'], aps)
#             zones = Zone.create_point_list(list(f['Zones']), gps)
#             floors.append(Floor(floor_id=floor_id, access_points=aps, grid_points=gps, zones=zones))
#
#         return floors


class Building:
    def __init__(self, building_name: str, floors: List[Floor], access_points: List[AccessPoint]):
        self.building_name: str = building_name
        self.floors: List[Floor] = floors
        self.access_points = access_points

    @property
    def zones(self) -> list:
        zones = list()
        for f in self.floors:
            zones += f.zones
        return zones

    @property
    def data(self) -> list:
        data_list = list()
        for f in self.floors:
            data_list += f.data
        return data_list

    @classmethod
    def create_building_list(cls, building_list: list):
        buildings = list()
        for b in building_list:
            building_name = b['building_name']
            if building_name == "Condo_Ble":
                access_points = AccessPoint.create_db_point_list(list(b['Access_Points']))
                floors = Floor.create_floor_list(list(b['floors']), access_points)
                # floors = Floor.create_floor_list_new(list(b['floors']), access_points)
                buildings.append(Building(building_name=building_name, floors=floors, access_points=access_points))

        return buildings
