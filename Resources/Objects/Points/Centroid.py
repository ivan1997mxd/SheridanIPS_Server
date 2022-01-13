import csv
from Resources.Objects.Points.GridPoint_RSSI import GridPoint
from Resources.Objects.Points.Point import Point


class Centroid(Point):
    def __init__(self, central: GridPoint, top_left: GridPoint,
                 top_right: GridPoint, bottom_left: GridPoint, bottom_right: GridPoint):
        super(Centroid, self).__init__(id=central.id, num=central.num, x=central.x, y=central.y, z=central.z)

        self.Center: GridPoint = central
        self.TopLeft: GridPoint = top_left
        self.TopRight: GridPoint = top_right
        self.BottomLeft: GridPoint = bottom_left
        self.BottomRight: GridPoint = bottom_right
        self.CornerPoints = [self.TopLeft, self.TopRight, self.BottomLeft, self.BottomRight]

    def __str__(self) -> str:
        Str = "ID: " + str(self.Center.id) + "\n"
        Str += "TopLeft: " + str(self.TopLeft.id) + "\n"
        Str += "TopRight: " + str(self.TopRight.id) + "\n"
        Str += "BottomLeft: " + str(self.BottomLeft.id) + "\n"
        Str += "BottomRight: " + str(self.BottomRight.id) + "\n"
        return Str


    def sort_corner_points_by_distance(self) -> None:
        # Selection sort
        for i in range(len(self.CornerPoints)):
            min_index = i
            for j in range(i + 1, len(self.CornerPoints)):
                if self.CornerPoints[min_index] > self.CornerPoints[j]:
                    min_index = j

            self.CornerPoints[i], self.CornerPoints[min_index] = self.CornerPoints[min_index], self.CornerPoints[i]

    # endregion

    # region Override Methods
    def _reset(self):
        self.CornerPoints = [self.TopLeft, self.TopRight, self.BottomLeft, self.BottomRight]

    @classmethod
    def create_point_list(cls, file_path: str, *args, **kwargs) -> list:
        assert len(args) == 0, "Centroids are unable to accept variable arguments."
        assert "grid_points" in kwargs.keys(), "You must pass a list of Grid Points."

        grid_points = kwargs["grid_points"]
        points = list()

        with open(file_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")

            for line in readCSV:
                central = grid_points[int(line[0]) - 1]
                top_left = grid_points[int(line[1]) - 1]
                top_right = grid_points[int(line[2]) - 1]
                bottom_left = grid_points[int(line[3]) - 1]
                bottom_right = grid_points[int(line[4]) - 1]

                points.append(Centroid(central, top_left, top_right, bottom_left, bottom_right))

        return points

    # endregion
    @classmethod
    def create_point(cls, cd: dict, gp_list: list):
        central = gp_list[int(cd['central']) - 1]
        top_left = gp_list[int(cd['top_left']) - 1]
        top_right = gp_list[int(cd['top_right']) - 1]
        bottom_left = gp_list[int(cd['bottom_left']) - 1]
        bottom_right = gp_list[int(cd['bottom_right']) - 1]

        points = Centroid(central, top_left, top_right, bottom_left, bottom_right)

        return points
