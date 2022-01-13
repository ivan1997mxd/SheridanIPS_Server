import math
from Resources.Objects.Points.GridPoint_RSSI import GridPoint


# This class implements components of the Line-Based Algorithm.
class LineBasedAlgorithm:

    # This method moves centroid S closer to the closest grid point and
    # calculates the midpoint between the new S, the midpoint M, and the closest
    # grid point.
    def compute_lba(self, s, m, closest_gps, shiftPercent, moveToMidpoint):

        # Either we need to move the centroid towards the grid point or the
        # midpoint.
        if (moveToMidpoint):
            newCentroid = self.move_pt(s, m, shiftPercent)
        else:
            newCentroid = self.move_pt(s, closest_gps[0], shiftPercent)

        # Calculate the final centroid.
        centroidQX = round(((closest_gps[0].x + newCentroid.x + m.x) / 3), 1)
        centroidQY = round(((closest_gps[0].y + newCentroid.y + m.y) / 3), 1)

        return [centroidQX, centroidQY]

    # This method will move the midpoint closer to the grid point or centroid
    # and calculate the result.
    def compute_lba_mid(self, s, m, closest_gps, shiftPercent, moveToGridpoint):

        # Either we need to move the midpoint towards the grid point or the
        # centroid.
        if (moveToGridpoint):
            newMidpoint = self.move_pt(m, closest_gps[0], shiftPercent)
        else:
            newMidpoint = self.move_pt(m, s, shiftPercent)

        # Calculate the final centroid.
        centroidQX = round(((closest_gps[0].x + newMidpoint.x + s.x) / 3), 1)
        centroidQY = round(((closest_gps[0].y + newMidpoint.y + s.y) / 3), 1)

        return [centroidQX, centroidQY]

    # This method will move the grid point closer to centroid s.
    def compute_lba_grid(self, s, m, closest_gps, shiftPercent, moveToMidpoint):

        # Either we need to move towards the centroid or the
        # midpoint.
        if (moveToMidpoint):
            newGridpoint = self.move_pt(closest_gps[0], m, shiftPercent)
        else:
            # Get the new grid point.
            newGridpoint = self.move_pt(closest_gps[0], s, shiftPercent)

        # Calculate the final centroid.
        centroidQX = round(((s.x + newGridpoint.x + m.x) / 3), 1)
        centroidQY = round(((s.y + newGridpoint.y + m.y) / 3), 1)

        return [centroidQX, centroidQY]

    # This method moves multiple points
    def compute_lba_multiple(self, s, m, closest_gps, shiftPercent, pointTo):
        closest = closest_gps[0]
        if s.point == closest.point or m.point == closest.point:
            closest = closest_gps[1]

        if (pointTo == 'm'):
            newCentroid = self.move_pt(s, m, shiftPercent)
            newGridpoint = self.move_pt(closest, m, shiftPercent)

            # Calculate the final centroid.
            centroidQX = round(((newGridpoint.x + newCentroid.x + m.x) / 3), 1)
            centroidQY = round(((newGridpoint.y + newCentroid.y + m.y) / 3), 1)

            return [centroidQX, centroidQY]

        elif (pointTo == 's'):
            newMidpoint = self.move_pt(m, s, shiftPercent)
            newGridpoint = self.move_pt(closest, s, shiftPercent)

            # Calculate the final centroid.
            centroidQX = round(((newGridpoint.x + s.x + newMidpoint.x) / 3), 1)
            centroidQY = round(((newGridpoint.y + s.y + newMidpoint.y) / 3), 1)

            return [centroidQX, centroidQY]

        else:
            newCentroid = self.move_pt(s, closest, shiftPercent)
            newMidpoint = self.move_pt(m, closest, shiftPercent)

            # Calculate the final centroid.
            centroidQX = round(((closest.x + newMidpoint.x + newCentroid.x) / 3), 1)
            centroidQY = round(((closest.y + newMidpoint.y + newCentroid.y) / 3), 1)

            return [centroidQX, centroidQY]

    # This will move a point to another by a certain percentage.
    # pointFrom and pointTo are Gridpoint objects.
    def move_pt(self, pointFrom, pointTo, shiftPercent):
        dist_bw_gp_s = self.getDistanceBetweenPoints(pointFrom, pointTo)

        # print ("Move Distance", dist_bw_gp_s)
        dist_tbm = (shiftPercent / 100) * dist_bw_gp_s

        pointVx = (pointFrom.x + pointTo.x) / 2
        pointVy = (pointFrom.y + pointTo.y) / 2

        ratio_bw_d_dt = dist_tbm / dist_bw_gp_s

        newFromX = round(((1 - ratio_bw_d_dt) * pointFrom.x + (ratio_bw_d_dt * pointTo.x)), 1)
        newFromY = round(((1 - ratio_bw_d_dt) * pointFrom.y + (ratio_bw_d_dt * pointTo.y)), 1)

        return GridPoint("new", 0, {}, newFromX, newFromY)

    # This method calculates the distance between two points.
    # pointA and pointB are Gridpoint objects.
    def getDistanceBetweenPoints(self, pointA, pointB):
        valuesX = math.pow(pointB.x - pointA.x, 2)
        valuesY = math.pow(pointB.y - pointA.y, 2)
        distance = math.sqrt(valuesX + valuesY)
        # print("XB", pointB.x)
        # print("YB", pointB.y)
        # print("XA", pointA.x)
        # print("YA", pointA.y)
        return distance

# For testing purposes only.

# gpas = [GridPoint(1, 0, 4, []), GridPoint(2, 0, 0, [])]
# s = GridPoint(3, 2, 2, []);
# lba = LineBasedAlgorithm()
# computed = lba.compute_lba(s, gpas, 20);

# print(computed[0])
# print(computed[1])
