def get_KNNzone(zone_list, knn_coordinate):

    x = knn_coordinate[0]
    y = knn_coordinate[1]

    for zone in zone_list:
        if zone.contains(x, y):
            return zone

    raise Exception("Zone could not be found for co-ordinates {}".format(knn_coordinate))
