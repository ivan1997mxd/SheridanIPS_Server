def get_zone(zone_list, nnv4_coordinate):

    x = nnv4_coordinate[0]
    y = nnv4_coordinate[1]

    for zone in zone_list:
        if zone.contains(x, y):
            return zone

    raise Exception("Zone could not be found for co-ordinates {}".format(nnv4_coordinate))
