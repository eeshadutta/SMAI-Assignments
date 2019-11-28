import csv
with open('P1HW2_data.csv', 'rt') as f:
    data = csv.reader(f)
    rows = [r for r in data]

    minimum = 100000000
    player = 'Kumar Sangakara'
    for i in range(1, 16):
        temp = 0
        for j in range(1, 7):
            temp = temp + (float(rows[17][j]) - float(rows[i][j])) * \
                (float(rows[17][j]) - float(rows[i][j]))
        if temp < minimum:
            minimum = temp
            player = rows[i][0]
    print('Player closest to Kumar Sangakarra is : ', player)

    minimum = 100000000
    player = 'David Warner'
    for i in range(1, 16):
        temp = 0
        for j in range(1, 7):
            temp = temp + (float(rows[18][j]) - float(rows[i][j])) * \
                (float(rows[18][j]) - float(rows[i][j]))
        if temp < minimum:
            minimum = temp
            player = rows[i][0]
    print('Player closest to David Warner is : ', player)

    minimum = 100000000
    player = 'Mitchell Starc'
    for i in range(1, 16):
        temp = 0
        for j in range(1, 7):
            temp = temp + (float(rows[19][j]) - float(rows[i][j])) * \
                (float(rows[19][j]) - float(rows[i][j]))
        if temp < minimum:
            minimum = temp
            player = rows[i][0]
    print('Player closest to Mitchell Starc is : ', player)

    arr_age = [float(rows[i][1]) for i in range(1, 16)]
    arr_height = [float(rows[i][2]) for i in range(1, 16)]
    arr_role = [float(rows[i][3]) for i in range(1, 16)]
    arr_bat_avg = [float(rows[i][4]) for i in range(1, 16)]
    arr_bowl_avg = [float(rows[i][5]) for i in range(1, 16)]
    arr_matches = [float(rows[i][6]) for i in range(1, 16)]

    var_age = numpy.var(arr_age)
    var_height = numpy.var(arr_height)
    var_role = numpy.var(arr_role)
    var_bat_avg = numpy.var(arr_bat_avg)
    var_bowl_avg = numpy.var(arr_bowl_avg)
    var_matches = numpy.var(arr_matches)

    arr_age = [arr_age[i]/var_age for i in range(1, 16)]
    arr_height = [arr_height[i]/var_height for i in range(1, 16)]
    arr_role = [arr_role[i]/var_role for i in range(1, 16)]
    arr_bat_avg = [arr_bat_avg[i]/var_bat_avg for i in range(1, 16)]
    arr_bowl_avg = [arr_bowl_avg[i]/var_bowl_avg for i in range(1, 16)]
    arr_matches = [arr_matches[i]/var_matches for i in range(1, 16)]
