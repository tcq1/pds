import math

def euclid_c(list x,list y):
    result = []
    for p1 in x:
        for p2 in y:
            result.append(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    return result