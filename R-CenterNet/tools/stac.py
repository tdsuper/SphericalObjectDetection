res = {}

with open('records.txt') as f:
    for line in f:
        c = int(line[:-1])
        if c in res.keys():
            res[c] += 1
        else:
            res[c] = 1

print(res)
