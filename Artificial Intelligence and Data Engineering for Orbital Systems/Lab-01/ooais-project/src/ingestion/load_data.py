with open("data/raw/observations.csv") as f:
    lines = f.readlines()
    
print("Number of records:", len(lines)-1)

objects = [line.split(",")[1] for line in lines[1:]]
print("Objects:", set(objects))