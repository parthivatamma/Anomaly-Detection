import numpy as np

index=1
f = open("sampleSet.csv", "w")
f.write("Overs,Scores\n")
randomArray = np.random.randint(36,size=(20))
for x in randomArray:
    cheese = str(index) + "," + str(x) + "\n"
    f.write(cheese)
    index+=1
