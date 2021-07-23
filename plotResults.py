import matplotlib.pyplot as plt 
import numpy as np
if __name__ == "__main__":
    
    f  = open("scan.txt", "r")
    content = f.readlines()
    
    sb = np.array([[float(i.split(" ")[0]), float(i.split(" ")[1])] for i in content])
    bench = sb[0]
    sb = sb[1:]
    
    x = [i[0] for i in sb]
    y = [i[1] for i in sb]
    
    ab = [i for i in sb if i[1] > bench[1]]
    ab = sorted(ab, key=lambda x: x[1])
    
    print(ab)
    print(bench)
    
    fig = plt.figure(figsize=(10,10))
    plt.scatter(x,y)
        
    plt.scatter(bench[0], bench[1], marker="x")
    
    fig.savefig("res.pdf")
        