import matplotlib.pyplot as plt

with open("debug.txt","r+") as f:
    raw_text = f.read()

times = []
n_free = []
lines = raw_text.split("\n")
for line in lines: 
    if "||" in line:
        splitted_line = line.split("||")
        times.append(int(splitted_line[0]))
        n_free.append(int(splitted_line[1]))

plt.figure()
plt.subplot("211")
plt.plot(times)
plt.subplot("212")
plt.plot(n_free)

plt.show()