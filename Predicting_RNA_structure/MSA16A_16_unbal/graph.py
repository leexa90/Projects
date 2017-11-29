f1= open('./updates.log','r')
f2= open('./updates.log','r')
f3= open('./updates.log','r')
import numpy as np
x1 = [[],[]]
for line in f1:
    if '.' in line:
        temp = map(np.float, line.split())
        x1[0] += [temp[0],]
        x1[1] += [temp[1],]

x2 = [[],[]]
for line in f2:
    if '.' in line:
        temp = map(np.float, line.split())
        x2[0] += [temp[0],]
        x2[1] += [temp[1],]

x3 = [[],[]]
for line in f3:
    if '.' in line:
        temp = map(np.float, line.split())
        x3[0] += [temp[0],]
        x3[1] += [temp[1],]
import matplotlib.pyplot as plt

all = [[],[]]
plot = []
for i in [x1,x2,x3][0:3]:
    all = [[],[]]
    for j in range(10,len(i[0]),2):
        all[0] +=  [(j+1)//2,]
        all[1] +=  [[i[1][j+1],i[1][j],(j+1)//2],]
    print sorted(all[1], key = lambda x : 8*x[0]+0*x[1])[-10:]
    plot += [[x[0] for x in sorted(all[1], key = lambda x : 10*x[0]+0*x[1])[-90:]],]
plt.hist(plot,bins=40,range=(0.785,0.825),label=['10','11','12']);plt.legend();plt.show()
all = [[],[]]
for i in [x1,x2,x3][0:3]:
    for j in range(0,len(i[0]),2):
        all[0] +=  [i[1][j],]
        all[1] +=  [i[1][j+1],]
#print sorted(all[1], key = lambda x : x[0]+x[1])
plt.plot(all[0],all[1],'ro');plt.show()
die
plt.plot(all[0],all[1],'ro');plt.show()
plt.plot(range(len(x1[0][0::2])),x1[0][0::2],'r');
plt.plot(range(len(x1[0][1::2])),x1[0][1::2],'b');

plt.plot(range(len(x2[0][0::2])),x2[0][0::2],'m');
plt.plot(range(len(x2[0][1::2])),x2[0][1::2],'c');

plt.plot(range(len(x3[0][0::2])),x3[0][0::2],'orange');
plt.plot(range(len(x3[0][1::2])),x3[0][1::2],'green');
plt.show()

#plt.plot(range(len(x1[1][0::2])),x1[1][0::2],'r');
plt.plot(range(len(x1[1][1::2])),x1[1][1::2],'b');

#plt.plot(range(len(x2[1][0::2])),x2[1][0::2],'m');
plt.plot(range(len(x2[1][1::2])),x2[1][1::2],'c');

#plt.plot(range(len(x3[1][0::2])),x3[1][0::2],'orange');
plt.plot(range(len(x3[1][1::2])),x3[1][1::2],'green');
plt.show()


