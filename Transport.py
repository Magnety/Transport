import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cvxpy as cvx
np.set_printoptions(linewidth=400)

#m denotes the num of factories and n denotes the num of warehouses
m=6
n=5

#pi denotes the location of factory[i] and qj denotes the location of warehouse[j]
pi = np.random.randint(10,90,2*m).reshape((m,2))
qj = np.random.randint(10,90,2*n).reshape((n,2))

#xi denotes the yield of factory[i] and yj denotes the capacity of warehouse[j]
xi = np.random.randint(0,100,m)
sum_xi = np.sum(xi)
yj = np.random.randint(0,100,n)

#randomly init the yj to confirm sum(xi)==sum(yj)
while sum_xi-np.sum(yj[0:n-1])<0:
    yj = np.random.randint(0, 100, n)
yj[n-1] = sum_xi-np.sum(yj[0:n-1])

print("每个工厂的产量:")
print(xi)
print("每个仓库的储量:")
print(yj)
#Dis[i,j] denotes the distance between factory[i] and warehouse[j]
Dis = np.empty((m,n),dtype=float)
for i in range(m):
    for j in range(n):
        Dis[i][j] = np.sqrt(np.square(pi[i][0]-qj[j][0])+np.square(pi[i][1]-qj[j][1]))

#C[i,j] denotes the products from factory[i] to warehouse[j]
C = cvx.Variable((m,n))

#Minimize the sum(C[i][j]*Dis[i][j])
#obj = cvx.Minimize(cvx.sum(Dis*C))
obj = cvx.Minimize(cvx.sum(cvx.multiply(Dis,C)))

#constraints: C[i][j]>=0 ; sum(C[i][:]) = xi[i] ; sum(C[:][j]) = yj[j]
constraints = []
for i in range(m):
    constraints += [
        cvx.sum(C[i,:])==xi[i]
    ]
for j in range(n):
    constraints += [
        cvx.sum(C[:, j]) == yj[j]
    ]
for i in range(m):
    for j in range(n):
        constraints +=[
            C[i,j]>=0
        ]

prob = cvx.Problem(obj,constraints)
print("Optimal value",prob.solve())
print(C.value)
result =  np.rint(C.value)
print(result)


print("工厂：")
for i in range(m):
    print("No:%s->loc:(%s,%s)->yield:%s"%(i,pi[i,0],pi[i,1],xi[i]))
print("仓库：")
for j in range(n):
    print("No:%s->loc:(%s,%s)->capacity:%s"%(j,qj[j,0],qj[j,1],yj[j]))
fig1 = plt.figure('Factories and Warehouses Information',figsize=(10,10))
def show_gen_loc():
    plt.scatter(pi[:, 0], pi[:, 1], s=40 ,marker='x',label='factory')
    plt.scatter(qj[:, 0], qj[:, 1], s=40 ,marker='^',label='warehouse')
    for j in range(n):
        plt.annotate("%s" % yj[j],
                    xy=((qj[j,0]+4), (qj[j,1]+4)),
                    xytext=((qj[j,0]), (qj[j,1])),
                    color="b",
                    )
    for j in range(m):
        plt.annotate("%s" % xi[j],
                    xy=((pi[j,0]+4), (pi[j,1]+4)),
                    xytext=((pi[j,0]), (pi[j,1])),
                    color="r",
                    )
show_gen_loc()
plt.legend()
plt.show()
fig,ax = plt.subplots(figsize=(10, 10))
dot =[]
_dot =[]
color =['r','g','b','c','violet','lightsalmon','sage','skyblue','y','pink']
linestyle = ['o','o','o','o','o','o','o','o','o','o']
for i in range(m):
    for j in range(n):
        exec("dot%s%s,=ax.plot([], [],color = color[%s],marker = linestyle[%s], animated=False)" % (i,j,i,j))
#初始化点
def init():
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    for i in range(m):
        for j in range (n):
            exec("_dot.append(dot%s%s)" % (i,j))
    return _dot
#生成动态的点
def gen_dot():
    for i in range(100):
        dot.clear()
        for k in range(m):
            for j in range(n):
                if (result[k][j] > 0):
                    x_frame = np.linspace(pi[k, 0], qj[j, 0], 100)
                    y_frame = np.linspace(pi[k, 1], qj[j, 1], 100)
                    newdot = [x_frame[i], y_frame[i]]
                    dot.append(newdot)
        yield dot
#为animation更新点
def update_dot(newd):
    _dot.clear()
    i = 0
    for k in range(m):
        for j in range(n):
            if (result[k][j] > 0):
                exec("dot%s%s.set_data(newd[i][0], newd[i][1])" % (k,j))
                exec("_dot.append(dot%s%s)" % (k,j))
                i += 1
    return _dot
#生成运输路线
def gen_line(i):
    for j in range(n):
        if (result[i][j] > 0):
            x_line = np.linspace(pi[i, 0], qj[j, 0], 100)
            y_line = np.linspace(pi[i, 1], qj[j, 1], 100)
            ax.plot(x_line,y_line,color=color[i],linestyle='-.')
#生成工厂、仓库位置和产量储量
def gen_loc():
    plt.scatter(pi[:, 0], pi[:, 1], s=40 ,marker='x',label='factory')
    plt.scatter(qj[:, 0], qj[:, 1], s=40 ,marker='^',label='warehouse')
    for j in range(n):
        plt.annotate("%s" % yj[j],xy=((qj[j,0]+4), (qj[j,1]+4)),xytext=((qj[j,0]), (qj[j,1])), color="b",)
    for j in range(m):
        plt.annotate("%s" % xi[j],xy=((pi[j,0]+4), (pi[j,1]+4)),xytext=((pi[j,0]), (pi[j,1])),color="r",)
#标记该条路线的运输量
def gen_text(i):
    for j in range(n):
        if (result[i][j] > 0):
            plt.annotate("%s" % result[i][j],xy=(((pi[i, 0]+qj[j,0])/2), ((pi[i, 1]+qj[j,1])/2)),
                         xytext=(((pi[i, 0]+qj[j,0])/2), ((pi[i, 1]+qj[j,1])/2)),color="k", )
gen_loc()
for i in range(m):
    gen_line(i)
    gen_text(i)
plt.legend()
#生成动画
ani = FuncAnimation(fig, update_dot, frames=gen_dot, init_func=init, interval=5)
ani.save(filename="result.gif")
plt.show()



