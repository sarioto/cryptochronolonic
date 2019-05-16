import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class nDimensionTree:
    
    def __init__(self, in_coord, width, level):
        self.w = 0.0
        self.coord = in_coord
        self.width = width
        self.lvl = level
        self.num_children = 2**len(self.coord)
        self.cs = []
        self.signs = self.set_signs()
        #print(self.signs)
    def set_signs(self):
        return list(itertools.product([1,-1], repeat=len(self.coord)))
    
    def divide_childrens(self):
        for x in range(self.num_children):
            new_coord = []
            for y in range(len(self.coord)):
                new_coord.append(self.coord[y] + (self.width/(2*self.signs[x][y])))
            newby = nDimensionTree(new_coord, self.width/2, self.lvl+1)
            self.cs.append(newby)


class nDimensionGoldenTree:

    def __init__(self, in_coord, width, level):
        self.w = 0.0
        self.coord = in_coord
        self.width = width
        self.full_width = width * 2
        self.sub_width = self.full_width / 1.61805
        self.offset_dist = self.sub_width - self.width
        self.lvl = level
        self.num_children = 2**len(self.coord)
        self.cs = []
        self.subbed_dimen_count = 0
        self.lvl = level
        self.signs = self.set_signs()

    def set_signs(self):
        return list(itertools.product([1,-1], repeat=len(self.coord)))

    def divide_childrens(self):
        sign = 1
        golden_cube = []
        dimen = len(self.coord)
        #got the appropriate dimensions for out golden cube
        #we will now permute its position inside the unit hypercube
        #at the current depth
        for i in range(self.num_children):
            child_root = []
            new_center = 0.0
            for y in range(dimen):
                # shift the root inversely from the direction to the center of the sub cube
                new_center = self.coord[y] + ((self.signs[i][y]*-1)*self.offset_dist)
                # us new root position in this dimension as spot to offset from
                child_root.append(new_center + (self.sub_width/(2*self.signs[i][y])))
            self.cs.append(nDimensionGoldenTree(child_root, self.sub_width/2, self.lvl+1))



tree = nDimensionGoldenTree([0.0, 0.0, 0.0], 2.0, 1)

tree.divide_childrens()
xs = []
ys = []
zs = []


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in tree.cs:
    xs.append(i.coord[0])
    ys.append(i.coord[1])
    zs.append(i.coord[2])
    xc = (tree.coord[0], i.coord[0])
    yc = (tree.coord[1], i.coord[1])
    zc = (tree.coord[2], i.coord[2])
    ax.plot3D(xc, yc, zc, color='g')
for i in tree.cs:
    i.divide_childrens()
    for ix in i.cs:
        xs.append(ix.coord[0])
        ys.append(ix.coord[1])
        zs.append(ix.coord[2])
        xc = (i.coord[0], ix.coord[0])
        yc = (i.coord[1], ix.coord[1])
        zc = (i.coord[2], ix.coord[2])
        ax.plot3D(xc, yc, zc, color='g')
for i in tree.cs:
    i.divide_childrens()
    for x in i.cs:
        x.divide_childrens()
        for ix in x.cs:
            xs.append(ix.coord[0])
            ys.append(ix.coord[1])
            zs.append(ix.coord[2])
            xc = (i.coord[0], ix.coord[0])
            yc = (i.coord[1], ix.coord[1])
            zc = (i.coord[2], ix.coord[2])
            ax.plot3D(xc, yc, zc, color='g')
ax.scatter(xs, ys, zs, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()    


