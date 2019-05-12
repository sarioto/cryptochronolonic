import itertools

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
'''
tree = nDimensionGoldenTree([0.0, 0.0, 0.0], 1.0, 1)

tree.sub_divide()

for i in tree.cs:
    print(i.coord)
'''
                


