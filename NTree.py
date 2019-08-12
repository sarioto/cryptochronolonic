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

    phi = 1.61805

    def __init__(self, in_coord, width, level):
        self.w = 0.0
        self.coord = in_coord
        self.width = width
        self.lvl = level
        self.num_children = 2**len(self.coord)
        self.cs = []
        self.subbed_dimen_count = 0
        self.bounds = []
        for x in range(len(self.coord)):
            self.bounds.append(width)
    
    def set_signs(self):
        return list(itertools.product[1,-1], repeat=len(self.coord))

    def sub_divide(self):
        child = []
        sign = 1
        for y in range(len(self.coord)):
            golden_rekt = []
            for x in range(len(self.coord)):
                if x != y:
                    golden_rekt.append(self.bounds[x/phi])
                else:
                    golden_rekt.append(self.bounds[x])
            cube_from_rekt = godlen_rekt 