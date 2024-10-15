import numpy as np
import os
import csv
import statistics
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore as pqc
from pyqtgraph.Qt import QtWidgets as pqw  # Correct import for QApplication

# 
#open file and gives the data in a list and of dictionaries format
def ReadData_csv(filename, data_length = 0):

    path = os.getcwd()+filename
    
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for i in range(3):
            csvfile.readline()
        #transform reader into a list of numbers
        string_list = list(reader)
        #data length is necessary when the length is not the same as the first step
        if data_length == 0:
            data = [None]*(int(string_list[-1]['id'])+1)   #create empy list
        else:
            data = [None]*(data_length)   #create empy list
        
        for row in string_list:
            data_row = dict([('id', int(row['id'])), ('x', float(row['x'])), ('y', float(row['y'])), ('z', float(row['z']))])
            data[data_row['id']] = data_row
            
    return data

def Axes(data_dict):
    
    xyz = []
    for line in data_dict:
        ###############
        #check if it is appending data in the correct order compared to the dictionary and arrays
        #if ((line['x'] != None) and (line['y'] != None) and (line['z'] != None)) and (line != None):
        if line != None:
            x = line['x']
            y = line['y']
            z = line['z']

            xyz.append([x,y,z])

    coord = np.array(xyz, float)
    # compute geometric center
    center = np.mean(coord, 0)

    # center with geometric center
    coord = coord - center


    # compute principal axis matrix
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    # warning eigen values are not necessary ordered!
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html

    # axis1 is the principal axis with the biggest eigen value (eval1)
    # axis2 is the principal axis with the second biggest eigen value (eval2)
    # axis3 is the principal axis with the smallest eigen value (eval3)
    #--------------------------------------------------------------------------
    order = np.argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()
    #print("Inertia axis are now ordered !")
    #print(eval1, eval2, eval3)
    #print([axis1, axis2, axis3])
    MRot = np.matrix([axis1, axis2, axis3])
    #theoretically it could be solved with a matrix, but it is not considering id-number
    #Mdata = np.matmul(coord, MRot)

    data = []
    for v in data_dict:
        #print(v)
        if v != None:
            v_i = dict([ ('id', 1), ('x', 0), ('y', 0.), ('z', 0.) ])
            vect = DictVect(v)
            v_i['id'] = v['id']
            v_i['x'] = np.dot(MRot[0:], vect)[0,0]
            v_i['y'] = np.dot(MRot[1:], vect)[0,0]
            v_i['z'] = np.dot(MRot[2:], vect)[0,0]

        else:
            v_i = v

        data.append(v_i)

    return(data)

def DictVect(xyz_dict):
    
    xyz = np.zeros(3)
    xyz[0] = xyz_dict['x']
    xyz[1] = xyz_dict['y']
    xyz[2] = xyz_dict['z']
    
    return xyz

class init_step:
    
    #the element size is given as rate of the total scanned surface
    def __init__(self, filename, el_size):
        self.data_0axes = ReadData_csv(filename)
        #data with axes in the new coordinate system
        self.data = Axes(self.data_0axes)
        self.el_size = el_size
    
    def npData(self):
        #gives an ordered version of the coordinates in a numpy format
        xyz = np.zeros((len(self.data),3))
        i = 0
        for v in self.data:
            if v != None:
                xyz[i,:] = DictVect(v)  
            else:
                xyz[i,:] = None
            i +=1
        return xyz

    def el_top(self):
        
        el_top = []
        #finds the total number of elements
        id_tot = len(self.data)
        #print('Total number of elements', id_tot)
        n_el = int(id_tot*self.el_size)
        #chooses the first id_tot/100 elements on the top of the specimen
        for row in self.data:
            #print(self.data.index(row), n_el)
            if self.data.index(row) < n_el and row != None:
                el_top.append(row)
        
            else:
                for el_1 in el_top:
                    if el_1 != None and row != None:
                        if el_1['x'] > row['x']:
                            el_min=el_top[0]
                            for el_2 in el_top:
                                if el_2['x'] > el_min['x']:
                                    el_min = el_2
                            el_top[el_top.index(el_min)] = row
                            break
        return el_top
    
    def el_bottom(self):
        
        el_bottom = []
        #finds the total number of elements and a 1/100 of them
        id_tot = len(self.data)
        n_el = int(id_tot*self.el_size)

        #chooses the last id_tot/100 elements on the bottom of the specimen
        for row in reversed(self.data):
            if self.data.index(row) > (id_tot-n_el) and row != None:
                el_bottom.append(row)

            else:
                for el_1 in el_bottom:
                    if el_1 != None and row != None:
                        if el_1['x'] < row['x']:
                            el_max = el_bottom[0]
                            for el_2 in el_bottom:
                                if el_2['x'] < el_max['x']:
                                    el_max = el_2
                            el_bottom[el_bottom.index(el_max)] = row
                            break
        
        return el_bottom

class step:

    def __init__(self, filename, filename_step0):
        #filename_step0 is the name of the first file to determine the necessary length
        data_0 = ReadData_csv(filename_step0)
        self.data_0axes = ReadData_csv(filename, len(data_0))
        #data with axes in the new coordinate system
        self.data = Axes(self.data_0axes)

    
    def npData(self):
        #gives an ordered version of the coordinates in a numpy format
        xyz = np.zeros((len(self.data),3))
        i = 0
        for v in self.data:
            if v != None:
                xyz[i,:] = DictVect(v)  
            else:
                xyz[i,:] = None
            i +=1
        return xyz

    def el_top(self, el_top_0):

        el_top_i = []
        for el_0 in el_top_0:
            el_top_i.append(self.data[el_0['id']])

        return el_top_i

    def list_diff_top(self, el_top_0):

        list_diff = []
        diff = dict([ ('id', 1), ('dx', 0), ('dy', 0.), ('dz', 0.) ])
        self.gap_counter_top = 0

        for el_0 in el_top_0:
            el_i = self.data[el_0['id']]
            if el_i != None:
                diff['id'] = el_i['id']
                diff['dx'] = ( (el_i['x'] - el_0['x']) )
                diff['dy'] = ( (el_i['y'] - el_0['y']) )
                diff['dz'] = ( (el_i['z'] - el_0['z']) )
                list_diff.append(diff)
                #debug
                #print(diff)
            else:
                self.gap_counter_top += 1

        return list_diff

    def vect_diff_top(self, el_top_0):
        
        diff_x = []
        diff_y = []
        diff_z = []

        for diff in self.list_diff_top(el_top_0):
            diff_x.append(diff['dx'])
            diff_y.append(diff['dy'])
            diff_z.append(diff['dz'])

        vect_diff_top = dict([ ('x', statistics.mean(diff_x)), ('y', statistics.mean(diff_y)), ('z', statistics.mean(diff_z)) ])
        
        return vect_diff_top
        
    def el_bottom(self, el_bottom_0):
        
        el_bottom_i = []
        for el_0 in el_bottom_0:
            el_bottom_i.append(self.data[el_0['id']])
        
        return el_bottom_i
    
    def list_diff_bottom(self, el_bottom_0):

        list_diff = []
        diff = dict([ ('id', 1), ('dx', 0), ('dy', 0.), ('dz', 0.) ])
        self.gap_counter_bottom = 0

        for el_0 in el_bottom_0:
            el_i = self.data[el_0['id']]
            if el_i != None:
                diff['id'] = el_i['id']
                diff['dx'] = ( (el_i['x'] - el_0['x']) )
                diff['dy'] = ( (el_i['y'] - el_0['y']) )
                diff['dz'] = ( (el_i['z'] - el_0['z']) )
                list_diff.append(diff)
            else:
                self.gap_counter_bottom += 1

        return list_diff
    
    def vect_diff_bottom(self, el_bottom_0):

        diff_x = []
        diff_y = []
        diff_z = []

        for diff in self.list_diff_bottom(el_bottom_0):
            diff_x.append(diff['dx'])
            diff_y.append(diff['dy'])
            diff_z.append(diff['dz'])

        vect_diff_bottom = dict([ ('x', statistics.mean(diff_x)), ('y', statistics.mean(diff_y)), ('z', statistics.mean(diff_z)) ])
        
        return vect_diff_bottom

    def gap_counter(self):
        gap_counter = self.gap_counter_top + self.gap_counter_bottom
        return gap_counter

# GUI Class Definition
class MyGLView(gl.GLViewWidget):
    def __init__(self, data, withLabels=False):
        super(MyGLView, self).__init__()
        self.data = data
        self.withLabels = withLabels
        self.texts = []

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def addLabel(self, text, pos):
        """
        Add a label at a specific position
        """
        self.texts.append((text, pos))

    def paintGL(self):
        """
        Paint the GL view widget and overlay text labels in the same 3D space as the points.
        """
        # Call base class paintGL to ensure the scene is rendered
        super(MyGLView, self).paintGL()
        
        # Render the labels in 3D space using OpenGL's renderText()
        if self.withLabels:
            for i, pos in enumerate(self.data):
                # Extract the 3D coordinates of the point
                # TODO Render label of each point with self.renderText(x, y, z, str(i))
                x, y, z = pos


# Main Code Adjustments
if __name__ == '__main__':
    app = pqw.QApplication([])

    # Initialization of variables
    folder_list = os.listdir(os.getcwd())
    chosen_folder = 'CSV'
    sub_folder_A = chosen_folder

    file_list = os.listdir(os.path.join(os.getcwd(), sub_folder_A))
    file_list = [file_name for file_name in file_list if file_name.endswith('.csv')]

    path_0 = '\\' + sub_folder_A + '\\' + file_list[0]
    step_0 = init_step(path_0, 0.10)
    el_top_0 = step_0.el_top()
    el_bottom_0 = step_0.el_bottom()

    # For testing, we'll only load one step file
    for step_file in file_list[0]:    
        path = '\\' + sub_folder_A + '\\' + step_file
        step_i = step(path, path_0)

        ### GUI ###
        test_matrix = step_i.npData()
        test_matrix = test_matrix[~np.isnan(test_matrix).any(axis=1)]

        # Create the GLViewWidget instance with the data
        w = MyGLView(test_matrix, withLabels=True)
        w.setWindowTitle(chosen_folder)
        w.show()

        # Add grid and axes
        g = gl.GLGridItem()
        g.setSize(10, 10)  # Fixed size to avoid the issue
        a = gl.GLAxisItem()
        a.setSize(20, 20, 20)
        w.addItem(g)
        w.addItem(a)

        # Scatter plot
        sp3 = gl.GLScatterPlotItem(pos=test_matrix, color=(1, 1, 1, .3), size=2, pxMode=False)
        w.addItem(sp3)

    app.exec()
