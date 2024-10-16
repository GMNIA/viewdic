import numpy as np
import os
import csv
import statistics
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets as pqw


def ReadData_csv(filename, data_length=0):
    """Read a CSV file containing 3D coordinates and return a list of dictionaries with the data.

    Keyword arguments:
    filename -- the name of the CSV file
    data_length -- the expected number of data rows (default 0, calculated from the file if not provided)
    """

    # Open the csv file
    path = os.path.join(os.getcwd(), filename)
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for i in range(3):  # Skip first three lines
            csvfile.readline()

        # Convert reader to list and create an empty list with appropriate length
        string_list = list(reader)
        if data_length == 0:
            data = [None] * (int(string_list[-1]['id']) + 1)
        else:
            data = [None] * data_length
        
        for row in string_list:
            data_row = dict([('id', int(row['id'])), ('x', float(row['x'])), ('y', float(row['y'])), ('z', float(row['z']))])
            data[data_row['id']] = data_row
    return data


def Axes(data_dict):
    """Calculate the principal axes using eigenvalue decomposition.

    Keyword arguments:
    data_dict -- the input data as a list of dictionaries with 'x', 'y', and 'z' coordinates
    """

    # Initialise coordinates
    xyz = []
    for line in data_dict:
        if line is not None:
            xyz.append([line['x'], line['y'], line['z']])
    coord = np.array(xyz, float)

    # Compute geometric center and center data by subtracting geometric center
    center = np.mean(coord, 0)
    coord = coord - center

    # Compute principal axis matrix and eigenvalues/eigenvectors
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)

    # Sort eigenvalues and corresponding eigenvectors
    order = np.argsort(e_values)
    axis3, axis2, axis1 = e_vectors[:, order].transpose()

    MRot = np.matrix([axis1, axis2, axis3])

    data = []
    for v in data_dict:
        if v is not None:
            v_i = dict([('id', 1), ('x', 0), ('y', 0), ('z', 0)])
            vect = DictVect(v)
            v_i['id'] = v['id']
            v_i['x'] = np.dot(MRot[0:], vect)[0, 0]
            v_i['y'] = np.dot(MRot[1:], vect)[0, 0]
            v_i['z'] = np.dot(MRot[2:], vect)[0, 0]
        else:
            v_i = v

        data.append(v_i)

    return data


def DictVect(xyz_dict):
    """Convert a dictionary with 'x', 'y', 'z' keys into a numpy array.

    Keyword arguments:
    xyz_dict -- the dictionary with keys 'x', 'y', and 'z'
    """

    # Convert dictionary to array
    xyz = np.zeros(3)
    xyz[0] = xyz_dict['x']
    xyz[1] = xyz_dict['y']
    xyz[2] = xyz_dict['z']
    return xyz


class init_step:
    """Initialize the first step with 3D data from the CSV file.

    Keyword arguments:
    filename -- the name of the CSV file
    el_size -- the element size as a proportion of the total scanned surface
    """
    def __init__(self, filename, el_size):
        self.data_0axes = ReadData_csv(filename)
        self.data = Axes(self.data_0axes)
        self.el_size = el_size


    def npData(self):
        """Return the data as a numpy array."""
        xyz = np.zeros((len(self.data), 3))
        for i, v in enumerate(self.data):
            if v is not None:
                xyz[i, :] = DictVect(v)
            else:
                xyz[i, :] = None
        return xyz


    def el_top(self):
        """Return the top elements based on the element size ratio."""

        # Initialise list describing top part of the surface
        el_top = []
        id_tot = len(self.data)
        n_el = int(id_tot * self.el_size)

        # Loop through data point and create top part
        for row in self.data:
            if self.data.index(row) < n_el and row is not None:
                el_top.append(row)
            else:
                for el_1 in el_top:
                    if el_1 is not None and row is not None and el_1['x'] > row['x']:
                        el_min = min(el_top, key=lambda x: x['x'])
                        el_top[el_top.index(el_min)] = row
                        break
        return el_top


    def el_bottom(self):
        """Return the bottom elements based on the element size ratio."""

        # Initialise list describing bottom part of the surface
        el_bottom = []
        id_tot = len(self.data)
        n_el = int(id_tot * self.el_size)

        # Loop through data point and create bottom part
        for row in reversed(self.data):
            if self.data.index(row) > (id_tot - n_el) and row is not None:
                el_bottom.append(row)
            else:
                for el_1 in el_bottom:
                    if el_1 is not None and row is not None and el_1['x'] < row['x']:
                        el_max = max(el_bottom, key=lambda x: x['x'])
                        el_bottom[el_bottom.index(el_max)] = row
                        break

        return el_bottom


class step:
    """Process a subsequent step with 3D data and compare it to the initial step.

    Keyword arguments:
    filename -- the name of the CSV file for this step
    filename_step0 -- the name of the initial step file for determining data length
    """
    def __init__(self, filename, filename_step0):
        data_0 = ReadData_csv(filename_step0)
        self.data_0axes = ReadData_csv(filename, len(data_0))
        self.data = Axes(self.data_0axes)


    def npData(self):
        """Return the data as a numpy array."""
        xyz = np.zeros((len(self.data), 3))
        for i, v in enumerate(self.data):
            if v is not None:
                xyz[i, :] = DictVect(v)
            else:
                xyz[i, :] = None
        return xyz


    def el_top(self, el_top_0):
        """Return the top elements for this step based on initial step data.

        Keyword arguments:
        el_top_0 -- the top elements from the initial step
        """
        return [self.data[el_0['id']] for el_0 in el_top_0]


    def list_diff_top(self, el_top_0):
        """Return the list of differences for top elements between this and initial step.

        Keyword arguments:
        el_top_0 -- the top elements from the initial step
        """

        # Initialise difference / distances from points in 3d
        list_diff = []
        diff = {'id': 1, 'dx': 0, 'dy': 0, 'dz': 0}
        self.gap_counter_top = 0
        for el_0 in el_top_0:
            el_i = self.data[el_0['id']]
            if el_i is not None:
                diff['id'] = el_i['id']
                diff['dx'] = el_i['x'] - el_0['x']
                diff['dy'] = el_i['y'] - el_0['y']
                diff['dz'] = el_i['z'] - el_0['z']
                list_diff.append(diff)
            else:
                self.gap_counter_top += 1
        return list_diff


    def vect_diff_top(self, el_top_0):
        """Return the average differences for top elements.

        Keyword arguments:
        el_top_0 -- the top elements from the initial step
        """

        # Calculate difference / distances from points in 3d in the top part of the surface
        diffs = self.list_diff_top(el_top_0)
        diff_x = [d['dx'] for d in diffs]
        diff_y = [d['dy'] for d in diffs]
        diff_z = [d['dz'] for d in diffs]
        return {'x': statistics.mean(diff_x), 'y': statistics.mean(diff_y), 'z': statistics.mean(diff_z)}


    def el_bottom(self, el_bottom_0):
        """Return the bottom elements for this step based on initial step data.

        Keyword arguments:
        el_bottom_0 -- the bottom elements from the initial step
        """
        return [self.data[el_0['id']] for el_0 in el_bottom_0]


    def list_diff_bottom(self, el_bottom_0):
        """Return the list of differences for bottom elements between this and initial step.

        Keyword arguments:
        el_bottom_0 -- the bottom elements from the initial step
        """

        # Initialise difference / distances from points in 3d
        list_diff = []
        diff = {'id': 1, 'dx': 0, 'dy': 0, 'dz': 0}
        self.gap_counter_bottom = 0
        for el_0 in el_bottom_0:
            el_i = self.data[el_0['id']]
            if el_i is not None:
                diff['id'] = el_i['id']
                diff['dx'] = el_i['x'] - el_0['x']
                diff['dy'] = el_i['y'] - el_0['y']
                diff['dz'] = el_i['z'] - el_0['z']
                list_diff.append(diff)
            else:
                self.gap_counter_bottom += 1
        return list_diff


    def vect_diff_bottom(self, el_bottom_0):
        """Return the average differences for bottom elements.

        Keyword arguments:
        el_bottom_0 -- the bottom elements from the initial step
        """

        # Calculate the difference vectors
        diffs = self.list_diff_bottom(el_bottom_0)
        diff_x = [d['dx'] for d in diffs]
        diff_y = [d['dy'] for d in diffs]
        diff_z = [d['dz'] for d in diffs]
        return {'x': statistics.mean(diff_x), 'y': statistics.mean(diff_y), 'z': statistics.mean(diff_z)}


    def gap_counter(self):
        """Return the total number of gaps in both top and bottom elements."""
        return self.gap_counter_top + self.gap_counter_bottom


class MyGLView(gl.GLViewWidget):
    """Create a custom GL view widget for visualizing 3D data.

    Keyword arguments:
    data -- the 3D coordinates to visualize
    withLabels -- whether to display labels for the data points (default False)
    """


    def __init__(self, data, withLabels=False):
        super(MyGLView, self).__init__()
        self.data = data
        self.withLabels = withLabels
        self.texts = []


    def setText(self, text):
        """Set text to display."""
        self.text = text
        self.update()


    def setX(self, X):
        """Set X-axis values."""
        self.X = X
        self.update()


    def setY(self, Y):
        """Set Y-axis values."""
        self.Y = Y
        self.update()


    def setZ(self, Z):
        """Set Z-axis values."""
        self.Z = Z
        self.update()


    def addLabel(self, text, pos):
        """Add a label at a specific 3D position.

        Keyword arguments:
        text -- the label text
        pos -- the position (x, y, z) for the label
        """

        self.texts.append((text, pos))


    def paintGL(self):
        """Render the GL view widget and overlay text labels in the same 3D space as the points."""
        super(MyGLView, self).paintGL()
        if self.withLabels:
            for _, pos in enumerate(self.data):
                # Render label for each point (currently commented out)
                # TODO apply a method as self.renderText(x, y, z, str(i)) for rendering text
                x, y, z = pos



if __name__ == '__main__':
    # Initialise the app
    app = pqw.QApplication([])

    # Folder and file initialization
    folder_list = os.listdir(os.getcwd())
    chosen_folder = 'CSV'
    sub_folder_A = chosen_folder
    file_list = os.listdir(os.path.join(os.getcwd(), sub_folder_A))
    file_list = [file_name for file_name in file_list if file_name.endswith('.csv')]

    # Load initial step
    path_0 = os.path.join(sub_folder_A, file_list[0])
    step_0 = init_step(path_0, 0.10)
    el_top_0 = step_0.el_top()
    el_bottom_0 = step_0.el_bottom()

    # Load a single step for visualization compared to initial one (step_0 VS step_i)
    for step_file in file_list[:1]:
        # Create first step
        step_i = step(os.path.join(sub_folder_A, step_file), path_0)

        # Prepare data for visualization
        test_matrix = step_i.npData()
        test_matrix = test_matrix[~np.isnan(test_matrix).any(axis=1)]

        # Create the GLViewWidget instance with the data
        w = MyGLView(test_matrix, withLabels=True)
        w.setWindowTitle(chosen_folder)
        w.show()

        # Add grid and axes
        g = gl.GLGridItem()
        g.setSize(10, 10)
        a = gl.GLAxisItem()
        a.setSize(20, 20, 20)
        w.addItem(g)
        w.addItem(a)

        # Scatter plot
        sp3 = gl.GLScatterPlotItem(pos=test_matrix, color=(1, 1, 1, 0.3), size=2, pxMode=False)
        w.addItem(sp3)

    # Run the 3d graphics application
    app.exec()
