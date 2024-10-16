import numpy as np
import os
import csv
import statistics
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets as pqw


def read_csv_data(filename, data_length=0):
    """Read a CSV file containing 3D coordinates and return a list of dictionaries with the data.

    Keyword arguments:
    filename -- the name of the CSV file
    data_length -- the expected number of data rows (default 0, calculated from the file if not provided)
    """

    # Build file path and open the CSV file
    path = os.path.join(os.getcwd(), filename)
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for _ in range(3):  # Skip first three lines
            csvfile.readline()

        # Convert reader to list and initialize the data list with the correct length
        string_list = list(reader)
        data = [None] * (int(string_list[-1]['id']) + 1) if data_length == 0 else [None] * data_length

        # Populate the data list with dictionaries containing the coordinates
        for row in string_list:
            data_row = {
                'id': int(row['id']),
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row['z'])
            }
            data[data_row['id']] = data_row
    return data


def compute_principal_axes(data_dict):
    """Calculate the principal axes using eigenvalue decomposition.

    Keyword arguments:
    data_dict -- the input data as a list of dictionaries with 'x', 'y', and 'z' coordinates
    """
    
    # Collect valid coordinates into a numpy array
    coordinates = np.array([[row['x'], row['y'], row['z']] for row in data_dict if row is not None], dtype=float)

    # Compute geometric center and center the coordinates
    center = np.mean(coordinates, axis=0)
    centered_coords = coordinates - center

    # Compute principal axis matrix and eigenvalues/eigenvectors
    inertia_matrix = np.dot(centered_coords.T, centered_coords)
    eigenvalues, eigenvectors = np.linalg.eig(inertia_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, sorted_indices].T  # Transpose to match axes

    # Rotate the data using the principal axes
    rotated_data = []
    for row in data_dict:
        if row is not None:
            vector = dict_to_vector(row)
            rotated_vector = np.dot(principal_axes, vector)
            rotated_data.append({
                'id': row['id'],
                'x': rotated_vector[0],
                'y': rotated_vector[1],
                'z': rotated_vector[2]
            })
        else:
            rotated_data.append(None)

    return rotated_data


def dict_to_vector(coord_dict):
    """Convert a dictionary with 'x', 'y', 'z' keys into a numpy array.

    Keyword arguments:
    coord_dict -- the dictionary with keys 'x', 'y', and 'z'
    """
    return np.array([coord_dict['x'], coord_dict['y'], coord_dict['z']], dtype=float)


class InitialStep:
    """Initialize the first step with 3D data from the CSV file.

    Keyword arguments:
    filename -- the name of the CSV file
    element_size_ratio -- the element size as a proportion of the total scanned surface
    """


    def __init__(self, filename, element_size_ratio):
        self.original_data = read_csv_data(filename)
        self.processed_data = compute_principal_axes(self.original_data)
        self.element_size_ratio = element_size_ratio


    def get_data_as_numpy(self):
        """Return the data as a numpy array."""
        coordinates = np.array(
            [dict_to_vector(row) if row is not None else [None, None, None] for row in self.processed_data]
        )
        return coordinates


    def get_top_elements(self):
        """Return the top elements based on the element size ratio."""
        total_elements = len(self.processed_data)
        num_top_elements = int(total_elements * self.element_size_ratio)
        
        top_elements = []
        for idx, row in enumerate(self.processed_data):
            if idx < num_top_elements and row is not None:
                top_elements.append(row)
            else:
                if row is not None:
                    min_element = min(top_elements, key=lambda x: x['x'])
                    if row['x'] < min_element['x']:
                        top_elements[top_elements.index(min_element)] = row
        return top_elements


    def get_bottom_elements(self):
        """Return the bottom elements based on the element size ratio."""
        total_elements = len(self.processed_data)
        num_bottom_elements = int(total_elements * self.element_size_ratio)
        
        bottom_elements = []
        for idx, row in enumerate(reversed(self.processed_data)):
            if idx < num_bottom_elements and row is not None:
                bottom_elements.append(row)
            else:
                if row is not None:
                    max_element = max(bottom_elements, key=lambda x: x['x'])
                    if row['x'] > max_element['x']:
                        bottom_elements[bottom_elements.index(max_element)] = row
        return bottom_elements


class Step:
    """Process a subsequent step with 3D data and compare it to the initial step.

    Keyword arguments:
    filename -- the name of the CSV file for this step
    initial_filename -- the name of the initial step file for determining data length
    """
    
    def __init__(self, filename, initial_filename):
        initial_data = read_csv_data(initial_filename)
        self.current_data = compute_principal_axes(read_csv_data(filename, len(initial_data)))

    def get_data_as_numpy(self):
        """Return the data as a numpy array."""
        coordinates = np.array(
            [dict_to_vector(row) if row is not None else [None, None, None] for row in self.current_data]
        )
        return coordinates

    def get_elements_by_ids(self, element_list):
        """Return the elements in the current step that correspond to given IDs."""
        return [self.current_data[el['id']] for el in element_list if el is not None]

    def compute_differences(self, initial_elements):
        """Return the list of differences for elements between this and the initial step."""
        differences = []
        gap_counter = 0
        
        for initial_element in initial_elements:
            current_element = self.current_data[initial_element['id']]
            if current_element is not None:
                diff = {
                    'id': current_element['id'],
                    'dx': current_element['x'] - initial_element['x'],
                    'dy': current_element['y'] - initial_element['y'],
                    'dz': current_element['z'] - initial_element['z']
                }
                differences.append(diff)
            else:
                gap_counter += 1

        return differences, gap_counter

    def compute_average_differences(self, initial_elements):
        """Return the average differences for elements between this and the initial step."""
        differences, _ = self.compute_differences(initial_elements)
        
        diff_x = [diff['dx'] for diff in differences]
        diff_y = [diff['dy'] for diff in differences]
        diff_z = [diff['dz'] for diff in differences]

        return {
            'x': statistics.mean(diff_x),
            'y': statistics.mean(diff_y),
            'z': statistics.mean(diff_z)
        }


class MyGLView(gl.GLViewWidget):
    """Create a custom GL view widget for visualizing 3D data.

    Keyword arguments:
    data -- the 3D coordinates to visualize
    with_labels -- whether to display labels for the data points (default False)
    """

    def __init__(self, data, with_labels=False):
        super(MyGLView, self).__init__()
        self.data = data
        self.with_labels = with_labels
        self.texts = []

    def add_label(self, text, position):
        """Add a label at a specific 3D position.

        Keyword arguments:
        text -- the label text
        position -- the position (x, y, z) for the label
        """
        self.texts.append((text, position))

    def paintGL(self):
        """Render the GL view widget and overlay text labels in the same 3D space as the points."""
        super(MyGLView, self).paintGL()
        if self.with_labels:
            for _, pos in enumerate(self.data):
                # Render label for each point (currently commented out)
                # TODO apply a method as self.renderText(x, y, z, str(i)) for rendering text
                x, y, z = pos


if __name__ == '__main__':
    # Initialize the app
    app = pqw.QApplication([])

    # Folder and file initialization
    chosen_folder = 'CSV'
    file_list = [file_name for file_name in os.listdir(os.path.join(os.getcwd(), chosen_folder)) if file_name.endswith('.csv')]

    # Load initial step
    path_initial = os.path.join(chosen_folder, file_list[0])
    initial_step = InitialStep(path_initial, 0.10)
    top_elements_initial = initial_step.get_top_elements()
    bottom_elements_initial = initial_step.get_bottom_elements()

    # Load a single step for visualization compared to the initial one
    for step_file in file_list[:1]:
        # Create che current step
        current_step = Step(os.path.join(chosen_folder, step_file), path_initial)

        # Prepare data for visualization
        visualized_data = current_step.get_data_as_numpy()
        visualized_data = np.array([row for row in visualized_data if row is not None])

        # Create the GLViewWidget instance with the data
        view_widget = MyGLView(visualized_data, with_labels=True)
        view_widget.setWindowTitle(chosen_folder)
        view_widget.show()

        # Add grid and axes
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        axes = gl.GLAxisItem()
        axes.setSize(20, 20, 20)
        view_widget.addItem(grid)
        view_widget.addItem(axes)

        # Scatter plot
        scatter_plot = gl.GLScatterPlotItem(pos=visualized_data, color=(1, 1, 1, 0.3), size=2, pxMode=False)
        view_widget.addItem(scatter_plot)

    # Run the 3D graphics application
    app.exec()
