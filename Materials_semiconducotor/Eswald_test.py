import logging
import os
import sys
sys.path.append('/home/leexa/anaconda3/lib/python3.6/site-packages')
import numpy as np
import matplotlib.pyplot as plt

import pymatgen
from pymatgen.analysis import ewald


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Atom:

    # Equality epsilon
    ee = 1e-6

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 t="",
                 c=0):

        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.c = c


def split_data_into_id_x_y(data, data_type="train"):

    if data_type == "train":
        n, m = data.shape
        ids = data[:, 0].reshape(-1, 1)
        x = data[:, 1:(m-2)]
        y_fe = data[:, m-2].reshape(-1, 1)
        y_bg = data[:, m-1].reshape(-1, 1)
    else:
        ids = data[:, 0].reshape(-1, 1)
        x = data[:, 1:]
        y_fe = np.array([])
        y_bg = np.array([])

    return ids, x, y_fe, y_bg


def convert_uc_atoms_to_input_for_pymatgen(uc_atoms):

    n = len(uc_atoms)
    atom_coords = []
    atom_labels = []
    charge_list = []
    for i in range(n):
        x = uc_atoms[i].x
        y = uc_atoms[i].y
        z = uc_atoms[i].z
        t = uc_atoms[i].t
        c = uc_atoms[i].c

        vec = [x, y, z]

        atom_coords.append(vec)
        atom_labels.append(t)
        charge_list.append(c)
    site_properties = {"charge": charge_list}

    return atom_coords, atom_labels, site_properties


def vector_length(vec):
    return np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def read_geometry_file(path_to_file):
    """
    Read geometry file and save the data into
    a list of vectors and a list of Atoms.
    :param path_to_file:
    :return:
    """
    logger.info("Reading geometry file.")
    with open(path_to_file) as f:
        lines = f.readlines()

    vec_x = lines[3].split()
    vec_y = lines[4].split()
    vec_z = lines[5].split()

    vec_x = [float(vec_x[i]) for i in range(1, len(vec_x))]
    vec_y = [float(vec_y[i]) for i in range(1, len(vec_y))]
    vec_z = [float(vec_z[i]) for i in range(1, len(vec_z))]

    ga_mass = 31.0
    al_mass = 13.0
    in_mass = 49.0
    o_mass = 8.0

    vectors = [vec_x, vec_y, vec_z]
    uc_atoms = []
    for i in range(6, len(lines)):
        sl = lines[i].split()
        x = float(sl[1])
        y = float(sl[2])
        z = float(sl[3])
        t = sl[4]

        if sl[4] == "Ga":
            c = ga_mass
        elif sl[4] == "Al":
            c = al_mass
        elif sl[4] == "In":
            c = in_mass
        elif sl[4] == "O":
            c = o_mass

        a = Atom(x, y, z, t, c)
        uc_atoms.append(a)
    logger.info("Geomtery file read.")

    return vectors, uc_atoms


def ewald_matrix_features(data,
                          data_type="train",
                          file_name=""):

    # noa - number of atoms in unit cell
    # ids - ids of each point
    # x - the provides features in *.csv
    # y_fe - formation energy (not used here)
    # y_bg - band gap
    ids, x, y_fe, y_bg = split_data_into_id_x_y(data,data_type=data_type)

    n, m = ids.shape
    ewald_sum_data = {}
    for i in range(n):
        ewald_sum_data[i] = [[],[],[],[],[]]
        c_id = int(ids[i, 0])
        logger.info("c_id: {0}".format(c_id))

        vectors, uc_atoms = read_geometry_file(data_type + "/" + str(c_id) + "/geometry.xyz")
        atom_coords, atom_labels, site_properties = convert_uc_atoms_to_input_for_pymatgen(uc_atoms)

        # Check the vectors from *.csv with the ones
        # from geometry.xyz.
        lv1 = x[c_id - 1, 5]
        lv2 = x[c_id - 1, 6]
        lv3 = x[c_id - 1, 7]

        lv1_c = vector_length(vectors[0])
        lv2_c = vector_length(vectors[1])
        lv3_c = vector_length(vectors[2])
        print x.shape
        alpha = x[c_id - 1, 8]
        beta = x[c_id - 1, 9]
        gamma = x[c_id - 1, 10]

        logger.info("lv1: {0}, lv2: {1}, lv3: {2}".format(lv1, lv2, lv3))
        logger.info("lv1: {0}, lv2: {1}, lv3: {2}".format(lv1_c, lv2_c, lv3_c))
        logger.info("alpha: {0}, beta: {1}, gamma: {2}".format(alpha, beta, gamma))

        # Create a lattice
        lattice = pymatgen.Lattice.from_parameters(a=lv1,
                                                   b=lv2,
                                                   c=lv3,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   gamma=gamma)

        # Create a structure representation in pymatgen
        structure = pymatgen.Structure(lattice, atom_labels, atom_coords, site_properties=site_properties)

        # Get the Ewald sum
        ewald_sum = ewald.EwaldSummation(structure,compute_forces=True)
        
        logger.info("ewald_sum: \n{0}".format(ewald_sum))

        logger.info("Real space energy: {0}".format(ewald_sum.real_space_energy))
        logger.info("Reciprocal energy: {0}".format(ewald_sum.reciprocal_space_energy))
        logger.info("Point energy: {0}".format(ewald_sum.point_energy))
        logger.info("Total energy: {0}".format(ewald_sum.total_energy) )
        

        # Calcualte the traces.
        # Note: point_energy_matrix is an array. We convert it
        # into a diagonal matrix and then compute the trace.
        ewald_sum_data[i][0] = ewald_sum.real_space_energy_matrix
        ewald_sum_data[i][1] = ewald_sum.reciprocal_space_energy_matrix
        ewald_sum_data[i][2] = ewald_sum.total_energy_matrix
        ewald_sum_data[i][3] = ewald_sum.point_energy_matrix
        ewald_sum_data[i][4] = ewald_sum.forces

    # Take only space group and number of total atoms from x.
    features = ewald_sum_data
    np.save(file_name, features)

    return features

def extract_data_by_index_and_value(features, index, value):

    condition = features[:, index] == value
    d = features[condition]

    return d


if __name__ == "__main__":

    data = np.loadtxt("test.csv", delimiter=",", skiprows=1)

    data_type="test/test"
    file_name = "test_ewald_sum_data.npy"

    features = None
    if os.path.isfile(file_name) == False:
        features = ewald_matrix_features(data,
                                         data_type=data_type,
                                         file_name=file_name)
    else:
        features = np.load(file_name)
        logger.info("features.shape: {0}".format(features.shape))

    # nota - number of total atoms
    nota = [10, 20, 30, 40, 60, 80]

    feature_index = {"real_space_energy_matrix": 0,
                     "reciprocal_space_energy_matrix": 1,
                     "total_energy_matrix": 2,
                     "point_energy_matrix": 3,
                     "spacegroup": 4,
                     "number_of_total_atoms": 5,
                     "band_gap": 6}

    to_plot = ["real_space_energy_matrix",
               "reciprocal_space_energy_matrix",
               "total_energy_matrix",
               "point_energy_matrix"]

##    for mat in to_plot:
##        logger.info("feature_index: {0}".format(feature_index[mat]))
##        for i in range(len(nota)):
##
##            d_nota = extract_data_by_index_and_value(features, feature_index["number_of_total_atoms"], nota[i])
##            logger.info("d_nota.shape: {0}".format(d_nota.shape))
##
##            sg = np.unique(d_nota[:, feature_index["spacegroup"]])
##            logger.info("sg: {0}".format(sg))
##
##            plt.figure()
##            for j in range(len(sg)):
##
##                d_sg = extract_data_by_index_and_value(d_nota, feature_index["spacegroup"], int(sg[j]))
##                logger.info("d_sg.shape: {0}".format(d_sg.shape))
##
##                x_ew = d_sg[:, feature_index[mat]]
##                y_bg = d_sg[:, feature_index["band_gap"]]
##
##                p = np.polyfit(x_ew.ravel(), y_bg.ravel(), 2)
##                poly_model = np.poly1d(p)
##                xp = np.linspace(np.min(x_ew), np.max(x_ew), 1000)


