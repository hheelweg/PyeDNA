import os
from pathlib import Path
import numpy as np
from itertools import combinations
from math import comb
import MDAnalysis as mda
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

PAIR_FEATURES = ["dist",
                "n1_dist", "u1_dist", "v1_dist",
                "n2_dist", "u2_dist", "v2_dist",
                "n1_n2", "n1_u2", "n1_v2",
                "u1_n2", "u1_u2", "u1_v2",
                "v1_n2", "v1_u2", "v1_v2"]

WEIGHT_MODES = ["Distance", 
                "Displacement", 
                "Orientation",
                "PrincipalAxes", 
                "DistancePrincipalAxes",
                "DisplacementPrincipalAxes",
                "DistanceNormal"]

class Clusterer:
    """
    Cluster DNA-dye chromophore configurations by dye features.

    TODO: Change major_atoms and minor_atoms to define separate axes for each dye type in dye_names.
          Not strictly necessary, unless the specified atoms are collinear in a dye, or don't exist.
    
    """

    def __init__(self, structure_name, dye_names, structure_dir="structures", 
                    major_atoms=("N1", "N2"),
                    minor_atoms=("N1", "C2"),
                    N_res_rad=0):

        # Locate structure directory
        self.structure_dir = Path(structure_dir)

        # Collect all PDB files 
        pdb_files = list(self.structure_dir.glob(f"{structure_name}*.pdb"))

        # Get labels of each pdb file 
        self.pdb_labels = [Path(p).stem.removeprefix(f"{structure_name}_") for p in pdb_files]

        self.N_res_rad = N_res_rad
        self.major_atoms = major_atoms
        self.minor_atoms = minor_atoms

        # Read pdb files into separate universes 
        self.universes = {
                            f"pdb_{i}": mda.Universe(pdb_file)
                            for i, pdb_file in enumerate(pdb_files, start=1)
                        }
        self.N_pdb = len(pdb_files)

        # Create reference pdb file (all files should have same topology)
        u_ref = self.universes["pdb_1"]

        # Create dict of dye locations
        self.dye_resids = {
            dye: u_ref.select_atoms(f"resname {dye}").residues.resids.tolist() for dye in dye_names
        }

        # Create set of dye pair residue IDs
        self.all_dye_resids = [resid for resids in self.dye_resids.values() for resid in resids]
        self.N_dyes = len(self.all_dye_resids)
        self.dye_pair_resids = [tuple(sorted(pair)) for pair in combinations(self.all_dye_resids, 2)]
        self.N_dye_pairs = len(self.dye_pair_resids)

        # Create set of environment residue IDs around each dye
        self.env_resids = [
                            list(range(dye_resid - N_res_rad, dye_resid + N_res_rad + 1))
                            for dye_resids_list in self.dye_resids.values()
                            for dye_resid in dye_resids_list
                          ]

        # Construct individual pair feature index helpers
        PAIR_FEATURES = [
            "dist",
            "n1_dist", "u1_dist", "v1_dist",
            "n2_dist", "u2_dist", "v2_dist",
            "n1_n2", "n1_u2", "n1_v2",
            "u1_n2", "u1_u2", "u1_v2",
            "v1_n2", "v1_u2", "v1_v2",
        ]

        self.PAIR_FEATURE_IDX = {name: i for i, name in enumerate(PAIR_FEATURES)}

        # Construct group pair feature index helpers
        self.DIST_IDX = []
        self.DISP_IDX = []
        self.ORIENT_IDX = []
        self.PRINC_ORIENT_IDX = []
        self.NORMAL_IDX = []
        
        for pair_idx in range(self.N_dye_pairs):
            offset = 16 * pair_idx
    
            self.DIST_IDX.append(offset)
            self.DISP_IDX.extend(offset + np.arange(1, 7))
            self.ORIENT_IDX.extend(offset + np.arange(7, 16))
            self.PRINC_ORIENT_IDX.extend(offset + np.array([7, 11, 15]))
            self.NORMAL_IDX.append(offset + np.array([7]))
    
        # Construct environment feature index helpers
        self.N_features = 16 * comb(self.N_dyes, 2) + 6 * N_res_rad * self.N_dyes
        self.ENV_IDX = np.arange(16 * self.N_dye_pairs, self.N_features)
        
    def make_weights(self, weight_mode, ws=[0]):

        weights = np.zeros(self.N_features)

        if weight_mode == "Distance":
            weights[self.DIST_IDX] = 1.0

        elif weight_mode == "Displacement":
            weights[self.DISP_IDX] = 1.0
        
        elif weight_mode == "Orientation":
            weights[self.ORIENT_IDX] = 1.0

        elif weight_mode == "PrincipalAxes":
            weights[self.PRINC_ORIENT_IDX] = 1.0

        elif weight_mode == "DistancePrincipalAxes":
            assert len(ws) == 2
            weights[self.DIST_IDX] = ws[0]
            weights[self.PRINC_ORIENT_IDX] = ws[1]

        elif weight_mode == "DisplacementPrincipalAxes":
            assert len(ws) == 2
            weights[self.DISP_IDX] = ws[0]
            weights[self.PRINC_ORIENT_IDX] = ws[1]

        elif weight_mode == "DistanceNormal":
            assert len(ws) == 2
            weights[self.DIST_IDX] = ws[0]
            weights[self.NORMAL_IDX] = ws[1]

        elif weight_mode == "All":
            assert len(ws) == 6
            weights[self.DIST_IDX] = ws[0]
            weights[self.DISP_IDX] = ws[1]
            weights[self.ORIENT_IDX] = ws[2]
            weights[self.PRINC_ORIENT_IDX] = ws[3]      # Will rewrite over some ORIENT_IDX
            weights[self.NORMAL_IDX] = ws[4]            # Will rewrite over some PRINC_ORIENT_IDX
            weights[self.ENV_IDX] = ws[5]  

        return weights 

    def pair_feature_idx(self, resid_i, resid_j, feature_name):

        pair = tuple(sorted((resid_i, resid_j)))

        pair_idx = self.dye_pair_resids.index(pair)
        feat_idx = self.PAIR_FEATURE_IDX[feature_name]

        return 16 * pair_idx + feat_idx
    
    def get_com_envs(self, pdb):

        com_envs = {}
        for env_resid_list in self.env_resids:
            for resid in env_resid_list:
                residue = self.universes[pdb].select_atoms(f"resid {resid}")
                com_envs[resid] = residue.center_of_mass()

        return com_envs

    def get_com_env_disps(self, com_envs):

        com_env_disps = {}
        for env_resid_list in self.env_resids:
            dye_resid = env_resid_list[self.N_res_rad]
            com_dye = com_envs[dye_resid]

            com_env_disps[dye_resid] = {}
            for resid in env_resid_list:
                if resid == dye_resid:
                    continue

                com_env_disps[dye_resid][resid] = com_envs[resid] - com_dye

        return com_env_disps

    def get_oriented_princ_axes(self, pdb, dye_resid):

        # Select dye residue
        res = self.universes[pdb].select_atoms(f"resid {dye_resid}")

        # MDAnalysis principal axes
        axes = res.principal_axes()
        normal, minor, major = axes

        # Dye-fixed reference for major axis
        r1 = res.select_atoms(f"name {self.major_atoms[0]}").positions[0]
        r2 = res.select_atoms(f"name {self.major_atoms[1]}").positions[0]
        ref_major = r2 - r1

        if np.dot(major, ref_major) < 0:
            major *= -1

        # Dye-fixed reference for minor axis
        r1 = res.select_atoms(f"name {self.minor_atoms[0]}").positions[0]
        r2 = res.select_atoms(f"name {self.minor_atoms[1]}").positions[0]
        ref_minor = r2 - r1

        if np.dot(minor, ref_minor) < 0:
            minor *= -1

        # Construct normal so frame is consistently right-handed
        normal = np.cross(minor, major)
        normal /= np.linalg.norm(normal)

        princ_axes = np.column_stack((normal, minor, major))

        return princ_axes

    def get_orient_princ_axes(self, pdb):

        princ_axes = {}
        for dye_resid in self.all_dye_resids:
            princ_axes_resid = self.get_oriented_princ_axes(pdb, dye_resid)
            princ_axes[dye_resid] = princ_axes_resid

        return princ_axes

    def pair_features(self, resid_i, resid_j, com_envs, princ_axes):

        # Displacement vector between dyes
        com_dye_i = com_envs[resid_i]
        com_dye_j = com_envs[resid_j]
        d_ij = com_dye_i - com_dye_j
        dist_ij = np.linalg.norm(d_ij)

        # Get sorted axes orientations
        axes_i = princ_axes[resid_i]
        axes_j = princ_axes[resid_j]

        # Calculate orientation-displacement vectors
        Di_ij = np.transpose(axes_i) @ d_ij
        Dj_ij = np.transpose(axes_j) @ d_ij

        # Calculate relative orientation matrix 
        R_ij = np.transpose(axes_i) @ axes_j 
        R_ij = R_ij.flatten()

        # Collect features 
        feat_ij = np.concatenate(([dist_ij], Di_ij, Dj_ij, R_ij))

        return feat_ij

    def get_pair_features(self, com_envs, princ_axes):
        pair_feats = np.empty((self.N_dye_pairs, 16))
        for idx, (resid_i, resid_j) in enumerate(self.dye_pair_resids):
            pair_feats[idx] = self.pair_features(resid_i, resid_j, com_envs, princ_axes)

        return pair_feats

    def get_env_features(self, com_env_disps, princ_axes):
        env_feats = np.empty((self.N_dyes, 6 * self.N_res_rad))
        for i, resid_i in enumerate(self.all_dye_resids):
            axes_i = princ_axes[resid_i]
            trans_axes_i = np.transpose(axes_i)

            env_feats_i = np.empty((2 * self.N_res_rad, 3))
            for j, com_env_disp_ij in enumerate(com_env_disps[resid_i].values()):
                env_feats_i[j] = trans_axes_i @ com_env_disp_ij

            env_feats[i] = env_feats_i.flatten()

        return env_feats

    def get_features_pdb(self, pdb):

        # Get dye locations (and surrounding residues)
        com_envs = self.get_com_envs(pdb)  

        # Get principal axes 
        princ_axes = self.get_orient_princ_axes(pdb)        

        # Compute displacements within each dye environment
        com_env_disps = self.get_com_env_disps(com_envs)      

        # Compute pairwise dye distance and orientation features
        pair_feats = self.get_pair_features(com_envs, princ_axes)

        # Compute environment displacement features
        env_feats = self.get_env_features(com_env_disps, princ_axes)

        # Concatenate all features
        features = np.concatenate((pair_feats.flatten(), env_feats.flatten()))

        return features

    def get_features(self):

        features = np.empty((self.N_pdb, self.N_features))
        for i in range(self.N_pdb):
            print(f"Extracting features from structure {i+1}...")
            features[i] = self.get_features_pdb(f"pdb_{i+1}")

        return features

    def weight_features(self, features, weights):

        assert len(features[0]) == len(weights), "Number of features and weights do not match"
        
        # Multiply columns of features by their weights
        w_features = features * weights 

        return w_features 

    def cluster(self, features, weights, min_cluster_size, min_samples):

        w_features = self.weight_features(features, weights)

        clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )

        clust_labels = clusterer.fit_predict(w_features)
        N_clust = len(np.unique(clust_labels[clust_labels != -1]))

        return clust_labels, N_clust

    def extract_feature_column(self, features, feature_name, dye1=0, dye2=1):

        resid_1 = self.all_dye_resids[dye1]
        resid_2 = self.all_dye_resids[dye2]
        idx = self.pair_feature_idx(resid_1, resid_2, feature_name)
        feature_col = features[:, idx]

        return feature_col

    def plot_clusters(self, features, clust_labels, 
                      featureA_label, featureA_name, 
                      featureB_label, featureB_name,
                      save_name, save_path,
                      dye1=0, dye2=1,
                      color_map="tab10", num_labels=True):
        
        # Mask noise data
        mask = clust_labels == -1

        # Extract feature columns 
        featureA = self.extract_feature_column(features, featureA_name, dye1, dye2)
        featureB = self.extract_feature_column(features, featureB_name, dye1, dye2)

        # Plot data
        plt.scatter(featureA[~mask], featureB[~mask], c=clust_labels[~mask], cmap=color_map)
        plt.scatter(featureA[mask], featureB[mask], c="black")
        plt.xlabel(featureA_label)
        plt.ylabel(featureB_label)

        if num_labels == True:
            for i, (xi, yi) in enumerate(zip(featureA, featureB)):
                plt.annotate(self.pdb_labels[i], (xi, yi))

        os.makedirs(save_path, exist_ok=True)
        rel_save_file = os.path.join(save_path, save_name + ".png")
        abs_save_file = os.path.abspath(rel_save_file)
        plt.savefig(abs_save_file, dpi=300, bbox_inches="tight")
        print(f"Saved cluster plot to: {abs_save_file}")

        plt.show()