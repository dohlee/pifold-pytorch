import argparse
import torch
import numpy as np
import biotite.structure as struc

from scipy.spatial.distance import cdist
from biotite.structure.io.pdb import PDBFile

N_IDX, CA_IDX, C_IDX, O_IDX = 0, 1, 2, 3
amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
               'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL',
               'TRP', 'TYR']
aa2idx = {aa:i for i, aa in enumerate(amino_acids)}

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True, help='Input PDB file.')
    parser.add_argument('-o', '--output', required=True, help='Output pt file.')
    parser.add_argument('-k', '--n-neighbors', default=30, type=int, help='Number of k-nearest-neighbor to consider.')

    return parser.parse_args()


def knn_edge_index(structure, k=30):
    ca_coords = np.array([a.coord for a in structure if a.atom_name == 'CA'])
    pdist = cdist(ca_coords, ca_coords, metric='euclidean')

    topk_indices = pdist.argsort(axis=1)[:, 1:k+1]
    edge_idx = np.array([[u, v] for u, neighbors in enumerate(topk_indices) for v in neighbors]).T

    return edge_idx

def to_four_atom_coordinates(atoms):
    coords = np.array([a.coord for a in atoms])
    return coords.reshape(-1, 4, 3)

def rotmat_to_quat(r):
    """Returns quarternion (x, y, z, w) converted from rotation matrix R

    R : (n, 3, 3) tensor
    returns : (n, 4) tensor
    """
    
    xx, yy, zz = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]

    q = 0.5 * np.sqrt(np.abs(1 + np.vstack([
        + xx - yy - zz,
        - xx + yy - zz,
        - xx - yy + zz,
        + xx + yy + zz,
    ])))

    sign = np.sign([
        r[:, 2, 1] - r[:, 1, 2],
        r[:, 0, 2] - r[:, 2, 0],
        r[:, 1, 0] - r[:, 0, 1],
        np.ones(len(r)),
    ])

    q = (sign * q).T
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    return q

def main():
    args = parse_arguments()

    retdict = {}

    # Read PDB file and extract structure
    pdb_file = PDBFile.read(args.input)
    structure = pdb_file.get_structure(model=1)

    # Generate k-NN graph
    edge_idx = knn_edge_index(structure, k=args.n_neighbors)
    retdict['edge_idx'] = torch.from_numpy(edge_idx).long()

    four_atoms = [a for a in structure if a.atom_name in ['N', 'CA', 'C', 'O']]
    four_atom_coords = to_four_atom_coordinates(four_atoms) # (#res, 4, 3)

    u = four_atom_coords[:, CA_IDX] - four_atom_coords[:, N_IDX]
    v = four_atom_coords[:, C_IDX] - four_atom_coords[:, CA_IDX]

    b = (u - v) / np.linalg.norm((u - v), axis=-1, keepdims=True)

    n = np.cross(u, v)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)

    q = np.concatenate([
        b[:, :, None],
        n[:, :, None],
        np.cross(b, n)[:, :, None],
    ], axis=-1)
    retdict['q'] = torch.from_numpy(q).float()

    # NOTE: distance features will be computed on-the-fly in the model.
    #
    # Node angle features
    #
    phi, psi, omega = np.nan_to_num( struc.dihedral_backbone(structure), 0.0)

    # angles
    backbone = structure[struc.filter_backbone(structure)]
    n = len(backbone)

    triplet_indices = np.array([
        np.arange(n-2),
        np.arange(1, n-1),
        np.arange(2, n)
    ]).T

    theta1 = struc.index_angle(backbone, triplet_indices[range(0, n-2, 3)])
    theta2 = struc.index_angle(backbone, triplet_indices[range(1, n-2, 3)])
    theta3 = struc.index_angle(backbone, triplet_indices[range(2, n-2, 3)])

    node_angle_feat = np.array([
        phi,
        psi,
        omega,
        theta1,
        np.hstack([theta2, 0.0]), # theta2 is not defined for the last residue
        np.hstack([theta3, 0.0]), # theta3 is not defined for the last residue
    ]).T

    node_angle_feat = np.concatenate([ np.cos(node_angle_feat), np.sin(node_angle_feat) ], axis=1)
    retdict['node_angle_feat'] = torch.from_numpy(node_angle_feat).float()

    #
    # Node direction features
    #
    node_direction_feat = []

    for atom_idx in [N_IDX, C_IDX, O_IDX]:
        vec_to_ca = four_atom_coords[:, atom_idx] - four_atom_coords[:, CA_IDX]
        vec_to_ca = vec_to_ca / np.linalg.norm(vec_to_ca, axis=1, keepdims=True) # Normalize
        
        direction_feature = (q.transpose(0, 2, 1) @ vec_to_ca[:, :, None]).squeeze(-1)
        node_direction_feat.append( direction_feature )
        
    node_direction_feat = np.concatenate(node_direction_feat, axis=-1)
    retdict['node_dir_feat'] = torch.from_numpy(node_direction_feat).float()

    #
    # Edge angle features
    #
    qi, qj = q[edge_idx[0]], q[edge_idx[1]]
    edge_angle_feat = rotmat_to_quat(qi.transpose(0, 2, 1) @ qj) # x, y, z, w

    retdict['edge_angle_feat'] = torch.from_numpy(edge_angle_feat).float()

    #
    # Edge direction features
    #
    src_idx, dst_idx = edge_idx[0], edge_idx[1]

    qi = q[src_idx]
    four_atom_coords_i, four_atom_coords_j = four_atom_coords[src_idx], four_atom_coords[dst_idx]

    edge_direction_feat = []

    for atom_idx in [N_IDX, C_IDX, O_IDX]:
        vec_to_ca = four_atom_coords_j[:, atom_idx] - four_atom_coords_i[:, CA_IDX]
        vec_to_ca = vec_to_ca / np.linalg.norm(vec_to_ca, axis=1, keepdims=True) # Normalize
        
        direction_feature = (qi.transpose(0, 2, 1) @ vec_to_ca[:, :, None]).squeeze(-1)
        edge_direction_feat.append( direction_feature )
        
    edge_direction_feat = np.concatenate(edge_direction_feat, axis=-1)

    retdict['edge_dir_feat'] = torch.from_numpy(edge_direction_feat).float()

    #
    # Amino acid labels
    #
    _, aa_seq = struc.get_residues(structure)
    aa_idx = [aa2idx[aa] for aa in aa_seq]

    retdict['aa_idx'] = torch.from_numpy(np.array(aa_idx)).long()

    # Finally, save the dictionary to output .pt file.
    torch.save(retdict, args.output)

if __name__ == '__main__':
    main()