import numpy as np
import os
import vine_skeletor
import pickle
import json
import open3d as o3d
from vine_skeletor.visualization.branch import BranchSkeletonVisualizer

def refactor_branch(b):
    if b.is_bearer:
        class_name = 'Bearer'
    else:
        class_name = 'Cane'

    if b.parent_id == -1:
        parent_id = 'Trunk'
    else:
        parent_id = str(b.parent_id)

    cane = {
        'meshes': None,
        'name': str(b._id),
        'class_name': class_name,
        'spine': {'points': b.xyz.tolist(), 'radius': b.radii.tolist()},
        'children': [],
        'parent_id': parent_id
    }

    n = 0
    for node in b.nodes:
        cane['children'].append({
            'meshes': None,
            'name': str(b._id) + '_node' + str(n),
            'class_name': 'Node',
            'spine': {'points': node.xyz.tolist(), 'radius': node.radius},
            'children': []
        })
        n += 1

    return cane


def refactor_vine(data):
    canes = {str(id): refactor_branch(b) for id, b in data.vine.skeleton.branches.items()}

    bearer_ids = [cane['name'] for cane in canes.values() if cane['class_name'] == 'Bearer']

    for cane in canes.values():
        if cane['parent_id'] not in ['Trunk'] + bearer_ids:
            cane['class_name'] = 'Shoot'
            

    trunk = {'meshes': None, 'name': 'Trunk', 'class_name': 'Trunk', 'spine': None, 'children': []}    
    canes['Trunk'] = trunk

    for b in canes.values():
        if b['name'] == 'Trunk':
            continue
        parent = canes.get(b['parent_id'])
        if parent:
            parent['children'].append(b)
        else:
            print(f"Parent {b['parent_id']} not found for branch {b['name']}")
    
    return trunk

def main():
    """
        Load vine-skeletonisation vine objects from model inference pickle files and transform to use for pruning.
        Generates:
            - x_tree.json: speed tree like heirarchical structure of vine
            - x_tubes.ply: mesh of canes
            - x_trunk.ply: point cloud of trunk
            - vine_enu_positions.json: shifts to bring vine to origin

    """
    directory = "/local/skeletons_output/pkl_outputs2"
    shifts_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            path = os.path.join(directory, filename)
            print(path)
        else:
            continue
        
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # reformat to .tree style json
        trunk = refactor_vine(data)
        heirarchy = {'filename': None, 'parts':trunk, 'classes':None}

        save_path = "/local/new_real_vines/" + path[-9:-4] + "_tree.json"
        with open(save_path, "w") as f:
            json.dump(heirarchy, f)

        # save mesh of canes
        data.vine.skeleton.save("/local/new_real_vines/" + path[-9:-4])


        # save point cloud of trunk
        trunk = data.vine.trunk_cld
        # trunk_voxels = trunk.as_voxel_grid(0.005)
        # voxel_points = np.asarray([voxel.grid_index for voxel in trunk_voxels.get_voxels()])
        trunk_point_cloud = o3d.geometry.PointCloud()
        trunk_point_cloud.points = o3d.utility.Vector3dVector(trunk.xyz)

        o3d.io.write_point_cloud("/local/new_real_vines/" + path[-9:-4] + "_trunk.ply", trunk_point_cloud)

        # get vine average pos
        average_pos = np.mean(trunk_point_cloud.points, axis=0)
        shifts_dict[path[-8:-4]] = list(average_pos)

    with open("/local/new_real_vines/vine_enu_positions.json", "w") as f:
        json.dump(shifts_dict, f)


def individual_meshes():
    # save meshes of canes indivividually for easier pruning visualisation
    directory = "/local/skeletons_output/pkl_outputs2"
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            path = os.path.join(directory, filename)
            print(path)
        else:
            continue
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vine_skeleton = data.vine.skeleton

        save_directory = "/local/new_real_vines/individual_canes/" + path[-9:-4]
        os.makedirs(save_directory, exist_ok=True)

        bearer_tubes = [(b._id,
            BranchSkeletonVisualizer(b)
            .as_o3d_tube()
            .paint_uniform_color((1, 0.0, 0)))  # Orange
            for b in vine_skeleton.branches.values()
            if b.is_bearer
        ]
        non_bearer_tubes = [(b._id,
            BranchSkeletonVisualizer(b).as_o3d_tube())
            for b in vine_skeleton.branches.values()
            if not b.is_bearer
        ]
        all_tubes = bearer_tubes + non_bearer_tubes

        for tube in all_tubes:
            path = save_directory + "/" + str(tube[0]) + ".ply"    
            o3d.io.write_triangle_mesh(path, tube[1])

def wires():
    wires_enu = {}
    directory = "/local/skeletons_output/pkl_outputs2"
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            path = os.path.join(directory, filename)
            print(path)
        else:
            continue

        with open(path, 'rb') as f:
            vinyard = pickle.load(f)
        
        # print(vinyard.wires)
        wire = vinyard.wires[0]

        wires_enu[path[-8:-4]] = {
            'a': wire.a.tolist(),
            'b': wire.b.tolist(),
        }

        print(wires_enu[path[-8:-4]])
        
    with open("/local/new_real_vines/wires_enu.json", "w") as f:
        json.dump(wires_enu, f)
    

    
if __name__ == "__main__":
    # main()
    # individual_meshes()
    wires()
            