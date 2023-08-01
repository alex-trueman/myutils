from typing import Dict, List, Union
import pathlib
import rmsp
from .MeshInstance import MeshInstance

mesh_model_type = Dict[str, MeshInstance]
mesh_type = Union[rmsp.Mesh, rmsp.Solid]
path_type = Union[str, pathlib.Path]


def import_meshes(mesh_dict: mesh_model_type) -> mesh_model_type:
    """Import external mesh files using a dictionary of mesh parameters.

    Arguments
    ---------
    mesh_dict : dictionary containing key for each mesh to import. Each
        key contains a MeshInstance, which will store the imported Mesh
        object. It has certain metadata that is used for the import:
        'file_type' (one of 'dm', 'dxf', or 'stl') and 'file' (string or
        path for file to import). Optionally the key 'translate' (tuple
        of x, y, and z translations) is used to translate the coordinates
        of meshes. 'dm' meshes have two files these must each be passed
        as a tuple of file names in 'file' with the point file first.

    Return
    ------
    Returns the input dictionary with added key 'mesh' for each MeshInstance
    containing the imported Meshes or Solids.
    """

    mesh: mesh_type

    # Loop through and update dictionary of MeshInstances.
    for m in mesh_dict.values():

        # Import mesh according to the external file type.
        if m.file_type == "dm":
            mesh = (
                rmsp.Solid.from_dm(*m.file)
                if m.is_solid
                else rmsp.Mesh.from_dm(*m.file)
            )
        elif m.file_type == "dxf":
            mesh = (
                rmsp.Solid.from_dxf(m.file)
                if m.is_solid
                else rmsp.Mesh.from_dxf(m.file)
            )
        elif m.file_type == "stl":
            mesh = (
                rmsp.Solid.from_stl(m.file)
                if m.is_solid
                else rmsp.Mesh.from_stl(m.file)
            )
        else:
            mesh = None

        # Translate coordinates if defined otherwise add the mesh to the
        # dictionary.
        if m.translation:
            m.mesh = mesh.translate(*m.translation)
        else:
            m.mesh = mesh

    return mesh_dict