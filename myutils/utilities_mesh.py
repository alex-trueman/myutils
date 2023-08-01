"""Various mesh-related utility functions."""

__author__ = "Alex Trueman"

import trimesh

def meshes_extents(meshes, buffer=None):
    """Find the spatial extents of a list of meshes.

    Parameters
    ----------
    meshes : Sequence of `rmsp.Mesh` or `rmsp.Solid` objects or `rmsp.MeshData`.
    buffer : Increase the extents by a factor (e.g., `0.05` increases the
        extents by 5% on each side of the axes for a total of 10% extra
        for each axis. The buffer is applied to each axis.

    Return
    ------
    Dictionary of two-tuples, where keys are 'x', 'y', and 'z' and the
    tuples have the minimum and maximum extents, respectively, for each
    of the axes. The returned dictionary is the same structure as that
    returned by `rmsp.Mesh.spatial_extents`.
    """

    # Initialize the extents dictionary.
    extents = {
        "x": [1e21, -1e21],
        "y": [1e21, -1e21],
        "z": [1e21, -1e21],
    }

    # Calculate the extents of the listed meshes.
    for m in meshes:
        se = m.spatial_extents
        extents["x"] = [
            min(extents["x"][0], se["x"][0]),
            max(extents["x"][1], se["x"][1]),
        ]
        extents["y"] = [
            min(extents["y"][0], se["y"][0]),
            max(extents["y"][1], se["y"][1]),
        ]
        extents["z"] = [
            min(extents["z"][0], se["z"][0]),
            max(extents["z"][1], se["z"][1]),
        ]

    # Apply the buffer to the extents.
    if buffer:
        buff = [(s[1] - s[0]) * buffer for s in extents.values()]
        extents = {
            k: (s[0] - b, s[1] + b) for b in buff for k, s in extents.items()
        }

    # Convert the listed extents to tuples.
    return {k: tuple(v) for k, v in extents.items()}


def from_obj(file, solid=True):
    """Import OBJ file to rmsp Mesh/Solid."""
    t_mesh: trimesh.Trimesh = trimesh.load(file)
    return rmsp.Solid(trimesh=t_mesh) if solid else rmsp.Mesh(trimesh=t_mesh)
