import trimesh

try:
    from .geometry import Volume
except ImportError:
    from geometry import Volume


class VolumeFactory:

    @staticmethod
    def sphere(radius=0.5, center=(0, 0, 0)):
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        mesh.apply_translation(center)
        return Volume(mesh)

    @staticmethod
    def cylinder(radius=0.5, height=1.0, segments=32):
        mesh = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=segments
        )
        return Volume(mesh)

    @staticmethod
    def box(extents=(1, 1, 1), center=(0, 0, 0)):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        return Volume(mesh)

    @staticmethod
    def from_file(path):
        mesh = trimesh.load(path, force='mesh')
        return Volume(mesh)