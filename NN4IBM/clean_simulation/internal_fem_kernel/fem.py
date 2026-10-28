from classification import Classification
from shape import Shape
from kinematics import Kinematics
from integration.gauss_integration import Gauss
from integration.octreesampler import OctreeSampler
from integration.element_stiffness import Stiffness
from integration.nitsche_bc import NitscheBC
from assembly.global_system import GlobalAssembly
from assembly.nitsche import NitscheAssembly
from assembly.stiffness import StiffnessAssembly
from assembly.force import ForceAssembly


class FEM:
    Classification = Classification
    Shape = Shape
    Kinematics = Kinematics
    Gauss = Gauss
    OctreeSampler = OctreeSampler
    Stiffness = Stiffness
    NitscheBC = NitscheBC
    GlobalAssembly = GlobalAssembly
    NitscheAssembly = NitscheAssembly
    StiffnessAssembly = StiffnessAssembly
    ForceAssembly = ForceAssembly

