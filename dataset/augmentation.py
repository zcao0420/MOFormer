from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import ase
from ase.io import read, write
import warnings
import numpy as np
from pymatgen.core.structure import Structure
import glob
from ase.io import  read, write
from ase import Atoms,Atom
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.transformation_abc import AbstractTransformation
import math
from typing import Optional, Union
from pymatgen.core.structure import Molecule, Structure


class AseAtomsAdaptor:
    """
    Adaptor serves as a bridge between ASE Atoms and pymatgen objects.
    """

    @staticmethod
    def get_atoms(structure, **kwargs):
        """
        Returns ASE Atoms object from pymatgen structure or molecule.
        Args:
            structure: pymatgen.core.structure.Structure or pymatgen.core.structure.Molecule
            **kwargs: other keyword args to pass into the ASE Atoms constructor
        Returns:
            ASE Atoms object
        """
        if not structure.is_ordered:
            raise ValueError("ASE Atoms only supports ordered structures")
        if not ase_loaded:
            raise ImportError(
                "AseAtomsAdaptor requires ase package.\n" "Use `pip install ase` or `conda install ase -c conda-forge`"
            )
        symbols = [str(site.specie.symbol) for site in structure]
        positions = [site.coords for site in structure]
        if hasattr(structure, "lattice"):
            cell = structure.lattice.matrix
            pbc = True
        else:
            cell = None
            pbc = None
        return Atoms(symbols=symbols, positions=positions, pbc=pbc, cell=cell, **kwargs)

    @staticmethod
    def get_structure(atoms, cls=None):
        """
        Returns pymatgen structure from ASE Atoms.
        Args:
            atoms: ASE Atoms object
            cls: The Structure class to instantiate (defaults to pymatgen structure)
        Returns:
            Equivalent pymatgen.core.structure.Structure
        """
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        lattice = atoms.get_cell()

        cls = Structure if cls is None else cls
        return cls(lattice, symbols, positions, coords_are_cartesian=True)


class RotationTransformation(AbstractTransformation):
    """
    The RotationTransformation applies a rotation to a structure.
    """

    def __init__(self):
        """
        Args:
            axis (3x1 array): Axis of rotation, e.g., [1, 0, 0]
            angle (float): Angle to rotate
            angle_in_radians (bool): Set to True if angle is supplied in radians.
                Else degrees are assumed.
        """
        self.axis = None
        self.angle = None
        # self.angle_in_radians = angle_in_radians
        # self._symmop = SymmOp.from_axis_angle_and_translation(self.axis, self.angle, self.angle_in_radians)

    def apply_transformation(self, structure, axis, angle, angle_in_radians=False):
        """
        Apply the transformation.
        Args:
            structure (Structure): Input Structure
        Returns:
            Rotated Structure.
        """
        self.axis = axis
        self.angle = angle
        self.angle_in_radians = angle_in_radians
        self._symmop = SymmOp.from_axis_angle_and_translation(self.axis, self.angle, self.angle_in_radians)

        s = structure.copy()
        s.apply_operation(self._symmop, fractional=True)
        # s = self._symmop(s)
        return s

    def __str__(self):
        return "Rotation Transformation about axis " + "{} with angle = {:.4f} {}".format(
            self.axis, self.angle, "radians" if self.angle_in_radians else "degrees"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Returns:
            Inverse Transformation.
        """
        return RotationTransformation(self.axis, -self.angle, self.angle_in_radians)

    @property
    def is_one_to_many(self):
        """Returns: False"""
        return False


class PerturbStructureTransformation(AbstractTransformation):
    """
    This transformation perturbs a structure by a specified distance in random
    directions. Used for breaking symmetries.
    """

    def __init__(
        self,
        distance: float = 0.01,
        min_distance: Optional[Union[int, float]] = None,
    ):
        """
        Args:
            distance: Distance of perturbation in angstroms. All sites
                will be perturbed by exactly that distance in a random
                direction.
            min_distance: if None, all displacements will be equidistant. If int
                or float, perturb each site a distance drawn from the uniform
                distribution between 'min_distance' and 'distance'.
        """
        self.distance = distance
        self.min_distance = min_distance

    def apply_transformation(self, structure: Structure) -> Structure:
        """
        Apply the transformation.
        Args:
            structure: Input Structure
        Returns:
            Structure with sites perturbed.
        """
        s = structure.copy()
        s.perturb(self.distance, min_distance=self.min_distance)
        return s

    def __str__(self):
        return "PerturbStructureTransformation : " + "Min_distance = {}".format(self.min_distance)

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Returns: None
        """
        return None

    @property
    def is_one_to_many(self):
        """
        Returns: False
        """
        return False


class SwapAxesTransformation(object):
    def __init__(self, p=0.5):
        self.p = p

    def apply_transformation(self, crys):
        if random.random() > self.p:
            return AseAtomsAdaptor.get_structure(crys)
        else:
            atoms = crys.copy()
            cell = atoms.cell

            choice = np.random.choice(3, 2, replace=False)
            cell[:,[choice[0], choice[1]]] = cell[:,[choice[1], choice[0]]] ## axes you want to swap
            cell[[choice[0], choice[1]]] = cell[[choice[1], choice[0]]]
            atoms.cell = cell
            angles = (cell.angles())
            angles[[choice[0], choice[1]]] = angles[[choice[1], choice[0]]]
            cell.angles  = angles
            pos = (atoms.positions)
            pos[:,[choice[0], choice[1]]] = pos[:,[choice[1], choice[0]]]
            atoms.arrays["positions"] = pos

            return AseAtomsAdaptor.get_structure(atoms)

    def __str__(self):
        return "SwapAxesTransformation"

    def __repr__(self):
        return self.__str__()

class RemoveSitesTransformation(AbstractTransformation):
    """
    Remove certain sites in a structure.
    """

    def __init__(self):
        """
        Args:
            indices_to_remove: List of indices to remove. E.g., [0, 1, 2]
        """

        self.indices_to_remove = None

    def apply_transformation(self, structure, indices_to_remove):
        """
        Apply the transformation.
        Arg:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.
        Return:
            Returns a copy of structure with sites removed.
        """
        self.indices_to_remove = indices_to_remove
        s = structure.copy()
        s.remove_sites(self.indices_to_remove)
        return s

    def __str__(self):
        return "RemoveSitesTransformation :" + ", ".join(map(str, self.indices_to_remove))

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """Return: None"""
        return None

    @property
    def is_one_to_many(self):
        """Return: False"""
        return False

class TranslateSitesTransformation(AbstractTransformation):
    """
    This class translates a set of sites by a certain vector.
    """

    def __init__(self, indices_to_move, translation_vector, vector_in_frac_coords=True):
        """
        Args:
            indices_to_move: The indices of the sites to move
            translation_vector: Vector to move the sites. If a list of list or numpy
                array of shape, (len(indices_to_move), 3), is provided then each
                translation vector is applied to the corresponding site in the
                indices_to_move.
            vector_in_frac_coords: Set to True if the translation vector is in
                fractional coordinates, and False if it is in cartesian
                coordinations. Defaults to True.
        """
        self.indices_to_move = indices_to_move
        self.translation_vector = np.array(translation_vector)
        self.vector_in_frac_coords = vector_in_frac_coords

    def apply_transformation(self, structure):
        """
        Apply the transformation.
        Arg:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.
        Return:
            Returns a copy of structure with sites translated.
        """
        s = structure.copy()
        if self.translation_vector.shape == (len(self.indices_to_move), 3):
            for i, idx in enumerate(self.indices_to_move):
                s.translate_sites(idx, self.translation_vector[i], self.vector_in_frac_coords)
        else:
            s.translate_sites(
                self.indices_to_move,
                self.translation_vector,
                self.vector_in_frac_coords,
            )
        return s

    def __str__(self):
        return "TranslateSitesTransformation for indices " + "{}, vect {} and vect_in_frac_coords = {}".format(
            self.indices_to_move,
            self.translation_vector,
            self.vector_in_frac_coords,
        )

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Returns:
            TranslateSitesTranformation with the reverse translation.
        """
        return TranslateSitesTransformation(self.indices_to_move, -self.translation_vector, self.vector_in_frac_coords)

    @property
    def is_one_to_many(self):
        """Return: False"""
        return False

    def as_dict(self):
        """
        Json-serializable dict representation.
        """
        d = MSONable.as_dict(self)
        d["translation_vector"] = self.translation_vector.tolist()
        return d
# for test purpose
if __name__ == '__main__':
    cif_file = read("original.cif")  # Path for the cif file

    atoms = cif_file.copy()
    cell = atoms.cell

    choice = np.random.choice(3, 2, replace = False)
    cell[:,[choice[0], choice[1]]] = cell[:,[choice[1], choice[0]]] ## axes you want to swap
    cell[[choice[0], choice[1]]] = cell[[choice[1], choice[0]]]
    atoms.cell = cell
    angles = (cell.angles())
    angles[[choice[0], choice[1]]] = angles[[choice[1], choice[0]]]
    cell.angles  = angles
    pos = (atoms.positions)
    pos[:,[choice[0], choice[1]]] = pos[:,[choice[1], choice[0]]]
    atoms.arrays["positions"] = pos
    #write("rotated_1.cif",atoms, format = "cif") # dont need to save 
    rotated_struct = AseAtomsAdaptor.get_structure(atoms) # rotational swapped object
    print(rotated_struct)

    crystal = Structure.from_file("original.cif")
    angle_to_rotate  = np.random.choice(360, 1, replace = False)
    rotate = RotationTransformation([1,1,1], angle_to_rotate)
    rotated = (rotate.apply_transformation(crystal))

    perturb = PerturbStructureTransformation(distance = 0.05)
    print(perturb.apply_transformation(crystal)) ### perturbed structure object
    print(rotate.apply_transformation(perturb.apply_transformation(crystal))) ### rotated structure object