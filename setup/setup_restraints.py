import logging
from importlib import reload

reload(logging)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p")

import glob as glob
import os as os
import subprocess as sp
import shutil as shutil
from itertools import compress
import numpy as np

import parmed as pmd
import paprika
from paprika.restraints import static_DAT_restraint
from paprika.restraints import DAT_restraint
from paprika.restraints.amber_restraints import amber_restraint_line
from paprika.restraints.restraints import create_window_list
from paprika.utils import make_window_dirs

logging.info("Started logging...")
logging.info("pAPRika version: " + paprika.__version__)


systems = ["a-coc-p"]

anchor_atoms = {
    "D1": f":8",
    "D2": f":9",
    "D3": f":10",
    "H1": f":1@O3",
    "H2": f":3@C1",
    "H3": f":5@C6",
    "G1": f":COC@C1",
    "G2": f":COC@O1",
}


def setup_static_restraints(
    anchor_atoms, windows, structure, distance_fc=5.0, angle_fc=100.0
):
    static_restraints = []
    static_restraint_atoms = [
        [anchor_atoms["D1"], anchor_atoms["H1"]],
        [anchor_atoms["D2"], anchor_atoms["D1"], anchor_atoms["H1"]],
        [anchor_atoms["D1"], anchor_atoms["H1"], anchor_atoms["H2"]],
        [
            anchor_atoms["D3"],
            anchor_atoms["D2"],
            anchor_atoms["D1"],
            anchor_atoms["H1"],
        ],
        [
            anchor_atoms["D2"],
            anchor_atoms["D1"],
            anchor_atoms["H1"],
            anchor_atoms["H2"],
        ],
        [
            anchor_atoms["D1"],
            anchor_atoms["H1"],
            anchor_atoms["H2"],
            anchor_atoms["H3"],
        ],
    ]

    for _, atoms in enumerate(static_restraint_atoms):
        this = static_DAT_restraint(
            restraint_mask_list=atoms,
            num_window_list=windows,
            ref_structure=structure,
            force_constant=angle_fc if len(atoms) > 2 else distance_fc,
            amber_index=True,
        )

        static_restraints.append(this)
    print(f"There are {len(static_restraints)} static restraints")
    return static_restraints


def setup_guest_restraints(
    anchor_atoms,
    windows,
    structure,
    distance_fc=5.0,
    angle_fc=100.0,
):
    guest_restraints = []

    guest_restraint_atoms = [
        [anchor_atoms["D1"], anchor_atoms["G1"]],
        [anchor_atoms["D2"], anchor_atoms["D1"], anchor_atoms["G1"]],
        [anchor_atoms["D1"], anchor_atoms["G1"], anchor_atoms["G2"]],
    ]
    guest_restraint_targets = {
        "initial": [6.0, 180.0, 180.0],
        "final": [24.0, 180.0, 180.0],
    }

    for index, atoms in enumerate(guest_restraint_atoms):
        if len(atoms) > 2:
            angle = True
        else:
            angle = False
        this = DAT_restraint()
        this.auto_apr = False
        this.amber_index = True
        this.topology = structure
        this.mask1 = atoms[0]
        this.mask2 = atoms[1]
        if angle:
            this.mask3 = atoms[2]
            this.attach["fc_final"] = angle_fc
        else:
            this.attach["fc_final"] = distance_fc
        this.attach["target"] = guest_restraint_targets["initial"][index]
        this.attach["fraction_list"] = attach_fractions

        this.initialize()

        guest_restraints.append(this)
    print(f"There are {len(guest_restraints)} guest restraints")
    return guest_restraints


def setup_conformation_restraints(
    template, targets, attach_fractions, structure, resname, fc=6.0
):

    conformational_restraints = []
    host_residues = len(structure[":{}".format(resname.upper())].residues)
    first_host_residue = structure[":{}".format(resname.upper())].residues[0].number + 1

    for n in range(first_host_residue, host_residues + first_host_residue):
        if n + 1 < host_residues + first_host_residue:
            next_residue = n + 1
        else:
            next_residue = first_host_residue

        for (index, atoms), target in zip(enumerate(template), targets):

            conformational_restraint_atoms = []
            if index == 0:
                conformational_restraint_atoms.append(f":{n}@{atoms[0]}")
                conformational_restraint_atoms.append(f":{n}@{atoms[1]}")
                conformational_restraint_atoms.append(f":{n}@{atoms[2]}")
                conformational_restraint_atoms.append(f":{next_residue}@{atoms[3]}")
            else:
                conformational_restraint_atoms.append(f":{n}@{atoms[0]}")
                conformational_restraint_atoms.append(f":{n}@{atoms[1]}")
                conformational_restraint_atoms.append(f":{next_residue}@{atoms[2]}")
                conformational_restraint_atoms.append(f":{next_residue}@{atoms[3]}")

            this = DAT_restraint()
            this.auto_apr = False
            this.amber_index = True
            this.topology = structure
            this.mask1 = conformational_restraint_atoms[0]
            this.mask2 = conformational_restraint_atoms[1]
            this.mask3 = conformational_restraint_atoms[2]
            this.mask4 = conformational_restraint_atoms[3]

            this.attach["fraction_list"] = attach_fractions
            this.attach["target"] = target
            this.attach["fc_final"] = fc

            this.initialize()
            conformational_restraints.append(this)
    print(f"There are {len(conformational_restraints)} conformational restraints")
    return conformational_restraints


def setup_guest_wall_restraints(
    template, targets, structure, windows, resname, angle_fc=500.0, distance_fc=50.0
):

    guest_wall_restraints = []
    host_residues = len(structure[":{}".format(resname.upper())].residues)
    first_host_residue = structure[":{}".format(resname.upper())].residues[0].number + 1

    for n in range(first_host_residue, host_residues + first_host_residue):
        for (index, atoms), target in zip(enumerate(template[0:2]), targets[0:2]):
            guest_wall_restraint_atoms = []
            guest_wall_restraint_atoms.append(f":{n}@{atoms[0]}")
            guest_wall_restraint_atoms.append(f"{atoms[1]}")

            this = DAT_restraint()
            this.auto_apr = False
            this.amber_index = True
            this.topology = structure
            this.mask1 = guest_wall_restraint_atoms[0]
            this.mask2 = guest_wall_restraint_atoms[1]
            this.attach["fc_initial"] = distance_fc
            this.attach["fc_final"] = distance_fc
            this.custom_restraint_values["rk2"] = 50.0
            this.custom_restraint_values["rk3"] = 50.0
            this.custom_restraint_values["r1"] = 0.0
            this.custom_restraint_values["r2"] = 0.0

            this.attach["target"] = target
            this.attach["num_windows"] = windows[0]

            this.initialize()
            guest_wall_restraints.append(this)
            print("Added guest wall distance restraint.")

    # Add a single angle restraint!
    guest_wall_restraint_atoms = []
    guest_wall_restraint_atoms.append(f"{template[2][0]}")
    guest_wall_restraint_atoms.append(f"{template[2][1]}")
    guest_wall_restraint_atoms.append(f"{template[2][2]}")
    target = targets[2]

    this = DAT_restraint()
    this.auto_apr = False
    this.amber_index = True
    this.topology = structure
    this.mask1 = guest_wall_restraint_atoms[0]
    this.mask2 = guest_wall_restraint_atoms[1]

    this.mask3 = guest_wall_restraint_atoms[2]
    this.attach["fc_initial"] = angle_fc
    this.attach["fc_final"] = angle_fc
    this.custom_restraint_values["rk2"] = 500.0
    this.custom_restraint_values["rk3"] = 0.0

    this.attach["target"] = target
    this.attach["num_windows"] = windows[0]

    this.initialize()
    guest_wall_restraints.append(this)
    print("Added guest wall angle restraint.")

    print(f"There are {len(guest_wall_restraints)} guest wall restraints")
    return guest_wall_restraints


attach_fractions = np.linspace(0, 1.0, 30)
pull_distances = []
release_fractions = []

windows = [len(attach_fractions), len(pull_distances), len(release_fractions)]
print(f"There are {windows} windows in this attach-pull-release calculation.")

for system in systems:
    structure = pmd.load_file(
        os.path.join("..", system, "smirnoff.prmtop"),
        os.path.join("..", system, "smirnoff.inpcrd"),
        structure=True,
    )
    static_restraints = setup_static_restraints(
        anchor_atoms, windows, os.path.join("..", system, "smirnoff.pdb"), distance_fc=5.0, angle_fc=100.0
    )

    guest_restraints = setup_guest_restraints(
        anchor_atoms,
        windows,
        structure,
        distance_fc=5.0,
        angle_fc=100.0,
    )
    host_conformational_template = [["O5", "C1", "O1", "C4"], ["C1", "O1", "C4", "C5"]]
    host_conformational_targets = [104.30, -108.8]
    conformational_restraints = setup_conformation_restraints(
        host_conformational_template,
        host_conformational_targets,
        attach_fractions,
        structure,
        resname="MGO",
        fc=6.0,
    )

    guest_wall_template = [
        ["O2", anchor_atoms["G1"]],
        ["O6", anchor_atoms["G1"]],
        [anchor_atoms["D2"], anchor_atoms["G1"], anchor_atoms["G2"]],
    ]
    if system[0] == "a":
        guest_wall_targets = [11.3, 13.3, 80.0]
    else:
        guest_wall_targets = [12.5, 14.5, 80.0]
    guest_wall_restraints = setup_guest_wall_restraints(
        guest_wall_template,
        guest_wall_targets,
        structure,
        resname="MGO",
        angle_fc=500.0,
        distance_fc=50.0,
    )

    restraints = (
        static_restraints
        + conformational_restraints
        + guest_restraints
        + guest_wall_restraints
    )

    window_list = create_window_list(guest_restraints)

    print("Writing restratint file in each window...")
    for window in window_list:
        if not os.path.exists(os.path.join("..", system, window)):
            os.makedirs(os.path.join("..", system, window))
        with open(
            os.path.join("..", system, window, "disang.rest"), "w"
        ) as file:
            if window[0] == "a":
                phase = "attach"
                restraints = (
                    static_restraints
                    + guest_restraints
                    + conformational_restraints
                    + guest_wall_restraints
                )
            if window[0] == "p":
                phase = "pull"
                restraints = (
                    static_restraints + conformational_restraints + guest_restraints
                )
            if window[0] == "r":
                phase = "release"
                restraints = (
                    static_restraints + conformational_restraints + guest_restraints
                )

            for restraint in restraints:
                string = amber_restraint_line(restraint, window)
                if string is not None:
                    file.write(string)
