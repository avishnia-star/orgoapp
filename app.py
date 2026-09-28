import io
import math
import os
import platform
import re

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    rdCIPLabeler,
    rdMolDescriptors,
    rdMolTransforms,
    rdPartialCharges,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


RDLogger.DisableLog("rdApp.*")


# ================================================================
# Colors and constants
# ================================================================

BLUE = (30, 90, 200)
RED = (200, 50, 50)
GREEN = (20, 140, 60)
PURPLE = (130, 40, 160)
GRAY = (110, 110, 110)
ORANGE = (230, 120, 0)
BLACK = (25, 25, 25)

SP = Chem.HybridizationType.SP
SP2 = Chem.HybridizationType.SP2
SP3 = Chem.HybridizationType.SP3

HYB = {
    SP: "sp",
    SP2: "sp²",
    SP3: "sp³",
}

ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts(
    "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"
)


# ================================================================
# Font handling
# ================================================================

def font(size, bold=False):
    """
    Find a font that exists across macOS, Windows, and Linux.
    """
    system = platform.system()

    candidates = []

    if system == "Darwin":
        candidates = [
            (
                "/System/Library/Fonts/Helvetica.ttc",
                1 if bold else 0,
            ),
            (
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                if bold
                else "/System/Library/Fonts/Supplemental/Arial.ttf",
                0,
            ),
        ]

    elif system == "Windows":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates = [
            (
                os.path.join(
                    windir,
                    "Fonts",
                    "arialbd.ttf" if bold else "arial.ttf",
                ),
                0,
            )
        ]

    else:
        candidates = [
            (
                "/usr/share/fonts/truetype/dejavu/"
                + (
                    "DejaVuSans-Bold.ttf"
                    if bold
                    else "DejaVuSans.ttf"
                ),
                0,
            ),
            (
                "/usr/share/fonts/truetype/liberation2/"
                + (
                    "LiberationSans-Bold.ttf"
                    if bold
                    else "LiberationSans-Regular.ttf"
                ),
                0,
            ),
        ]

    for path, index in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(
                    path,
                    size,
                    index=index,
                )
        except Exception:
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ================================================================
# Molecule-name handling
# ================================================================

COMMON_MOLECULES = {
    # Basic hydrocarbons
    "methane": "C",
    "ethane": "CC",
    "propane": "CCC",
    "butane": "CCCC",
    "pentane": "CCCCC",
    "hexane": "CCCCCC",
    "isobutane": "CC(C)C",
    "2-methylpropane": "CC(C)C",
    "neopentane": "CC(C)(C)C",
    "2,2-dimethylpropane": "CC(C)(C)C",

    # Alkenes and alkynes
    "ethene": "C=C",
    "ethylene": "C=C",
    "ethyne": "C#C",
    "acetylene": "C#C",
    "propene": "CC=C",
    "1-butene": "CCC=C",
    "2-butene": "CC=CC",
    "cis-2-butene": "C/C=C\\C",
    "trans-2-butene": "C/C=C/C",
    "(z)-2-butene": "C/C=C\\C",
    "(e)-2-butene": "C/C=C/C",
    "isobutylene": "CC(=C)C",
    "2-methylpropene": "CC(=C)C",
    "1,3-butadiene": "C=CC=C",
    "isoprene": "C=C(C)C=C",
    "propyne": "CC#C",
    "1-butyne": "CCC#C",
    "2-butyne": "CC#CC",

    # Cyclic compounds
    "cyclopropane": "C1CC1",
    "cyclobutane": "C1CCC1",
    "cyclopentane": "C1CCCC1",
    "cyclohexane": "C1CCCCC1",
    "cyclohexene": "C1=CCCCC1",
    "1,3-cyclohexadiene": "C1=CCCC=C1",
    "1,4-cyclohexadiene": "C1=CCC=CC1",

    # Aromatics
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "ethylbenzene": "CCc1ccccc1",
    "styrene": "C=Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1",
    "anisole": "COc1ccccc1",
    "pyridine": "n1ccccc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "o1cccc1",
    "thiophene": "s1cccc1",

    # Alcohols and ethers
    "methanol": "CO",
    "ethanol": "CCO",
    "1-propanol": "CCCO",
    "2-propanol": "CC(O)C",
    "isopropanol": "CC(O)C",
    "1-butanol": "CCCCO",
    "2-butanol": "CCC(O)C",
    "tert-butanol": "CC(C)(C)O",
    "tert-butyl alcohol": "CC(C)(C)O",
    "dimethyl ether": "COC",
    "methyl ethyl ether": "COCC",
    "ethyl methyl ether": "COCC",
    "diethyl ether": "CCOCC",
    "benzyl alcohol": "OCC1=CC=CC=C1",

    # Aldehydes and ketones
    "formaldehyde": "C=O",
    "methanal": "C=O",
    "acetaldehyde": "CC=O",
    "ethanal": "CC=O",
    "propanal": "CCC=O",
    "acetone": "CC(=O)C",
    "propanone": "CC(=O)C",
    "2-butanone": "CCC(=O)C",
    "acrolein": "C=CC=O",
    "benzaldehyde": "O=Cc1ccccc1",
    "acetophenone": "CC(=O)c1ccccc1",

    # Carboxylic acids and derivatives
    "formic acid": "C(=O)O",
    "methanoic acid": "C(=O)O",
    "acetic acid": "CC(=O)O",
    "ethanoic acid": "CC(=O)O",
    "propionic acid": "CCC(=O)O",
    "propanoic acid": "CCC(=O)O",
    "benzoic acid": "O=C(O)c1ccccc1",
    "methyl formate": "COC=O",
    "methyl acetate": "COC(=O)C",
    "ethyl acetate": "CCOC(=O)C",
    "acetamide": "CC(=O)N",
    "acetyl chloride": "CC(=O)Cl",
    "acetic anhydride": "CC(=O)OC(=O)C",

    # Nitrogen compounds
    "methylamine": "CN",
    "ethylamine": "CCN",
    "dimethylamine": "CNC",
    "trimethylamine": "CN(C)C",
    "acetonitrile": "CC#N",
    "acrylonitrile": "C=CC#N",
    "benzylamine": "NCC1=CC=CC=C1",
    "benzyl amine": "NCC1=CC=CC=C1",

    # Alkyl halides
    "chloromethane": "CCl",
    "bromomethane": "CBr",
    "iodomethane": "CI",
    "chloroethane": "CCCl",
    "bromoethane": "CCBr",
    "1-chloropropane": "CCCCl",
    "2-chloropropane": "CC(Cl)C",
    "tert-butyl chloride": "CC(C)(C)Cl",
    "benzyl chloride": "ClCC1=CC=CC=C1",
    "benzyl bromide": "BrCC1=CC=CC=C1",

    # Epoxides
    "oxirane": "C1CO1",
    "epoxide": "C1CO1",
    "ethylene oxide": "C1CO1",

    # Stereochemical examples
    "1,2-difluoroethene": "FC=CF",
    "cis-1,2-difluoroethene": "F/C=C\\F",
    "trans-1,2-difluoroethene": "F/C=C/F",
    "(z)-1,2-difluoroethene": "F/C=C\\F",
    "(e)-1,2-difluoroethene": "F/C=C/F",
    "cis-1,2-dichloroethene": "Cl/C=C\\Cl",
    "trans-1,2-dichloroethene": "Cl/C=C/Cl",
    "(z)-1,2-dichloroethene": "Cl/C=C\\Cl",
    "(e)-1,2-dichloroethene": "Cl/C=C/Cl",

    # Charged intermediates
    "tert-butyl cation": "CCC",
    "allyl cation": "C=C[CH2+]",
    "allyl anion": "C=C[CH2-]",
    "benzyl cation": "[CH2+]c1ccccc1",
    "benzyl anion": "[CH2-]c1ccccc1",
    "vinyl cation": "C=[CH+]",
    "vinyl anion": "C=[CH-]",
    "enolate": "C=C[O-]",
}


def normalize_name(name):
    name = name.strip().lower()
    name = name.replace("–", "-").replace("—", "-")
    name = re.sub(r"\s+", " ", name)
    return name


def name_to_smiles(name):
    """
    Interpret a supported common name or a valid SMILES string.

    PubChem lookup remains optional. The local dictionary and direct
    SMILES parsing work without an internet connection.
    """
    name = name.strip()

    if not name:
        raise ValueError("Enter a molecule name or SMILES string.")

    name_lower = normalize_name(name)

    if name_lower in COMMON_MOLECULES:
        return COMMON_MOLECULES[name_lower]

    mol = Chem.MolFromSmiles(name)

    if mol is not None:
        return name

    try:
        import pubchempy as pcp

        hits = pcp.get_compounds(name, "name")

        if hits:
            hit = hits[0]

            smiles = (
                getattr(hit, "isomeric_smiles", None)
                or getattr(hit, "connectivity_smiles", None)
                or getattr(hit, "canonical_smiles", None)
            )

            if smiles and Chem.MolFromSmiles(smiles) is not None:
                return smiles

    except Exception:
        pass

    raise ValueError(
        f"Unknown molecule '{name}'. Try a supported common name "
        "or a valid isomeric SMILES string."
    )


# ================================================================
# Chemical helpers
# ================================================================

def formal_charge(mol):
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def estimated_lone_pairs(atom):
    """
    Estimate Lewis-structure lone pairs for common main-group atoms.

    This is intentionally conservative. None means the simple estimate
    is not considered dependable for the atom.
    """
    atomic_number = atom.GetAtomicNum()

    if atomic_number == 1:
        return 0

    supported = {
        5,   # B
        6,   # C
        7,   # N
        8,   # O
        9,   # F
        14,  # Si
        15,  # P
        16,  # S
        17,  # Cl
        35,  # Br
        53,  # I
    }

    if atomic_number not in supported:
        return None

    try:
        valence_electrons = (
            Chem.GetPeriodicTable().GetNOuterElecs(atomic_number)
        )

        bond_order_sum = sum(
            bond.GetBondTypeAsDouble()
            for bond in atom.GetBonds()
        )

        nonbonding_electrons = (
            valence_electrons
            - atom.GetFormalCharge()
            - atom.GetNumRadicalElectrons()
            - int(round(bond_order_sum))
        )

        if nonbonding_electrons < 0:
            return None

        return max(0, nonbonding_electrons // 2)

    except Exception:
        return None


def lone_pairs(atom):
    value = estimated_lone_pairs(atom)
    return value if value is not None else 0


def stereo_label(mol):
    """
    Return only explicitly assigned E/Z and R/S labels.

    Unknown double-bond stereochemistry is not converted into E.
    """
    working = Chem.Mol(mol)

    Chem.AssignStereochemistry(
        working,
        cleanIt=True,
        force=True,
    )

    try:
        rdCIPLabeler.AssignCIPLabels(working)
    except Exception:
        pass

    labels = []

    for bond in working.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue

        stereo = bond.GetStereo()

        if stereo in (
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS,
        ):
            label = "Z"

        elif stereo in (
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOTRANS,
        ):
            label = "E"

        else:
            continue

        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()

        labels.append(
            f"{begin.GetSymbol()}{begin.GetIdx() + 1}="
            f"{end.GetSymbol()}{end.GetIdx() + 1}:{label}"
        )

    for atom in working.GetAtoms():
        if atom.HasProp("_CIPCode"):
            labels.append(
                f"{atom.GetSymbol()}{atom.GetIdx() + 1}:"
                f"{atom.GetProp('_CIPCode')}"
            )

    return labels


def stereo_identity(mol):
    labels = stereo_label(mol)

    if labels:
        return ", ".join(labels)

    return "input structure"


def enumerate_defined_stereoisomers(mol, maximum=16):
    """
    Enumerate stereoisomers without manually mutating bond stereo flags.

    Only unique, sanitized isomeric SMILES representations are returned.
    """
    options = StereoEnumerationOptions(
        onlyUnassigned=False,
        unique=True,
        maxIsomers=maximum,
        tryEmbedding=False,
    )

    unique = {}

    try:
        for isomer in EnumerateStereoisomers(
            Chem.Mol(mol),
            options=options,
        ):
            candidate = Chem.Mol(isomer)

            Chem.AssignStereochemistry(
                candidate,
                cleanIt=True,
                force=True,
            )

            try:
                rdCIPLabeler.AssignCIPLabels(candidate)
            except Exception:
                pass

            smiles = Chem.MolToSmiles(
                candidate,
                isomericSmiles=True,
            )

            unique[smiles] = candidate

    except Exception:
        return [Chem.Mol(mol)]

    if not unique:
        return [Chem.Mol(mol)]

    return list(unique.values())


def opposite_alkene_isomer(mol):
    """
    Find a stereoisomer with a different explicitly assigned E/Z label.

    This avoids directly changing the bond stereo enum without updating
    the complete stereochemical representation.
    """
    current_labels = {
        label
        for label in stereo_label(mol)
        if "=" in label and label.endswith((":E", ":Z"))
    }

    if not current_labels:
        return None

    current_smiles = Chem.MolToSmiles(
        mol,
        isomericSmiles=True,
    )

    for candidate in enumerate_defined_stereoisomers(mol):
        candidate_smiles = Chem.MolToSmiles(
            candidate,
            isomericSmiles=True,
        )

        if candidate_smiles == current_smiles:
            continue

        candidate_labels = {
            label
            for label in stereo_label(candidate)
            if "=" in label and label.endswith((":E", ":Z"))
        }

        if candidate_labels and candidate_labels != current_labels:
            return candidate

    return None


# ================================================================
# 3D conformers and force-field calculations
# ================================================================

def optimize_conformers(
    mol,
    num_conformers=20,
    seed=42,
):
    """
    Generate and optimize multiple conformers.

    Returns:
        optimized_molecule,
        list of dictionaries containing conformer id and energy,
        force-field label
    """
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol))

        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = 0.25
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.enforceChirality = True
        params.numThreads = 0

        heavy_atoms = mol.GetNumHeavyAtoms()

        actual_count = max(
            1,
            min(
                num_conformers,
                8 if heavy_atoms <= 4 else 20,
            ),
        )

        conformer_ids = list(
            AllChem.EmbedMultipleConfs(
                mol_h,
                numConfs=actual_count,
                params=params,
            )
        )

        if not conformer_ids:
            fallback = AllChem.EmbedMolecule(
                mol_h,
                randomSeed=seed,
                useRandomCoords=True,
            )

            if fallback != 0:
                return None, [], "embedding failed"

            conformer_ids = [0]

        results = []
        method = None

        mmff_properties = AllChem.MMFFGetMoleculeProperties(
            mol_h,
            mmffVariant="MMFF94s",
        )

        if mmff_properties is not None:
            optimization_results = AllChem.MMFFOptimizeMoleculeConfs(
                mol_h,
                numThreads=0,
                maxIters=2000,
                mmffVariant="MMFF94s",
            )

            method = "MMFF94s"

            for conformer_id, result in zip(
                conformer_ids,
                optimization_results,
            ):
                status, energy = result

                if math.isfinite(float(energy)):
                    results.append(
                        {
                            "conf_id": int(conformer_id),
                            "energy": float(energy),
                            "converged": status == 0,
                        }
                    )

        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            optimization_results = AllChem.UFFOptimizeMoleculeConfs(
                mol_h,
                numThreads=0,
                maxIters=2000,
            )

            method = "UFF"

            for conformer_id, result in zip(
                conformer_ids,
                optimization_results,
            ):
                status, energy = result

                if math.isfinite(float(energy)):
                    results.append(
                        {
                            "conf_id": int(conformer_id),
                            "energy": float(energy),
                            "converged": status == 0,
                        }
                    )

        else:
            return mol_h, [], "no supported force field"

        results.sort(key=lambda item: item["energy"])

        return mol_h, results, method

    except Exception:
        return None, [], "calculation failed"


def minimum_force_field_energy(
    mol,
    num_conformers=20,
    seed=42,
):
    mol_h, results, method = optimize_conformers(
        mol,
        num_conformers=num_conformers,
        seed=seed,
    )

    if not results:
        return mol_h, None, method, None

    converged_results = [
        result
        for result in results
        if result["converged"]
    ]

    selected = (
        converged_results[0]
        if converged_results
        else results[0]
    )

    return (
        mol_h,
        selected["energy"],
        method,
        selected["conf_id"],
    )


def get_3d_dipole(mol):
    """
    Estimate a conformer-specific dipole using Gasteiger point charges.

    Returns None for charged species, failed calculations, or invalid
    charges. It never converts failure into a zero dipole.
    """
    if formal_charge(mol) != 0:
        return None

    try:
        mol_h, energy, method, conformer_id = (
            minimum_force_field_energy(
                mol,
                num_conformers=20,
                seed=42,
            )
        )

        if (
            mol_h is None
            or conformer_id is None
            or mol_h.GetNumConformers() == 0
        ):
            return None

        rdPartialCharges.ComputeGasteigerCharges(mol_h)

        conformer = mol_h.GetConformer(conformer_id)

        mx = 0.0
        my = 0.0
        mz = 0.0

        for atom in mol_h.GetAtoms():
            if not atom.HasProp("_GasteigerCharge"):
                return None

            charge = float(
                atom.GetProp("_GasteigerCharge")
            )

            if not math.isfinite(charge):
                return None

            position = conformer.GetAtomPosition(
                atom.GetIdx()
            )

            mx += charge * position.x
            my += charge * position.y
            mz += charge * position.z

        electron_angstrom_to_debye = 4.803204712

        magnitude = (
            math.sqrt(
                mx * mx
                + my * my
                + mz * mz
            )
            * electron_angstrom_to_debye
        )

        if not math.isfinite(magnitude):
            return None

        return magnitude

    except Exception:
        return None


# ================================================================
# Conformer classification
# ================================================================

def wrap_angle(angle):
    angle = angle % 360.0

    if angle > 180.0:
        angle -= 360.0

    return angle


def torsion_class(angle):
    """
    Use broad descriptive bins instead of treating exact 60 degree
    grid points as universal conformer identities.
    """
    signed = wrap_angle(angle)
    absolute = abs(signed)

    if absolute >= 150:
        return "anti-like"

    if 30 <= absolute <= 90:
        return (
            "gauche-like +"
            if signed > 0
            else "gauche-like −"
        )

    if absolute < 30:
        return "syn-like"

    if 90 < absolute < 150:
        return "skew"

    return f"{signed:.0f}°"


def select_torsion_atoms(mol_h, atom_b, atom_c):
    """
    Choose one heavy-atom substituent on each side of a rotatable bond.

    Returns four atom indices or None.
    """
    begin = mol_h.GetAtomWithIdx(atom_b)
    end = mol_h.GetAtomWithIdx(atom_c)

    begin_candidates = [
        neighbor
        for neighbor in begin.GetNeighbors()
        if (
            neighbor.GetIdx() != atom_c
            and neighbor.GetAtomicNum() > 1
        )
    ]

    end_candidates = [
        neighbor
        for neighbor in end.GetNeighbors()
        if (
            neighbor.GetIdx() != atom_b
            and neighbor.GetAtomicNum() > 1
        )
    ]

    if not begin_candidates or not end_candidates:
        return None

    begin_candidates.sort(
        key=lambda atom: (
            atom.GetAtomicNum(),
            atom.GetDegree(),
        ),
        reverse=True,
    )

    end_candidates.sort(
        key=lambda atom: (
            atom.GetAtomicNum(),
            atom.GetDegree(),
        ),
        reverse=True,
    )

    return (
        begin_candidates[0].GetIdx(),
        atom_b,
        atom_c,
        end_candidates[0].GetIdx(),
    )


def find_first_rotatable_torsion(mol, mol_h):
    if ROTATABLE_BOND_SMARTS is None:
        return None

    matches = mol.GetSubstructMatches(
        ROTATABLE_BOND_SMARTS
    )

    for begin_idx, end_idx in matches:
        torsion = select_torsion_atoms(
            mol_h,
            begin_idx,
            end_idx,
        )

        if torsion is not None:
            return torsion

    return None


def conformer_minima(mol):
    """
    Sample multiple conformers and retain the lowest-energy member of
    each broad torsional class for the first suitable rotatable bond.
    """
    mol_h, results, method = optimize_conformers(
        mol,
        num_conformers=30,
        seed=42,
    )

    if mol_h is None or len(results) < 2:
        return [], method

    torsion = find_first_rotatable_torsion(
        mol,
        mol_h,
    )

    if torsion is None:
        return [], method

    atom_a, atom_b, atom_c, atom_d = torsion

    best_by_class = {}

    for result in results:
        conformer = mol_h.GetConformer(
            result["conf_id"]
        )

        angle = rdMolTransforms.GetDihedralDeg(
            conformer,
            atom_a,
            atom_b,
            atom_c,
            atom_d,
        )

        label = torsion_class(angle)

        candidate = {
            "label": label,
            "energy": result["energy"],
            "angle": angle,
            "converged": result["converged"],
        }

        if (
            label not in best_by_class
            or candidate["energy"]
            < best_by_class[label]["energy"]
        ):
            best_by_class[label] = candidate

    selected = sorted(
        best_by_class.values(),
        key=lambda item: item["energy"],
    )

    return selected[:5], method


# ================================================================
# Geometry and drawing helpers
# ================================================================

def unit(vector):
    magnitude = math.hypot(
        vector[0],
        vector[1],
    ) or 1.0

    return (
        vector[0] / magnitude,
        vector[1] / magnitude,
    )


def mid(point_a, point_b):
    return (
        (point_a[0] + point_b[0]) / 2,
        (point_a[1] + point_b[1]) / 2,
    )


def away_dir(positions, atom):
    point = positions[atom.GetIdx()]

    sum_x = 0.0
    sum_y = 0.0

    for neighbor in atom.GetNeighbors():
        neighbor_point = positions[neighbor.GetIdx()]

        direction = unit(
            (
                neighbor_point[0] - point[0],
                neighbor_point[1] - point[1],
            )
        )

        sum_x += direction[0]
        sum_y += direction[1]

    if abs(sum_x) + abs(sum_y) < 1e-6:
        return 0.0, -1.0

    return unit((-sum_x, -sum_y))


def ellipse_pts(center, direction, major, minor, n=28):
    ux, uy = direction
    vx, vy = -uy, ux

    points = []

    for index in range(n):
        angle = 2 * math.pi * index / n

        points.append(
            (
                center[0]
                + major * math.cos(angle) * ux
                + minor * math.sin(angle) * vx,
                center[1]
                + major * math.cos(angle) * uy
                + minor * math.sin(angle) * vy,
            )
        )

    return points


def arrow_head(
    draw,
    tip,
    direction,
    color,
    size=11,
):
    ux, uy = direction
    wx, wy = -uy, ux

    draw.polygon(
        [
            tip,
            (
                tip[0] - ux * size + wx * 6,
                tip[1] - uy * size + wy * 6,
            ),
            (
                tip[0] - ux * size - wx * 6,
                tip[1] - uy * size - wy * 6,
            ),
        ],
        fill=color,
    )


def curved_arrow(
    draw,
    point_0,
    point_2,
    color,
    label=None,
    selected_font=None,
    bulge=0.35,
    width=3,
    dashed=False,
):
    dx = point_2[0] - point_0[0]
    dy = point_2[1] - point_0[1]

    length = math.hypot(dx, dy) or 1.0

    nx = -dy / length
    ny = dx / length

    control = (
        (point_0[0] + point_2[0]) / 2
        + nx * length * bulge,
        (point_0[1] + point_2[1]) / 2
        + ny * length * bulge,
    )

    points = []

    for integer in range(41):
        t = integer / 40

        point = (
            (1 - t) ** 2 * point_0[0]
            + 2 * (1 - t) * t * control[0]
            + t * t * point_2[0],
            (1 - t) ** 2 * point_0[1]
            + 2 * (1 - t) * t * control[1]
            + t * t * point_2[1],
        )

        points.append(point)

    if dashed:
        for index in range(0, 40, 4):
            draw.line(
                points[index:index + 3],
                fill=color,
                width=width,
            )
    else:
        draw.line(
            points,
            fill=color,
            width=width,
            joint="curve",
        )

    arrow_head(
        draw,
        points[-1],
        unit(
            (
                points[-1][0] - points[-3][0],
                points[-1][1] - points[-3][1],
            )
        ),
        color,
    )

    if label:
        draw.text(
            (
                control[0] + nx * 10,
                control[1] + ny * 10,
            ),
            label,
            font=selected_font,
            fill=color,
            anchor="mm",
        )


# ================================================================
# Structure drawing and chemical annotations
# ================================================================

def draw_structure(mol, size):
    """
    Add explicit H atoms only for smaller molecules so lone pairs and
    selected C-H bonds remain visually usable.
    """
    if mol.GetNumHeavyAtoms() <= 12:
        drawing_molecule = Chem.AddHs(
            Chem.Mol(mol)
        )
    else:
        drawing_molecule = Chem.Mol(mol)

    try:
        AllChem.Compute2DCoords(drawing_molecule)
    except Exception:
        pass

    try:
        rdCIPLabeler.AssignCIPLabels(
            drawing_molecule
        )
    except Exception:
        pass

    drawer = rdMolDraw2D.MolDraw2DCairo(
        *size
    )

    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.bondLineWidth = 3
    options.padding = 0.24
    options.fixedBondLength = 85

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        drawing_molecule,
    )

    drawer.FinishDrawing()

    image = Image.open(
        io.BytesIO(drawer.GetDrawingText())
    ).convert("RGBA")

    positions = {
        index: (
            drawer.GetDrawCoords(index).x,
            drawer.GetDrawCoords(index).y,
        )
        for index in range(
            drawing_molecule.GetNumAtoms()
        )
    }

    return drawing_molecule, image, positions


def conjugation_candidates(drawing_molecule, positions):
    """
    Identify connectivity-based donor-acceptor possibilities.

    These are not asserted to be active hyperconjugative interactions.
    The 2D graph does not establish the required 3D orbital alignment.
    """
    candidates = []

    try:
        hydrogens = lambda atom: [
            neighbor
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        ]

        def c_h_midpoint(carbon):
            attached_hydrogens = hydrogens(carbon)

            if not attached_hydrogens:
                return positions[carbon.GetIdx()]

            return mid(
                positions[carbon.GetIdx()],
                positions[
                    attached_hydrogens[0].GetIdx()
                ],
            )

        unsaturated_atoms = {
            atom_index
            for bond in drawing_molecule.GetBonds()
            if bond.GetBondType()
            in (
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            )
            for atom_index in (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            )
        }

        # Possible sigma C-H donor adjacent to a p or pi center.
        for acceptor_atom in drawing_molecule.GetAtoms():
            if acceptor_atom.GetAtomicNum() != 6:
                continue

            is_cation_or_radical = (
                acceptor_atom.GetFormalCharge() > 0
                or acceptor_atom.GetNumRadicalElectrons() > 0
            )

            is_unsaturated = (
                acceptor_atom.GetIdx()
                in unsaturated_atoms
            )

            if not (
                is_cation_or_radical
                or is_unsaturated
            ):
                continue

            for donor_carbon in acceptor_atom.GetNeighbors():
                if (
                    donor_carbon.GetAtomicNum() == 6
                    and donor_carbon.GetHybridization() == SP3
                    and hydrogens(donor_carbon)
                ):
                    label = (
                        "possible σ(C-H)→p"
                        if is_cation_or_radical
                        else "possible σ(C-H)→π*"
                    )

                    candidates.append(
                        (
                            c_h_midpoint(donor_carbon),
                            positions[
                                acceptor_atom.GetIdx()
                            ],
                            GREEN,
                            label,
                            "sigma-pi",
                        )
                    )

        # Possible sigma donor into a neighboring C-X antibond.
        sigma_acceptor_pattern = Chem.MolFromSmarts(
            "[#6X4]-[F,Cl,Br,I,O,N]"
        )

        if sigma_acceptor_pattern is not None:
            for carbon_idx, x_idx in (
                drawing_molecule.GetSubstructMatches(
                    sigma_acceptor_pattern
                )
            ):
                carbon = (
                    drawing_molecule.GetAtomWithIdx(
                        carbon_idx
                    )
                )

                for neighboring_carbon in carbon.GetNeighbors():
                    if (
                        neighboring_carbon.GetIdx()
                        == x_idx
                    ):
                        continue

                    if (
                        neighboring_carbon.GetAtomicNum()
                        == 6
                        and hydrogens(
                            neighboring_carbon
                        )
                    ):
                        candidates.append(
                            (
                                c_h_midpoint(
                                    neighboring_carbon
                                ),
                                mid(
                                    positions[carbon_idx],
                                    positions[x_idx],
                                ),
                                PURPLE,
                                "possible σ(C-H)→σ*(C-X)",
                                "sigma-sigma",
                            )
                        )

        # Possible lone-pair conjugation with an adjacent pi system.
        lone_pair_pattern = Chem.MolFromSmarts(
            "[N,O,S,F,Cl,Br;!+]-"
            "[#6,#7,#8]=[#6,#7,#8]"
        )

        if lone_pair_pattern is not None:
            for hetero_idx, pi_start, pi_end in (
                drawing_molecule.GetSubstructMatches(
                    lone_pair_pattern
                )
            ):
                hetero_atom = (
                    drawing_molecule.GetAtomWithIdx(
                        hetero_idx
                    )
                )

                if estimated_lone_pairs(
                    hetero_atom
                ) in (None, 0):
                    continue

                candidates.append(
                    (
                        positions[hetero_idx],
                        mid(
                            positions[pi_start],
                            positions[pi_end],
                        ),
                        ORANGE,
                        "possible n/π conjugation",
                        "n-pi",
                    )
                )

    except Exception:
        return []

    unique = []
    seen = set()

    for candidate in candidates:
        signature = (
            round(candidate[0][0], 1),
            round(candidate[0][1], 1),
            round(candidate[1][0], 1),
            round(candidate[1][1], 1),
            candidate[4],
        )

        if signature not in seen:
            seen.add(signature)
            unique.append(candidate)

    return unique[:8]


def resonance_adjusted_hybridization(atom):
    """
    Use RDKit hybridization by default.

    Apply only narrow, explicit intro-organic resonance patterns instead
    of labeling every N or O attached to an sp2 atom as sp2.
    """
    if atom.GetIsAromatic():
        return "sp² (arom.)"

    hybridization = atom.GetHybridization()

    if hybridization == SP:
        return "sp"

    if hybridization == SP2:
        return "sp²"

    if hybridization == SP3:
        molecule = atom.GetOwningMol()
        atom_index = atom.GetIdx()

        explicit_patterns = [
            # Amide nitrogen
            "[N;X3;!+]-[C](=[O,S])",

niline-like nitrogen
            "[N;X3;!+]-[c]",

            # Enamine nitrogen
            "[N;X3;!+]-[C]=[C,N]",

            # Ester or carboxylic-acid single-bond oxygen
            "[O;X2]-[C](=O)",

# Enol or enol-ether oxygen
            "[O;X2]-[C]=[C,N]",

            # Phenolic or aryl-ether oxygen
            "[O;X2]-[c]",
        ]

        for smarts in explicit_patterns:
            pattern = Chem.MolFromSmarts(smarts)

            if pattern is None:
                continue

            for match in molecule.GetSubstructMatches(
                pattern
            ):
                if atom_index == match[0\]:
                    return "sp²-like"

        return "sp³"

    return ""


def approximate_2d_charge_direction(
    drawing_molecule,
    positions,
):
    """
    Estimate the projected direction toward the negative end using
    Gasteiger charges on the 2D drawing.

    This is used only as a qualitative 2D arrow direction. The displayed
    magnitude comes from a separately optimized 3D conformer.
    """
    try:
        working = Chem.Mol(drawing_molecule)

        rdPartialCharges.ComputeGasteigerCharges(
            working
        )

        charge_center_x = 0.0
        charge_center_y = 0.0

        for atom in working.GetAtoms():
            if not atom.HasProp("_GasteigerCharge"):
                return None

            charge = float(
                atom.GetProp("_GasteigerCharge")
            )

            if not math.isfinite(charge):
                return None

            point = positions[atom.GetIdx()]

            charge_center_x += charge * point[0]
            charge_center_y += charge * point[1]

        # Sum(q*r) follows the physical dipole-vector convention,
        # which points toward positive charge. The chemistry arrow with
        # a crossed positive tail points toward the negative end.
        toward_negative = (
            -charge_center_x,
            -charge_center_y,
        )

        if math.hypot(*toward_negative) < 1e-8:
            return None

        return unit(toward_negative)

    except Exception:
        return None


def annotate_structure(
    mol,
    dipole_magnitude,
    box_size,
):
    drawing_molecule, image, positions = (
        draw_structure(
            mol,
            box_size,
        )
    )

    label_font = font(15, True)
    small_font = font(13)

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    overlay_draw = ImageDraw.Draw(overlay)

    # Schematic p-lobes around atoms in pi systems.
    try:
        for bond in drawing_molecule.GetBonds():
            if bond.GetBondType() not in (
                Chem.BondType.DOUBLE,
                Chem.BondType.AROMATIC,
            ):
                continue

            for atom, other in (
                (
                    bond.GetBeginAtom(),
                    bond.GetEndAtom(),
                ),
                (
                    bond.GetEndAtom(),
                    bond.GetBeginAtom(),
                ),
            ):
                point = positions[atom.GetIdx()]
                other_point = positions[
                    other.GetIdx()
                ]

                bond_direction = unit(
                    (
                        other_point[0] - point[0],
                        other_point[1] - point[1],
                    )
                )

                perpendicular = (
                    -bond_direction[1],
                    bond_direction[0],
                )

                for sign, color in (
                    (1, (60, 110, 230, 95)),
                    (-1, (230, 80, 80, 95)),
                ):
                    center = (
                        point[0]
                        + sign
                        * perpendicular[0]
                        * 26,
                        point[1]
                        + sign
                        * perpendicular[1]
                        * 26,
                    )

                    overlay_draw.polygon(
                        ellipse_pts(
                            center,
                            (
                                sign * perpendicular[0],
                                sign * perpendicular[1],
                            ),
                            22,
                            11,
                        ),
                        fill=color,
                        outline=color[:3] + (200,),
                    )

    except Exception:
        pass

    image = Image.alpha_composite(
        image,
        overlay,
    )

    draw = ImageDraw.Draw(image)

    # Lone pairs and hybridization.
    try:
        for atom in drawing_molecule.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue

            point = positions[atom.GetIdx()]
            outward = away_dir(
                positions,
                atom,
            )

            lone_pair_count = (
                estimated_lone_pairs(atom)
            )

            if lone_pair_count:
                base_angle = math.atan2(
                    outward[1],
                    outward[0],
                )

                offsets = {
                    1: [0],
                    2: [-50, 50],
                    3: [-75, 0, 75],
                    4: [-90, -30, 30, 90],
                }.get(
                    min(lone_pair_count, 4),
                    [0],
                )

                for offset in offsets:
                    angle = (
                        base_angle
                        + math.radians(offset)
                    )

                    center_x = (
                        point[0]
                        + 26 * math.cos(angle)
                    )

                    center_y = (
                        point[1]
                        + 26 * math.sin(angle)
                    )

                    for spacing in (-4.5, 4.5):
                        x = (
                            center_x
                            - spacing
                            * math.sin(angle)
                        )

                        y = (
                            center_y
                            + spacing
                            * math.cos(angle)
                        )

                        draw.ellipse(
                            (
                                x - 3,
                                y - 3,
                                x + 3,
                                y + 3,
                            ),
                            fill=RED,
                        )

            tag = resonance_adjusted_hybridization(
                atom
            )

            if atom.GetAtomicNum() not in (
                6,
                7,
                8,
            ):
                tag = ""

            if atom.GetFormalCharge():
                charge_text = (
                    f"{atom.GetFormalCharge():+d}"
                )

                tag = (
                    f"{tag} {charge_text}"
                    if tag
                    else charge_text
                )

            if tag:
                radius = (
                    48
                    if lone_pair_count
                    else 30
                )

                draw.text(
                    (
                        point[0]
                        + outward[0] * radius,
                        point[1]
                        + outward[1] * radius,
                    ),
                    tag.strip(),
                    font=label_font,
                    fill=BLUE,
                    anchor="mm",
                )

    except Exception:
        pass

    # Possible orbital interactions based only on connectivity.
    candidates = conjugation_candidates(
        drawing_molecule,
        positions,
    )

    try:
        for index, candidate in enumerate(
            candidates
        ):
            (
                point_0,
                point_2,
                color,
                label,
                interaction_type,
            ) = candidate

            curved_arrow(
                draw,
                point_0,
                point_2,
                color,
                label,
                small_font,
                bulge=(
                    0.35
                    if index % 2 == 0
                    else -0.35
                ),
            )

    except Exception:
        pass

    # Dipole box.
    width, height = image.size

    box_x = width - 170
    box_y = height - 110

    draw.rectangle(
        (
            box_x,
            box_y,
            width - 12,
            height - 12,
        ),
        outline=GRAY,
    )

    draw.text(
        (
            box_x + 8,
            box_y + 6,
        ),
        "dipole μ",
        font=label_font,
        fill=BLACK,
    )

    if formal_charge(mol) != 0:
        draw.text(
            (
                box_x + 15,
                box_y + 45,
            ),
            "N/A",
            font=font(22, True),
            fill=BLACK,
        )

        draw.text(
            (
                box_x + 8,
                height - 32,
            ),
            "(charged species)",
            font=small_font,
            fill=GRAY,
        )

    elif dipole_magnitude is None:
        draw.text(
            (
                box_x + 15,
                box_y + 45,
            ),
            "N/A",
            font=font(22, True),
            fill=BLACK,
        )

        draw.text(
            (
                box_x + 8,
                height - 32,
            ),
            "estimate unavailable",
            font=small_font,
            fill=GRAY,
        )

    elif dipole_magnitude > 0.15:
        direction = (
            approximate_2d_charge_direction(
                drawing_molecule,
                positions,
            )
        )

        if direction is not None:
            center = (
                box_x + 80,
                box_y + 65,
            )

            tip = (
                center[0]
                + direction[0] * 45,
                center[1]
                + direction[1] * 45,
            )

            tail = (
                center[0]
                - direction[0] * 45,
                center[1]
                - direction[1] * 45,
            )

            draw.line(
                (
                    tail,
                    tip,
                ),
                fill=BLACK,
                width=3,
            )

            arrow_head(
                draw,
                tip,
                direction,
                BLACK,
            )

            draw.line(
                (
                    tail[0]
                    - direction[1] * 7,
                    tail[1]
                    + direction[0] * 7,
                    tail[0]
                    + direction[1] * 7,
                    tail[1]
                    - direction[0] * 7,
                ),
                fill=BLACK,
                width=3,
            )
        else:
            draw.text(
                (
                    box_x + 28,
                    box_y + 45,
                ),
                "polar",
                font=font(20, True),
                fill=BLACK,
            )

        draw.text(
            (
                box_x + 8,
                height - 32,
            ),
            f"≈ {dipole_magnitude:.1f} D*",
            font=small_font,
            fill=GRAY,
        )

    else:
        draw.text(
            (
                box_x + 30,
                box_y + 45,
            ),
            "μ ≈ 0",
            font=font(22, True),
            fill=BLACK,
        )

        draw.text(
            (
                box_x + 8,
                height - 32,
            ),
            "point-charge est.",
            font=small_font,
            fill=GRAY,
        )

    # Legend.
    legend_x = 14
    legend_y = height - 100

    draw.polygon(
        ellipse_pts(
            (
                legend_x + 14,
                legend_y + 8,
            ),
            (0, -1),
            10,
            5,
        ),
        fill=(60, 110, 230, 120),
    )

    draw.text(
        (
            legend_x + 32,
            legend_y,
        ),
        "schematic p orbital",
        font=small_font,
        fill=BLACK,
    )

    draw.ellipse(
        (
            legend_x + 8,
            legend_y + 22,
            legend_x + 14,
            legend_y + 28,
        ),
        fill=RED,
    )

    draw.ellipse(
        (
            legend_x + 17,
            legend_y + 22,
            legend_x + 23,
            legend_y + 28,
        ),
        fill=RED,
    )

    draw.text(
        (
            legend_x + 32,
            legend_y + 18,
        ),
        "estimated lone pair",
        font=small_font,
        fill=BLACK,
    )

    legend_items = (
        (
            GREEN,
            "possible σ→π*/p",
        ),
        (
            PURPLE,
            "possible σ→σ*",
        ),
        (
            ORANGE,
            "possible n/π conjugation",
        ),
    )

    for index, (
        color,
        text,
    ) in enumerate(legend_items):
        y = legend_y + 44 + index * 18

        draw.line(
            (
                legend_x + 4,
                y,
                legend_x + 26,
                y,
            ),
            fill=color,
            width=3,
        )

        draw.text(
            (
                legend_x + 32,
                legend_y + 36 + index * 18,
            ),
            text,
            font=small_font,
            fill=BLACK,
        )

    return image.convert("RGB"), candidates


# ================================================================
# Energy analysis
# ================================================================

def energy_items(mol):
    """
    Produce conservative force-field energy information.

    No hard-coded experimental cis or gauche energies are used.
    """
    try:
        current_label = stereo_identity(mol)

        current_h, current_energy, method, conf_id = (
            minimum_force_field_energy(
                mol,
                num_conformers=24,
                seed=42,
            )
        )

        opposite = opposite_alkene_isomer(mol)

        if (
            opposite is not None
            and current_energy is not None
        ):
            (
                opposite_h,
                opposite_energy,
                opposite_method,
                opposite_conf_id,
            ) = minimum_force_field_energy(
                opposite,
                num_conformers=24,
                seed=314,
            )

            if (
                opposite_energy is not None
                and opposite_method == method
            ):
                return (
                    [
                        (
                            current_label,
                            current_energy,
                            True,
                        ),
                        (
                            stereo_identity(opposite),
                            opposite_energy,
                            False,
                        ),
                    ],
                    (
                        "sampled stereoisomers "
                        f"({method})"
                    ),
                    (
                        "Lowest sampled force-field conformer; "
                        "not experimental ΔG."
                    ),
                )

        conformers, conformer_method = (
            conformer_minima(mol)
        )

        if len(conformers) > 1:
            items = []

            for index, conformer in enumerate(
                conformers
            ):
                label = (
                    f"{conformer['label']} "
                    f"({conformer['angle'\]:+.0f}°)"
                )

                items.append(
                    (
                        label,
                        conformer["energy"],
                        index == 0,
                    )
                )

            return (
                items,
                (
                    "sampled conformers "
                    f"({conformer_method})"
                ),
                (
                    "Relative force-field energies for "
                    "sampled conformers only."
                ),
            )

        if current_energy is not None:
            return (
                [
                    (
                        current_label,
                        current_energy,
                        True,
                    )
                ],
                f"one sampled minimum ({method})",
                (
                    "Absolute force-field energy is not "
                    "an experimental stability value."
                ),
            )

        return (
            [
                (
                    "not available",
                    0.0,
                    False,
                )
            ],
            "calculation unavailable",
            (
                "No supported force-field result was obtained."
            ),
        )

    except Exception:
        return (
            [
                (
                    "not available",
                    0.0,
                    False,
                )
            ],
            "calculation unavailable",
            (
                "No supported force-field result was obtained."
            ),
        )


def draw_energy_panel(
    image,
    box,
    items,
    kind,
    note,
):
    draw = ImageDraw.Draw(image)

    x0, y0, x1, y1 = box

    header_font = font(17, True)
    small_font = font(13)

    draw.rectangle(
        box,
        outline=GRAY,
    )

    draw.text(
        (
            x0 + 12,
            y0 + 8,
        ),
        f"Energy  ({kind})",
        font=header_font,
        fill=BLACK,
    )

    try:
        energies = [
            energy
            for _, energy, _ in items
        ]

        minimum_energy = min(energies)

        relative_energies = [
            energy - minimum_energy
            for energy in energies
        ]

        span = max(
            max(relative_energies),
            1.0,
        )

        top = y0 + 65
        bottom = y1 - 85

        item_count = max(
            len(items),
            1,
        )

        available_width = x1 - x0 - 70
        item_width = (
            available_width / item_count
        )

        draw.line(
            (
                x0 + 30,
                bottom + 15,
                x0 + 30,
                top - 10,
            ),
            fill=BLACK,
            width=2,
        )

        arrow_head(
            draw,
            (
                x0 + 30,
                top - 10,
            ),
            (0, -1),
            BLACK,
        )

        draw.text(
            (
                x0 + 18,
                top - 28,
            ),
            "E",
            font=small_font,
            fill=BLACK,
        )

        for index, (
            label,
            energy,
            current,
        ) in enumerate(items):
            relative = energy - minimum_energy

            y = (
                bottom
                - relative
                / span
                * (bottom - top)
            )

            start_x = (
                x0
                + 50
                + index * item_width
            )

            end_x = (
                start_x
                + item_width
                - 20
            )

            color = BLUE if current else GRAY

            draw.line(
                (
                    start_x,
                    y,
                    end_x,
                    y,
                ),
                fill=color,
                width=6,
            )

            draw.text(
                (
                    (start_x + end_x) / 2,
                    y - 6,
                ),
                f"{relative:+.2f} kcal/mol",
                font=small_font,
                fill=color,
                anchor="mb",
            )

            display_label = label

            if len(display_label) > 23:
                display_label = (
                    display_label[:20]
                    + "..."
                )

            draw.text(
                (
                    (start_x + end_x) / 2,
                    y + 8,
                ),
                display_label
                + (
                    "  ◀ input"
                    if current
                    else ""
                ),
                font=small_font,
                fill=color,
                anchor="mt",
            )

        if note:
            wrapped_note = (
                note
                if len(note) <= 62
                else note[:59] + "..."
            )

            draw.text(
                (
                    x0 + 12,
                    y1 - 42,
                ),
                wrapped_note,
                font=small_font,
                fill=RED,
            )

    except Exception:
        draw.text(
            (
                x0 + 20,
                y0 + 80,
            ),
            "Energy analysis unavailable.",
            font=small_font,
            fill=RED,
        )


# ================================================================
# Qualitative orbital panel
# ================================================================

def draw_mo_panel(
    image,
    box,
    mol,
    candidates,
):
    """
    Draw a schematic donor-acceptor panel.

    It deliberately does not label calculated HOMO/LUMO orbitals or
    imply that the vertical positions are computed orbital energies.
    """
    draw = ImageDraw.Draw(image)

    x0, y0, x1, y1 = box

    header_font = font(17, True)
    small_font = font(13)

    draw.rectangle(
        box,
        outline=GRAY,
    )

    draw.text(
        (
            x0 + 12,
            y0 + 8,
        ),
        "Orbital picture (qualitative)",
        font=header_font,
        fill=BLACK,
    )

    try:
        has_pi = (
            mol.HasSubstructMatch(
                Chem.MolFromSmarts(
                    "[#6,#7,#8]=[#6,#7,#8]"
                )
            )
            or mol.HasSubstructMatch(
                Chem.MolFromSmarts(
                    "[#6,#7]#[#6,#7]"
                )
            )
            or any(
                atom.GetIsAromatic()
                for atom in mol.GetAtoms()
            )
        )

        has_cx = mol.HasSubstructMatch(
            Chem.MolFromSmarts(
                "[#6]-[F,Cl,Br,I,O,N]"
            )
        )

        has_lone_pair = any(
            (
                estimated_lone_pairs(atom)
                or 0
            ) > 0
            for atom in mol.GetAtoms()
        )

        candidate_types = {
            candidate[4]
            for candidate in candidates
        }

        panel_height = y1 - y0

        levels = {}

        def level(
            name,
            fraction,
            x,
            color,
            occupied,
            tag="",
        ):
            y = y0 + fraction * panel_height

            draw.line(
                (
                    x,
                    y,
                    x + 90,
                    y,
                ),
                fill=color,
                width=4,
            )

            levels[name] = (
                x + 45,
                y,
            )

            draw.text(
                (
                    x + 95,
                    y,
                ),
                name + tag,
                font=small_font,
                fill=color,
                anchor="lm",
            )

            if occupied:
                for index, symbol in enumerate(
                    ("↑", "↓")
                ):
                    draw.text(
                        (
                            x
                            + 30
                            + index * 22,
                            y - 3,
                        ),
                        symbol,
                        font=font(20, True),
                        fill=color,
                        anchor="mb",
                    )

        left = x0 + 30
        middle = x0 + 180
        right = x0 + 340

        if has_pi:
            level(
                "π* acceptor",
                0.22,
                middle,
                BLUE,
                False,
            )

            level(
                "π donor",
                0.65,
                middle,
                BLUE,
                True,
            )

        else:
            level(
                "σ* acceptor",
                0.22,
                middle,
                BLUE,
                False,
            )

            level(
                "σ bond",
                0.80,
                middle,
                BLUE,
                True,
            )

        if has_cx:
            level(
                "σ*(C-X)",
                0.35,
                right,
                PURPLE,
                False,
            )

        if has_lone_pair:
            level(
                "n lone pair",
                0.50,
                left,
                ORANGE,
                True,
            )

        level(
            "σ(C-H)",
            0.88 if has_pi else 0.76,
            left,
            GREEN,
            True,
        )

        def link(
            donor,
            acceptor,
            color,
        ):
            if (
                donor in levels
                and acceptor in levels
            ):
                curved_arrow(
                    draw,
                    levels[donor],
                    levels[acceptor],
                    color,
                    None,
                    None,
                    bulge=0.25,
                    width=2,
                    dashed=True,
                )

        if (
            "sigma-pi"
            in candidate_types
            and has_pi
        ):
            link(
                "σ(C-H)",
                "π* acceptor",
                GREEN,
            )

        if (
            "sigma-sigma"
            in candidate_types
            and has_cx
        ):
            link(
                "σ(C-H)",
                "σ*(C-X)",
                PURPLE,
            )

        if (
            "n-pi"
            in candidate_types
            and has_pi
        ):
            link(
                "n lone pair",
                "π* acceptor",
                ORANGE,
            )

        draw.text(
            (
                x0 + 12,
                y1 - 58,
            ),
            "dashed = possible donor→acceptor overlap",
            font=small_font,
            fill=GRAY,
        )

        draw.text(
            (
                x0 + 12,
                y1 - 40,
            ),
            "connectivity only; 3D alignment not tested",
            font=small_font,
            fill=GRAY,
        )

        draw.text(
            (
                x0 + 12,
                y1 - 22,
            ),
            "levels are schematic, not computed HOMO/LUMO",
            font=small_font,
            fill=GRAY,
        )

    except Exception:
        draw.text(
            (
                x0 + 20,
                y0 + 70,
            ),
            "Orbital summary unavailable.",
            font=small_font,
            fill=RED,
        )


# ================================================================
# Study-card generation
# ================================================================

def make_card(name):
    smiles = name_to_smiles(name)

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Could not parse molecule: {name}"
        )

    Chem.AssignStereochemistry(
        mol,
        cleanIt=True,
        force=True,
    )

    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        pass

    width = 1500
    height = 960

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(image)

    dipole_magnitude = get_3d_dipole(mol)

    structure, candidates = annotate_structure(
        mol,
        dipole_magnitude,
        (
            920,
            840,
        ),
    )

    image.paste(
        structure,
        (
            25,
            95,
        ),
    )

    draw.rectangle(
        (
            25,
            95,
            945,
            935,
        ),
        outline=GRAY,
    )

    items, kind, note = energy_items(mol)

    draw_energy_panel(
        image,
        (
            965,
            95,
            1475,
            470,
        ),
        items,
        kind,
        note,
    )

    draw_mo_panel(
        image,
        (
            965,
            490,
            1475,
            935,
        ),
        mol,
        candidates,
    )

    try:
        mol_h = Chem.AddHs(mol)

        sigma_count = mol_h.GetNumBonds()

        kekule = Chem.Mol(mol_h)

        try:
            Chem.Kekulize(
                kekule,
                clearAromaticFlags=True,
            )
        except Exception:
            pass

        pi_count = sum(
            {
                Chem.BondType.DOUBLE: 1,
                Chem.BondType.TRIPLE: 2,
            }.get(
                bond.GetBondType(),
                0,
            )
            for bond in kekule.GetBonds()
        )

        draw.text(
            (
                25,
                14,
            ),
            name,
            font=font(30, True),
            fill=BLACK,
        )

        draw.text(
            (
                730,
                26,
            ),
            (
                "Qualitative auto-generated study aid: "
                "verify resonance, conformations, and mechanisms."
            ),
            font=font(14, True),
            fill=RED,
        )

        display_smiles = (
            smiles
            if len(smiles) <= 40
            else smiles[:37] + "..."
        )

        stereo_text = "  ".join(
            stereo_label(mol)
        )

        if not stereo_text:
            stereo_text = "stereo: none assigned"

        subtitle = (
            f"{rdMolDescriptors.CalcMolFormula(mol)}   "
            f"{Descriptors.MolWt(mol):.1f} g/mol   "
            f"σ {sigma_count}  "
            f"π {pi_count:g}   "
            f"{stereo_text}   "
            f"{display_smiles}"
        )

        draw.text(
            (
                25,
                58,
            ),
            subtitle,
            font=font(16),
            fill=GRAY,
        )

    except Exception:
        pass

    return image


@st.cache_data(show_spinner=False)
def get_card_image_bytes(name):
    image = make_card(name)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ================================================================
# Streamlit interface
# ================================================================

def safe_filename(name):
    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        name.strip(),
    )

    cleaned = cleaned.strip("._")

    return cleaned or "orgo_card"


def main():
    st.set_page_config(
        page_title="OrgoCard",
        layout="wide",
    )

    st.title("OrgoCard Generator")

    st.caption(
        "Structure annotations are automated study aids. "
        "Energy values are sampled force-field energies, not "
        "experimental thermodynamic data."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        name = st.text_input(
            "Molecule name or SMILES:",
            value="cis-1,2-difluoroethene",
        )

    with col2:
        st.write("")
        st.write("")

        draw_requested = st.button(
            "Draw",
            type="primary",
            use_container_width=True,
        )

    if draw_requested:
        with st.spinner(
            "Generating conformers and study card..."
        ):
            try:
                png_bytes = get_card_image_bytes(
                    name
                )

                st.image(
                    png_bytes,
                    use_container_width=True,
                )

                st.download_button(
                    label="Download PNG",
                    data=png_bytes,
                    file_name=(
                        f"{safe_filename(name)}.png"
                    ),
                    mime="image/png",
                )

                st.info(
                    "The dipole is a Gasteiger point-charge estimate "
                    "from a sampled 3D conformer. Energy comparisons "
                    "use the lowest sampled MMFF94s or UFF conformer. "
                    "Possible orbital interactions are based on "
                    "connectivity and do not prove favorable 3D overlap."
                )

            except Exception as error:
                st.error(
                    "Error generating molecule: "
                    f"{error}"
                )


if __name__ == "__main__":
    main()
