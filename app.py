import io
import math
import os
import platform
import re

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdCIPLabeler, rdMolDescriptors, rdPartialCharges
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

BLUE, RED, GREEN, PURPLE, GRAY, ORANGE, BLACK = (
    (30, 90, 200), (200, 50, 50), (20, 140, 60), (130, 40, 160),
    (110, 110, 110), (230, 120, 0), (25, 25, 25),
)
SP = Chem.HybridizationType.SP
SP2 = Chem.HybridizationType.SP2
SP3 = Chem.HybridizationType.SP3
CACHE_VERSION = "2026-09-28-v6"


def font(size, bold=False):
    system = platform.system()
    if system == "Darwin":
        candidates = [("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0)]
    elif system == "Windows":
        root = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [(os.path.join(root, "Fonts", "arialbd.ttf" if bold else "arial.ttf"), 0)]
    else:
        candidates = [
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
            ("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", 0),
        ]
    for path, index in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, size, index=index)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


COMMON = {
    "methane": "C", "ethane": "CC", "propane": "CCC", "butane": "CCCC",
    "ethene": "C=C", "ethylene": "C=C", "ethyne": "C#C", "acetylene": "C#C",
    "propene": "CC=C", "1-butene": "CCC=C", "2-butene": "CC=CC",
    "cis-2-butene": "C/C=C\\C", "trans-2-butene": "C/C=C/C",
    "(z)-2-butene": "C/C=C\\C", "(e)-2-butene": "C/C=C/C",
    "2-methylpropene": "CC(=C)C", "isobutylene": "CC(=C)C",
    "1,3-butadiene": "C=CC=C", "isoprene": "C=C(C)C=C",
    "propyne": "CC#C", "1-butyne": "CCC#C", "2-butyne": "CC#CC",
    "cyclopropane": "C1CC1", "cyclobutane": "C1CCC1", "cyclopentane": "C1CCCC1",
    "cyclohexane": "C1CCCCC1", "cyclohexene": "C1=CCCCC1",
    "benzene": "c1ccccc1", "toluene": "Cc1ccccc1", "phenol": "Oc1ccccc1",
    "aniline": "Nc1ccccc1", "anisole": "COc1ccccc1", "pyridine": "n1ccccc1",
    "pyrrole": "c1cc[nH]c1", "furan": "o1cccc1", "thiophene": "s1cccc1",
    "methanol": "CO", "ethanol": "CCO", "1-propanol": "CCCO",
    "2-propanol": "CC(O)C", "isopropanol": "CC(O)C", "tert-butanol": "CC(C)(C)O",
    "dimethyl ether": "COC", "methyl ethyl ether": "COCC", "diethyl ether": "CCOCC",
    "formaldehyde": "C=O", "acetaldehyde": "CC=O", "acetone": "CC(=O)C",
    "acrolein": "C=CC=O", "benzaldehyde": "O=Cc1ccccc1",
    "formic acid": "C(=O)O", "acetic acid": "CC(=O)O", "propionic acid": "CCC(=O)O",
    "benzoic acid": "O=C(O)c1ccccc1", "methyl acetate": "COC(=O)C",
    "ethyl acetate": "CCOC(=O)C", "acetamide": "CC(=O)N",
    "methylamine": "CN", "ethylamine": "CCN", "dimethylamine": "CNC",
    "trimethylamine": "CN(C)C", "acetonitrile": "CC#N", "acrylonitrile": "C=CC#N",
    "chloromethane": "CCl", "bromomethane": "CBr", "iodomethane": "CI",
    "chloroethane": "CCCl", "bromoethane": "CCBr",
    "oxirane": "C1CO1", "epoxide": "C1CO1", "ethylene oxide": "C1CO1",
    "benzyl alcohol": "OCC1=CC=CC=C1", "benzyl chloride": "ClCC1=CC=CC=C1",
    "benzyl bromide": "BrCC1=CC=CC=C1", "benzylamine": "NCC1=CC=CC=C1",
    "1,2-difluoroethene": "FC=CF", "cis-1,2-difluoroethene": "F/C=C\\F",
    "trans-1,2-difluoroethene": "F/C=C/F", "(z)-1,2-difluoroethene": "F/C=C\\F",
    "(e)-1,2-difluoroethene": "F/C=C/F", "cis-1,2-dichloroethene": "Cl/C=C\\Cl",
    "trans-1,2-dichloroethene": "Cl/C=C/Cl",
    "tert-butyl cation": "C[C+](C)C", "allyl cation": "C=C[CH2+]",
    "allyl anion": "C=C[CH2-]", "benzyl cation": "[CH2+]c1ccccc1",
    "benzyl anion": "[CH2-]c1ccccc1", "vinyl cation": "C=[CH+]",
    "vinyl anion": "C=[CH-]", "enolate": "C=C[O-]",
}


def name_to_smiles(text):
    text = text.strip()
    key = re.sub(r"\s+", " ", text.lower().replace("–", "-").replace("—", "-"))
    if key in COMMON:
        return COMMON[key]
    if Chem.MolFromSmiles(text) is not None:
        return text
    try:
        import pubchempy as pcp
        hits = pcp.get_compounds(text, "name")
        if hits:
            value = getattr(hits[0], "isomeric_smiles", None) or getattr(hits[0], "canonical_smiles", None)
            if value and Chem.MolFromSmiles(value) is not None:
                return value
    except Exception:
        pass
    raise ValueError(f"Unknown molecule '{text}'. Enter a supported name or valid isomeric SMILES.")


def formal_charge(mol):
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def lone_pairs(atom):
    if atom.GetAtomicNum() == 1:
        return 0
    if atom.GetAtomicNum() not in {5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}:
        return None
    try:
        valence_electrons = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
        bond_order = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        electrons = valence_electrons - atom.GetFormalCharge() - atom.GetNumRadicalElectrons() - int(round(bond_order))
        return None if electrons < 0 else max(0, electrons // 2)
    except Exception:
        return None


def stereo_label(mol):
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    try:
        rdCIPLabeler.AssignCIPLabels(work)
    except Exception:
        pass
    labels = []
    for bond in work.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            value = "Z"
        elif stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            value = "E"
        else:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        labels.append(f"{a.GetSymbol()}{a.GetIdx()+1}={b.GetSymbol()}{b.GetIdx()+1}:{value}")
    for atom in work.GetAtoms():
        if atom.HasProp("_CIPCode"):
            labels.append(f"{atom.GetSymbol()}{atom.GetIdx()+1}:{atom.GetProp('_CIPCode')}")
    return labels


def optimized_conformers(mol, count=20, seed=42):
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = 0.25
        params.enforceChirality = True
        params.numThreads = 0
        ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=max(1, count), params=params))
        if not ids:
            return None, [], "embedding failed"
        properties = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if properties is not None:
            raw = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=2000, mmffVariant="MMFF94s")
            method = "MMFF94s"
        elif AllChem.UFFHasAllMoleculeParams(mol_h):
            raw = AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=2000)
            method = "UFF"
        else:
            return mol_h, [], "no supported force field"
        results = []
        for conformer_id, (status, energy) in zip(ids, raw):
            if math.isfinite(float(energy)):
                results.append({"id": int(conformer_id), "energy": float(energy), "converged": status == 0})
        results.sort(key=lambda item: (not item["converged"], item["energy"]))
        return mol_h, results, method
    except Exception:
        return None, [], "calculation failed"


def minimum_energy(mol, seed=42):
    mol_h, results, method = optimized_conformers(mol, 20, seed)
    if not results:
        return mol_h, None, method, None
    best = results[0]
    return mol_h, best["energy"], method, best["id"]


def get_3d_dipole(mol):
    if formal_charge(mol) != 0:
        return None
    mol_h, _, _, conformer_id = minimum_energy(mol)
    if mol_h is None or conformer_id is None:
        return None
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol_h)
        conformer = mol_h.GetConformer(conformer_id)
        mx = my = mz = 0.0
        for atom in mol_h.GetAtoms():
            charge = float(atom.GetProp("_GasteigerCharge"))
            if not math.isfinite(charge):
                return None
            point = conformer.GetAtomPosition(atom.GetIdx())
            mx += charge * point.x
            my += charge * point.y
            mz += charge * point.z
        return math.sqrt(mx * mx + my * my + mz * mz) * 4.803204712
    except Exception:
        return None


def unit(vector):
    magnitude = math.hypot(vector[0], vector[1]) or 1.0
    return vector[0] / magnitude, vector[1] / magnitude


def mid(point_1, point_2):
    return (point_1[0] + point_2[0]) / 2, (point_1[1] + point_2[1]) / 2


def away_dir(positions, atom):
    point = positions[atom.GetIdx()]
    sx = sy = 0.0
    for neighbor in atom.GetNeighbors():
        direction = unit((positions[neighbor.GetIdx()][0] - point[0], positions[neighbor.GetIdx()][1] - point[1]))
        sx += direction[0]
        sy += direction[1]
    return (0, -1) if abs(sx) + abs(sy) < 1e-6 else unit((-sx, -sy))


def ellipse_pts(center, direction, major, minor, count=28):
    ux, uy = direction
    vx, vy = -uy, ux
    return [
        (center[0] + major * math.cos(angle) * ux + minor * math.sin(angle) * vx,
         center[1] + major * math.cos(angle) * uy + minor * math.sin(angle) * vy)
        for angle in (2 * math.pi * index / count for index in range(count))
    ]


def arrow_head(draw, tip, direction, color, size=11):
    ux, uy = direction
    wx, wy = -uy, ux
    draw.polygon([
        tip,
        (tip[0] - ux * size + wx * 6, tip[1] - uy * size + wy * 6),
        (tip[0] - ux * size - wx * 6, tip[1] - uy * size - wy * 6),
    ], fill=color)


def curved_arrow(draw, start, end, color, label=None, selected_font=None, bulge=0.35, width=3, dashed=False):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    control = ((start[0] + end[0]) / 2 + nx * length * bulge, (start[1] + end[1]) / 2 + ny * length * bulge)
    points = [
        ((1-t)**2*start[0] + 2*(1-t)*t*control[0] + t*t*end[0],
         (1-t)**2*start[1] + 2*(1-t)*t*control[1] + t*t*end[1])
        for t in (index / 40 for index in range(41))
    ]
    if dashed:
        for index in range(0, 40, 4):
            draw.line(points[index:index+3], fill=color, width=width)
    else:
        draw.line(points, fill=color, width=width, joint="curve")
    arrow_head(draw, points[-1], unit((points[-1][0]-points[-3][0], points[-1][1]-points[-3][1])), color)
    if label:
        draw.text((control[0] + nx*10, control[1] + ny*10), label, font=selected_font, fill=color, anchor="mm")


def draw_structure(mol, size):
    drawing_mol = Chem.AddHs(Chem.Mol(mol)) if mol.GetNumHeavyAtoms() <= 12 else Chem.Mol(mol)
    AllChem.Compute2DCoords(drawing_mol)
    try:
        rdCIPLabeler.AssignCIPLabels(drawing_mol)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.bondLineWidth = 3
    options.padding = 0.24
    options.fixedBondLength = 85
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, drawing_mol)
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    positions = {index: (drawer.GetDrawCoords(index).x, drawer.GetDrawCoords(index).y) for index in range(drawing_mol.GetNumAtoms())}
    return drawing_mol, image, positions


def is_amide_nitrogen(atom):
    if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() > 0:
        return False
    for carbon in atom.GetNeighbors():
        if carbon.GetAtomicNum() != 6:
            continue
        for bond in carbon.GetBonds():
            other = bond.GetOtherAtom(carbon)
            if other.GetIdx() != atom.GetIdx() and other.GetAtomicNum() in (8, 16) and bond.GetBondType() == Chem.BondType.DOUBLE:
                return True
    return False


def is_aryl_donor(atom):
    if atom.GetAtomicNum() not in (7, 8, 16) or atom.GetFormalCharge() > 0 or not (lone_pairs(atom) or 0):
        return False
    return any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors())


def is_allylic_cation_atom(atom):
    if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() <= 0:
        return False
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() != 6:
            continue
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            if other.GetIdx() != atom.GetIdx() and other.GetAtomicNum() == 6 and bond.GetBondType() == Chem.BondType.DOUBLE:
                return True
    return False


def hybridization_label(atom):
    if atom.GetIsAromatic():
        return "sp² (arom.)"
    if is_amide_nitrogen(atom) or is_aryl_donor(atom):
        return "sp²-like"
    if is_allylic_cation_atom(atom):
        return "sp²"
    return {SP: "sp", SP2: "sp²", SP3: "sp³"}.get(atom.GetHybridization(), "")


def p_orbital_atoms(molecule):
    indices = set()
    for atom in molecule.GetAtoms():
        if atom.GetIsAromatic() or is_amide_nitrogen(atom) or is_aryl_donor(atom) or is_allylic_cation_atom(atom):
            indices.add(atom.GetIdx())
    for bond in molecule.GetBonds():
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            indices.add(bond.GetBeginAtomIdx())
            indices.add(bond.GetEndAtomIdx())
    return indices


def aromatic_orbital_direction(molecule, positions, atom):
    aromatic_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetIsAromatic()]
    if len(aromatic_neighbors) >= 2:
        first = positions[aromatic_neighbors[0].GetIdx()]
        second = positions[aromatic_neighbors[1].GetIdx()]
        ring_tangent = unit((second[0] - first[0], second[1] - first[1]))
        return (-ring_tangent[1], ring_tangent[0])
    return away_dir(positions, atom)


def orbital_direction(molecule, positions, atom):
    if atom.GetIsAromatic():
        return aromatic_orbital_direction(molecule, positions, atom)
    conjugated_neighbors = [
        neighbor for neighbor in atom.GetNeighbors()
        if neighbor.GetIsAromatic() or neighbor.GetHybridization() == SP2 or neighbor.GetFormalCharge() != 0
    ]
    if conjugated_neighbors:
        point = positions[atom.GetIdx()]
        target = positions[conjugated_neighbors[0].GetIdx()]
        bond_direction = unit((target[0] - point[0], target[1] - point[1]))
        return (-bond_direction[1], bond_direction[0])
    return (0, -1)


def interaction_candidates(molecule, positions):
    candidates = []
    try:
        amide = Chem.MolFromSmarts("[N;X3;!+]-[C](=[O,S])")
        for nitrogen, carbon, oxygen in molecule.GetSubstructMatches(amide):
            candidates.append((positions[nitrogen], mid(positions[nitrogen], positions[carbon]), ORANGE, "amide resonance", "n-pi"))
            candidates.append((mid(positions[carbon], positions[oxygen]), positions[oxygen], ORANGE, None, "resonance-shift"))

        for atom in molecule.GetAtoms():
            if is_aryl_donor(atom):
                aromatic_neighbor = next(neighbor for neighbor in atom.GetNeighbors() if neighbor.GetIsAromatic())
                candidates.append((positions[atom.GetIdx()], positions[aromatic_neighbor.GetIdx()], ORANGE, "n→aromatic π", "aryl-n-pi"))

        allyl_pattern = Chem.MolFromSmarts("[C,c]=[C,c]-[C+]")
        for terminal, center, cation in molecule.GetSubstructMatches(allyl_pattern):
            candidates.append((mid(positions[terminal], positions[center]), mid(positions[center], positions[cation]), PURPLE, "allylic π→p", "allyl-pi-p"))

        non_aromatic = Chem.MolFromSmarts("[N,O,S;!+]-[#6,#7,#8]=[#6,#7,#8]")
        for hetero, carbon_1, carbon_2 in molecule.GetSubstructMatches(non_aromatic):
            atom = molecule.GetAtomWithIdx(hetero)
            if is_amide_nitrogen(atom) or is_aryl_donor(atom):
                continue
            if (lone_pairs(atom) or 0) > 0:
                candidates.append((positions[hetero], mid(positions[carbon_1], positions[carbon_2]), ORANGE, "possible n/π conjugation", "n-pi"))
    except Exception:
        pass
    return candidates[:8]


def projected_dipole_direction(molecule, positions):
    try:
        work = Chem.Mol(molecule)
        rdPartialCharges.ComputeGasteigerCharges(work)
        sx = sy = 0.0
        for atom in work.GetAtoms():
            charge = float(atom.GetProp("_GasteigerCharge"))
            if not math.isfinite(charge):
                return None
            sx += charge * positions[atom.GetIdx()][0]
            sy += charge * positions[atom.GetIdx()][1]
        return None if math.hypot(sx, sy) < 1e-8 else unit((-sx, -sy))
    except Exception:
        return None


def annotate_structure(mol, dipole_magnitude, box_size):
    drawing_mol, image, positions = draw_structure(mol, box_size)
    label_font, small_font = font(15, True), font(13)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Exactly one p-orbital pair per conjugated atom.
    for atom_index in sorted(p_orbital_atoms(drawing_mol)):
        atom = drawing_mol.GetAtomWithIdx(atom_index)
        point = positions[atom_index]
        direction = orbital_direction(drawing_mol, positions, atom)
        for sign, color in ((1, (60, 110, 230, 95)), (-1, (230, 80, 80, 95))):
            center = (point[0] + sign*direction[0]*26, point[1] + sign*direction[1]*26)
            overlay_draw.polygon(ellipse_pts(center, (sign*direction[0], sign*direction[1]), 22, 11), fill=color, outline=color[:3] + (200,))

    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    for atom in drawing_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        point = positions[atom.GetIdx()]
        outward = away_dir(positions, atom)
        pair_count = lone_pairs(atom)
        if pair_count:
            base_angle = math.atan2(outward[1], outward[0])
            offsets = {1: [0], 2: [-50, 50], 3: [-75, 0, 75], 4: [-90, -30, 30, 90]}.get(min(pair_count, 4), [0])
            for offset in offsets:
                angle = base_angle + math.radians(offset)
                cx, cy = point[0] + 26*math.cos(angle), point[1] + 26*math.sin(angle)
                for spacing in (-4.5, 4.5):
                    x = cx - spacing*math.sin(angle)
                    y = cy + spacing*math.cos(angle)
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=RED)
        tag = hybridization_label(atom) if atom.GetAtomicNum() in (6, 7, 8) else ""
        if atom.GetFormalCharge():
            tag = (tag + " " if tag else "") + f"{atom.GetFormalCharge():+d}"
        if tag:
            radius = 54 if pair_count else 34
            draw.text((point[0] + outward[0]*radius, point[1] + outward[1]*radius), tag, font=label_font, fill=BLUE, anchor="mm")

    candidates = interaction_candidates(drawing_mol, positions)
    for index, (start, end, color, label, _) in enumerate(candidates):
        curved_arrow(draw, start, end, color, label, small_font, 0.32 if index % 2 == 0 else -0.32)

    if any(candidate[4] == "allyl-pi-p" for candidate in candidates):
        draw.text((18, 18), "allylic positive charge is resonance-delocalized", font=small_font, fill=PURPLE)

    width, height = image.size
    bx, by = width-170, height-110
    draw.rectangle((bx, by, width-12, height-12), outline=GRAY)
    draw.text((bx+8, by+6), "dipole μ", font=label_font, fill=BLACK)
    if formal_charge(mol) != 0:
        draw.text((bx+15, by+45), "N/A", font=font(22, True), fill=BLACK)
        draw.text((bx+8, height-32), "(charged species)", font=small_font, fill=GRAY)
    elif dipole_magnitude is None:
        draw.text((bx+15, by+45), "N/A", font=font(22, True), fill=BLACK)
        draw.text((bx+8, height-32), "estimate unavailable", font=small_font, fill=GRAY)
    elif dipole_magnitude > 0.15:
        direction = projected_dipole_direction(drawing_mol, positions)
        if direction:
            center = (bx+80, by+56)
            tip = (center[0] + direction[0]*32, center[1] + direction[1]*32)
            tail = (center[0] - direction[0]*32, center[1] - direction[1]*32)
            draw.line((tail, tip), fill=BLACK, width=3)
            arrow_head(draw, tip, direction, BLACK)
            draw.line((tail[0]-direction[1]*7, tail[1]+direction[0]*7, tail[0]+direction[1]*7, tail[1]-direction[0]*7), fill=BLACK, width=3)
        draw.text((bx+8, height-43), f"≈ {dipole_magnitude:.1f} D", font=small_font, fill=GRAY)
        draw.text((bx+8, height-27), "Gasteiger estimate", font=font(11), fill=GRAY)
    else:
        draw.text((bx+30, by+45), "μ ≈ 0", font=font(22, True), fill=BLACK)
        draw.text((bx+8, height-32), "point-charge est.", font=small_font, fill=GRAY)

    lx, ly = 14, height-100
    draw.polygon(ellipse_pts((lx+14, ly+8), (0, -1), 10, 5), fill=(60, 110, 230, 120))
    draw.text((lx+32, ly), "schematic p orbital", font=small_font, fill=BLACK)
    draw.ellipse((lx+8, ly+22, lx+14, ly+28), fill=RED)
    draw.ellipse((lx+17, ly+22, lx+23, ly+28), fill=RED)
    draw.text((lx+32, ly+18), "estimated lone pair", font=small_font, fill=BLACK)
    draw.line((lx+4, ly+44, lx+26, ly+44), fill=PURPLE, width=3)
    draw.text((lx+32, ly+36), "π→p / resonance conjugation", font=small_font, fill=BLACK)
    draw.line((lx+4, ly+62, lx+26, ly+62), fill=ORANGE, width=3)
    draw.text((lx+32, ly+54), "lone-pair / π conjugation", font=small_font, fill=BLACK)
    return image.convert("RGB"), candidates


def energy_items(mol):
    assigned_ez = any("=" in label and label.endswith((":E", ":Z")) for label in stereo_label(mol))
    if assigned_ez:
        return [], "stereoisomer comparison suppressed", "MMFF/UFF may not reproduce experimental cis effects."
    _, energy, method, _ = minimum_energy(mol, 42)
    if energy is not None:
        return [], f"one optimized conformer ({method})", "No comparable second conformer; relative energy is not shown."
    return [], "calculation unavailable", "No supported force-field result was obtained."


def draw_energy_panel(image, box, items, kind, note):
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    header_font, small_font = font(17, True), font(13)
    draw.rectangle(box, outline=GRAY)
    draw.text((x0+12, y0+8), f"Energy  ({kind})", font=header_font, fill=BLACK)
    draw.text((x0+28, y0+145), "No relative-energy comparison", font=font(19, True), fill=GRAY)
    if "stereoisomer" in kind:
        message = "E/Z force-field ranking is intentionally hidden."
        detail = "Use curated experimental data for cis/trans stability."
    else:
        message = "Only one optimized state is available."
        detail = "A self-relative +0.00 level is not displayed."
    draw.text((x0+28, y0+180), message, font=small_font, fill=GRAY)
    draw.text((x0+28, y0+202), detail, font=small_font, fill=GRAY)
    draw.text((x0+12, y1-42), note if len(note) <= 62 else note[:59] + "...", font=small_font, fill=RED)


def draw_mo_panel(image, box, mol, candidates):
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    header_font, small_font = font(17, True), font(13)
    draw.rectangle(box, outline=GRAY)
    draw.text((x0+12, y0+8), "Orbital picture (qualitative)", font=header_font, fill=BLACK)

    kinds = {candidate[4] for candidate in candidates}
    has_pi = any(bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC) for bond in mol.GetBonds())
    levels = {}
    panel_height = y1-y0

    def level(name, fraction, x, color, occupied):
        y = y0 + fraction*panel_height
        draw.line((x, y, x+90, y), fill=color, width=4)
        levels[name] = (x+45, y)
        draw.text((x+95, y), name, font=small_font, fill=color, anchor="lm")
        if occupied:
            draw.text((x+30, y-3), "↑", font=font(20, True), fill=color, anchor="mb")
            draw.text((x+52, y-3), "↓", font=font(20, True), fill=color, anchor="mb")

    left, middle = x0+30, x0+220

    if not has_pi and not kinds:
        draw.text((x0+28, y0+175), "Localized σ-bond framework", font=font(18, True), fill=BLUE)
        draw.text((x0+28, y0+210), "No conjugated π system detected.", font=small_font, fill=GRAY)
        draw.text((x0+28, y0+232), "No donor-acceptor interaction assigned.", font=small_font, fill=GRAY)
    elif "allyl-pi-p" in kinds:
        level("empty p acceptor", 0.25, middle, PURPLE, False)
        level("π donor", 0.70, left, BLUE, True)
        curved_arrow(draw, levels["π donor"], levels["empty p acceptor"], PURPLE, dashed=True, width=2)
        draw.text((x0+28, y0+335), "allylic charge is resonance-delocalized", font=small_font, fill=PURPLE)
    else:
        if has_pi:
            level("π* acceptor", 0.25, middle, BLUE, False)
            level("π donor", 0.68, middle, BLUE, True)
        if "n-pi" in kinds or "aryl-n-pi" in kinds:
            donor_name = "n(N) donor" if any(atom.GetAtomicNum() == 7 and (is_amide_nitrogen(atom) or is_aryl_donor(atom)) for atom in mol.GetAtoms()) else "n donor"
            level(donor_name, 0.48, left, ORANGE, True)
            if "π* acceptor" in levels:
                curved_arrow(draw, levels[donor_name], levels["π* acceptor"], ORANGE, dashed=True, width=2)
        if not ("n-pi" in kinds or "aryl-n-pi" in kinds):
            draw.text((x0+28, y0+325), "No specific donor-acceptor interaction assigned.", font=small_font, fill=GRAY)

    draw.text((x0+12, y1-58), "dashed = possible donor→acceptor overlap", font=small_font, fill=GRAY)
    draw.text((x0+12, y1-40), "connectivity only; 3D alignment not tested", font=small_font, fill=GRAY)
    draw.text((x0+12, y1-22), "schematic levels, not computed HOMO/LUMO", font=small_font, fill=GRAY)


def make_card(name):
    smiles = name_to_smiles(name)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse molecule: {name}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        pass

    width, height = 1500, 960
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    dipole = get_3d_dipole(mol)
    structure, candidates = annotate_structure(mol, dipole, (920, 840))
    image.paste(structure, (25, 95))
    draw.rectangle((25, 95, 945, 935), outline=GRAY)
    items, kind, note = energy_items(mol)
    draw_energy_panel(image, (965, 95, 1475, 470), items, kind, note)
    draw_mo_panel(image, (965, 490, 1475, 935), mol, candidates)

    mol_h = Chem.AddHs(mol)
    sigma_count = mol_h.GetNumBonds()
    kekule = Chem.Mol(mol_h)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:
        pass
    pi_count = sum({Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}.get(bond.GetBondType(), 0) for bond in kekule.GetBonds())

    draw.text((25, 14), name, font=font(30, True), fill=BLACK)
    draw.text((730, 26), "Qualitative auto-generated study aid: verify resonance, conformations, and mechanisms.", font=font(14, True), fill=RED)
    shown_smiles = smiles if len(smiles) <= 40 else smiles[:37] + "..."
    stereo = "  ".join(stereo_label(mol)) or "stereo: none assigned"
    subtitle = f"{rdMolDescriptors.CalcMolFormula(mol)}   {Descriptors.MolWt(mol):.1f} g/mol   σ {sigma_count}  π {pi_count:g}   {stereo}   {shown_smiles}"
    draw.text((25, 58), subtitle, font=font(16), fill=GRAY)
    return image


@st.cache_data(show_spinner=False)
def get_card_image_bytes(name, cache_version):
    image = make_card(name)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def safe_filename(name):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or "orgo_card"


def main():
    st.set_page_config(page_title="OrgoCard", layout="wide")
    st.title("OrgoCard Generator")
    st.caption("Structure annotations are automated study aids. Energy values are sampled force-field energies, not experimental thermodynamic data.")
    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input("Molecule name or SMILES:", value="cis-1,2-difluoroethene")
    with col2:
        st.write("")
        st.write("")
        requested = st.button("Draw", type="primary", use_container_width=True)
    if requested:
        with st.spinner("Generating conformers and study card..."):
            try:
                png = get_card_image_bytes(name, CACHE_VERSION)
                st.image(png, use_container_width=True)
                st.download_button("Download PNG", png, f"{safe_filename(name)}.png", "image/png")
                st.info("Dipole values are Gasteiger point-charge estimates. E/Z force-field rankings are suppressed. Orbital interactions are qualitative possibilities inferred from connectivity.")
            except Exception as error:
                st.error(f"Error generating molecule: {error}")


if __name__ == "__main__":
    main()
