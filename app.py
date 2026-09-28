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
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

RDLogger.DisableLog("rdApp.*")

BLUE, RED, GREEN, PURPLE, GRAY, ORANGE, BLACK = (
    (30, 90, 200), (200, 50, 50), (20, 140, 60), (130, 40, 160),
    (110, 110, 110), (230, 120, 0), (25, 25, 25),
)
SP, SP2, SP3 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3


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
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())


def lone_pairs(atom):
    if atom.GetAtomicNum() == 1:
        return 0
    if atom.GetAtomicNum() not in {5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}:
        return None
    try:
        ve = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
        bond_order = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        electrons = ve - atom.GetFormalCharge() - atom.GetNumRadicalElectrons() - int(round(bond_order))
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
    out = []
    for bond in work.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        st = bond.GetStereo()
        if st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            label = "Z"
        elif st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            label = "E"
        else:
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        out.append(f"{a.GetSymbol()}{a.GetIdx()+1}={b.GetSymbol()}{b.GetIdx()+1}:{label}")
    for atom in work.GetAtoms():
        if atom.HasProp("_CIPCode"):
            out.append(f"{atom.GetSymbol()}{atom.GetIdx()+1}:{atom.GetProp('_CIPCode')}")
    return out


def opposite_alkene_isomer(mol):
    current = {x.rsplit(":", 1)[-1] for x in stereo_label(mol) if "=" in x}
    if not current:
        return None
    options = StereoEnumerationOptions(onlyUnassigned=False, unique=True, maxIsomers=32, tryEmbedding=False)
    current_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    try:
        for isomer in EnumerateStereoisomers(Chem.Mol(mol), options=options):
            candidate = Chem.Mol(isomer)
            Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
            if Chem.MolToSmiles(candidate, isomericSmiles=True) == current_smiles:
                continue
            labels = {x.rsplit(":", 1)[-1] for x in stereo_label(candidate) if "=" in x}
            if labels and labels != current:
                return candidate
    except Exception:
        pass
    return None


def optimized_conformers(mol, count=24, seed=42):
    try:
        mh = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = 0.25
        params.enforceChirality = True
        params.numThreads = 0
        ids = list(AllChem.EmbedMultipleConfs(mh, numConfs=max(1, count), params=params))
        if not ids:
            return None, [], "embedding failed"
        props = AllChem.MMFFGetMoleculeProperties(mh, mmffVariant="MMFF94s")
        if props is not None:
            raw = AllChem.MMFFOptimizeMoleculeConfs(mh, numThreads=0, maxIters=2000, mmffVariant="MMFF94s")
            method = "MMFF94s"
        elif AllChem.UFFHasAllMoleculeParams(mh):
            raw = AllChem.UFFOptimizeMoleculeConfs(mh, numThreads=0, maxIters=2000)
            method = "UFF"
        else:
            return mh, [], "no supported force field"
        results = []
        for cid, (status, energy) in zip(ids, raw):
            if math.isfinite(float(energy)):
                results.append({"id": int(cid), "energy": float(energy), "converged": status == 0})
        results.sort(key=lambda x: (not x["converged"], x["energy"]))
        return mh, results, method
    except Exception:
        return None, [], "calculation failed"


def minimum_energy(mol, seed=42):
    mh, results, method = optimized_conformers(mol, 24, seed)
    if not results:
        return mh, None, method, None
    best = results[0]
    return mh, best["energy"], method, best["id"]


def get_3d_dipole(mol):
    if formal_charge(mol) != 0:
        return None
    mh, _, _, cid = minimum_energy(mol)
    if mh is None or cid is None:
        return None
    try:
        rdPartialCharges.ComputeGasteigerCharges(mh)
        conf = mh.GetConformer(cid)
        mx = my = mz = 0.0
        for atom in mh.GetAtoms():
            q = float(atom.GetProp("_GasteigerCharge"))
            if not math.isfinite(q):
                return None
            p = conf.GetAtomPosition(atom.GetIdx())
            mx += q * p.x
            my += q * p.y
            mz += q * p.z
        return math.sqrt(mx * mx + my * my + mz * mz) * 4.803204712
    except Exception:
        return None


def unit(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return v[0] / n, v[1] / n


def mid(p, q):
    return (p[0] + q[0]) / 2, (p[1] + q[1]) / 2


def away_dir(pos, atom):
    p = pos[atom.GetIdx()]
    sx = sy = 0.0
    for nbr in atom.GetNeighbors():
        u = unit((pos[nbr.GetIdx()][0] - p[0], pos[nbr.GetIdx()][1] - p[1]))
        sx += u[0]
        sy += u[1]
    return (0, -1) if abs(sx) + abs(sy) < 1e-6 else unit((-sx, -sy))


def ellipse_pts(center, direction, major, minor, n=28):
    ux, uy = direction
    vx, vy = -uy, ux
    return [(center[0] + major * math.cos(t) * ux + minor * math.sin(t) * vx,
             center[1] + major * math.cos(t) * uy + minor * math.sin(t) * vy)
            for t in (2 * math.pi * i / n for i in range(n))]


def arrow_head(draw, tip, direction, color, size=11):
    ux, uy = direction
    wx, wy = -uy, ux
    draw.polygon([tip, (tip[0] - ux * size + wx * 6, tip[1] - uy * size + wy * 6),
                  (tip[0] - ux * size - wx * 6, tip[1] - uy * size - wy * 6)], fill=color)


def curved_arrow(draw, p0, p2, color, label=None, selected_font=None, bulge=0.35, width=3, dashed=False):
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    control = ((p0[0] + p2[0]) / 2 + nx * length * bulge, (p0[1] + p2[1]) / 2 + ny * length * bulge)
    pts = [((1-t)**2*p0[0] + 2*(1-t)*t*control[0] + t*t*p2[0],
            (1-t)**2*p0[1] + 2*(1-t)*t*control[1] + t*t*p2[1]) for t in (i/40 for i in range(41))]
    if dashed:
        for i in range(0, 40, 4):
            draw.line(pts[i:i+3], fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width, joint="curve")
    arrow_head(draw, pts[-1], unit((pts[-1][0]-pts[-3][0], pts[-1][1]-pts[-3][1])), color)
    if label:
        draw.text((control[0] + nx*10, control[1] + ny*10), label, font=selected_font, fill=color, anchor="mm")


def draw_structure(mol, size):
    md = Chem.AddHs(Chem.Mol(mol)) if mol.GetNumHeavyAtoms() <= 12 else Chem.Mol(mol)
    AllChem.Compute2DCoords(md)
    try:
        rdCIPLabeler.AssignCIPLabels(md)
    except Exception:
        pass
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 3
    opts.padding = 0.24
    opts.fixedBondLength = 85
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, md)
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    pos = {i: (drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y) for i in range(md.GetNumAtoms())}
    return md, image, pos


def hybridization_label(atom):
    if atom.GetIsAromatic():
        return "sp² (arom.)"
    return {SP: "sp", SP2: "sp²", SP3: "sp³"}.get(atom.GetHybridization(), "")


def interaction_candidates(md, pos):
    candidates = []
    try:
        pi_atoms = {i for b in md.GetBonds() if b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
                    for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
        for atom in md.GetAtoms():
            if atom.GetIdx() not in pi_atoms and atom.GetFormalCharge() <= 0:
                continue
            for carbon in atom.GetNeighbors():
                hydrogens = [n for n in carbon.GetNeighbors() if n.GetAtomicNum() == 1]
                if carbon.GetAtomicNum() == 6 and carbon.GetHybridization() == SP3 and hydrogens:
                    candidates.append((mid(pos[carbon.GetIdx()], pos[hydrogens[0].GetIdx()]), pos[atom.GetIdx()], GREEN,
                                       "possible σ(C-H)→π*/p", "sigma-pi"))
        pattern = Chem.MolFromSmarts("[N,O,S;!+]-[#6,#7,#8]=[#6,#7,#8]")
        for hetero, c1, c2 in md.GetSubstructMatches(pattern):
            if (lone_pairs(md.GetAtomWithIdx(hetero)) or 0) > 0:
                candidates.append((pos[hetero], mid(pos[c1], pos[c2]), ORANGE, "possible n/π conjugation", "n-pi"))
    except Exception:
        pass
    return candidates[:6]


def projected_dipole_direction(md, pos):
    try:
        work = Chem.Mol(md)
        rdPartialCharges.ComputeGasteigerCharges(work)
        sx = sy = 0.0
        for atom in work.GetAtoms():
            q = float(atom.GetProp("_GasteigerCharge"))
            if not math.isfinite(q):
                return None
            sx += q * pos[atom.GetIdx()][0]
            sy += q * pos[atom.GetIdx()][1]
        return None if math.hypot(sx, sy) < 1e-8 else unit((-sx, -sy))
    except Exception:
        return None


def annotate_structure(mol, mu_3d, box_size):
    md, image, pos = draw_structure(mol, box_size)
    fs, fxs = font(15, True), font(13)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for bond in md.GetBonds():
        if bond.GetBondType() not in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
            continue
        for atom, other in ((bond.GetBeginAtom(), bond.GetEndAtom()), (bond.GetEndAtom(), bond.GetBeginAtom())):
            p, q = pos[atom.GetIdx()], pos[other.GetIdx()]
            direction = unit((q[0]-p[0], q[1]-p[1]))
            perp = (-direction[1], direction[0])
            for sign, color in ((1, (60,110,230,95)), (-1, (230,80,80,95))):
                center = (p[0]+sign*perp[0]*26, p[1]+sign*perp[1]*26)
                od.polygon(ellipse_pts(center, (sign*perp[0], sign*perp[1]), 22, 11), fill=color, outline=color[:3]+(200,))
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    for atom in md.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        p, outward = pos[atom.GetIdx()], away_dir(pos, atom)
        lp = lone_pairs(atom)
        if lp:
            angle0 = math.atan2(outward[1], outward[0])
            offsets = {1:[0], 2:[-50,50], 3:[-75,0,75], 4:[-90,-30,30,90]}.get(min(lp,4), [0])
            for offset in offsets:
                angle = angle0 + math.radians(offset)
                cx, cy = p[0]+26*math.cos(angle), p[1]+26*math.sin(angle)
                for spacing in (-4.5, 4.5):
                    x, y = cx-spacing*math.sin(angle), cy+spacing*math.cos(angle)
                    draw.ellipse((x-3,y-3,x+3,y+3), fill=RED)
        tag = hybridization_label(atom) if atom.GetAtomicNum() in (6,7,8) else ""
        if atom.GetFormalCharge():
            tag = (tag + " " if tag else "") + f"{atom.GetFormalCharge():+d}"
        if tag:
            radius = 48 if lp else 30
            draw.text((p[0]+outward[0]*radius, p[1]+outward[1]*radius), tag, font=fs, fill=BLUE, anchor="mm")
    candidates = interaction_candidates(md, pos)
    for i, (p0, p2, color, label, _) in enumerate(candidates):
        curved_arrow(draw, p0, p2, color, label, fxs, 0.35 if i % 2 == 0 else -0.35)
    width, height = image.size
    bx, by = width-170, height-110
    draw.rectangle((bx,by,width-12,height-12), outline=GRAY)
    draw.text((bx+8,by+6), "dipole μ", font=fs, fill=BLACK)
    if formal_charge(mol) != 0:
        draw.text((bx+15,by+45), "N/A", font=font(22,True), fill=BLACK)
        draw.text((bx+8,height-32), "(charged species)", font=fxs, fill=GRAY)
    elif mu_3d is None:
        draw.text((bx+15,by+45), "N/A", font=font(22,True), fill=BLACK)
        draw.text((bx+8,height-32), "estimate unavailable", font=fxs, fill=GRAY)
    elif mu_3d > 0.15:
        direction = projected_dipole_direction(md, pos)
        if direction:
            center = (bx+80,by+65)
            tip = (center[0]+direction[0]*45, center[1]+direction[1]*45)
            tail = (center[0]-direction[0]*45, center[1]-direction[1]*45)
            draw.line((tail,tip), fill=BLACK, width=3)
            arrow_head(draw, tip, direction, BLACK)
            draw.line((tail[0]-direction[1]*7,tail[1]+direction[0]*7,tail[0]+direction[1]*7,tail[1]-direction[0]*7), fill=BLACK, width=3)
        draw.text((bx+8,height-32), f"≈ {mu_3d:.1f} D*", font=fxs, fill=GRAY)
    else:
        draw.text((bx+30,by+45), "μ ≈ 0", font=font(22,True), fill=BLACK)
        draw.text((bx+8,height-32), "point-charge est.", font=fxs, fill=GRAY)
    lx, ly = 14, height-100
    draw.polygon(ellipse_pts((lx+14,ly+8),(0,-1),10,5), fill=(60,110,230,120))
    draw.text((lx+32,ly), "schematic p orbital", font=fxs, fill=BLACK)
    draw.ellipse((lx+8,ly+22,lx+14,ly+28), fill=RED); draw.ellipse((lx+17,ly+22,lx+23,ly+28), fill=RED)
    draw.text((lx+32,ly+18), "estimated lone pair", font=fxs, fill=BLACK)
    for i,(color,text) in enumerate(((GREEN,"possible σ→π*/p"),(ORANGE,"possible n/π conjugation"))):
        draw.line((lx+4,ly+44+i*18,lx+26,ly+44+i*18), fill=color, width=3)
        draw.text((lx+32,ly+36+i*18), text, font=fxs, fill=BLACK)
    return image.convert("RGB"), candidates


def energy_items(mol):
    current_name = ", ".join(stereo_label(mol)) or "input structure"
    _, energy, method, _ = minimum_energy(mol, 42)
    other = opposite_alkene_isomer(mol)
    if other is not None and energy is not None:
        _, other_energy, other_method, _ = minimum_energy(other, 314)
        if other_energy is not None and other_method == method:
            return [(current_name,energy,True),(", ".join(stereo_label(other)) or "other stereoisomer",other_energy,False)], f"sampled stereoisomers ({method})", "Lowest sampled force-field conformer; not experimental ΔG."
    if energy is not None:
        return [(current_name,energy,True)], f"one sampled minimum ({method})", "Absolute force-field energy is not an experimental stability value."
    return [("unavailable",0.0,False)], "calculation unavailable", "No supported force-field result was obtained."


def draw_energy_panel(image, box, items, kind, note):
    draw = ImageDraw.Draw(image)
    x0,y0,x1,y1 = box
    fh,fs = font(17,True),font(13)
    draw.rectangle(box, outline=GRAY)
    draw.text((x0+12,y0+8), f"Energy  ({kind})", font=fh, fill=BLACK)
    energies = [e for _,e,_ in items]
    minimum = min(energies)
    span = max(max(e-minimum for e in energies),1.0)
    top,bottom = y0+65,y1-85
    item_width = (x1-x0-70)/max(len(items),1)
    draw.line((x0+30,bottom+15,x0+30,top-10), fill=BLACK, width=2)
    arrow_head(draw,(x0+30,top-10),(0,-1),BLACK)
    draw.text((x0+18,top-28),"E",font=fs,fill=BLACK)
    for i,(label,e,current) in enumerate(items):
        rel=e-minimum; y=bottom-rel/span*(bottom-top); xa=x0+50+i*item_width; xb=xa+item_width-20
        color=BLUE if current else GRAY
        draw.line((xa,y,xb,y),fill=color,width=6)
        draw.text(((xa+xb)/2,y-6),f"{rel:+.2f} kcal/mol",font=fs,fill=color,anchor="mb")
        shown=label if len(label)<=23 else label[:20]+"..."
        draw.text(((xa+xb)/2,y+8),shown+("  ◀ input" if current else ""),font=fs,fill=color,anchor="mt")
    draw.text((x0+12,y1-42),note if len(note)<=62 else note[:59]+"...",font=fs,fill=RED)


def draw_mo_panel(image, box, mol, candidates):
    draw=ImageDraw.Draw(image); x0,y0,x1,y1=box; fh,fs=font(17,True),font(13)
    draw.rectangle(box,outline=GRAY); draw.text((x0+12,y0+8),"Orbital picture (qualitative)",font=fh,fill=BLACK)
    has_pi=any(b.GetBondType() in (Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC) for b in mol.GetBonds())
    has_lp=any((lone_pairs(a) or 0)>0 for a in mol.GetAtoms())
    kinds={c[4] for c in candidates}; levels={}; height=y1-y0
    def level(name,fraction,x,color,occupied):
        y=y0+fraction*height; draw.line((x,y,x+90,y),fill=color,width=4); levels[name]=(x+45,y)
        draw.text((x+95,y),name,font=fs,fill=color,anchor="lm")
        if occupied:
            draw.text((x+30,y-3),"↑",font=font(20,True),fill=color,anchor="mb"); draw.text((x+52,y-3),"↓",font=font(20,True),fill=color,anchor="mb")
    left,middle=x0+30,x0+220
    if has_pi:
        level("π* acceptor",0.25,middle,BLUE,False); level("π donor",0.68,middle,BLUE,True)
    else:
        level("σ* acceptor",0.25,middle,BLUE,False); level("σ bond",0.78,middle,BLUE,True)
    if has_lp: level("n lone pair",0.50,left,ORANGE,True)
    level("σ(C-H)",0.86,left,GREEN,True)
    if "sigma-pi" in kinds and "π* acceptor" in levels: curved_arrow(draw,levels["σ(C-H)"],levels["π* acceptor"],GREEN,dashed=True,width=2)
    if "n-pi" in kinds and "n lone pair" in levels and "π* acceptor" in levels: curved_arrow(draw,levels["n lone pair"],levels["π* acceptor"],ORANGE,dashed=True,width=2)
    draw.text((x0+12,y1-58),"dashed = possible donor→acceptor overlap",font=fs,fill=GRAY)
    draw.text((x0+12,y1-40),"connectivity only; 3D alignment not tested",font=fs,fill=GRAY)
    draw.text((x0+12,y1-22),"schematic levels, not computed HOMO/LUMO",font=fs,fill=GRAY)


def make_card(name):
    smiles=name_to_smiles(name); mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Could not parse molecule: {name}")
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    try: rdCIPLabeler.AssignCIPLabels(mol)
    except Exception: pass
    width,height=1500,960; image=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(image)
    mu=get_3d_dipole(mol); structure,candidates=annotate_structure(mol,mu,(920,840))
    image.paste(structure,(25,95)); draw.rectangle((25,95,945,935),outline=GRAY)
    items,kind,note=energy_items(mol); draw_energy_panel(image,(965,95,1475,470),items,kind,note)
    draw_mo_panel(image,(965,490,1475,935),mol,candidates)
    mol_h=Chem.AddHs(mol); sigma=mol_h.GetNumBonds(); kek=Chem.Mol(mol_h)
    try: Chem.Kekulize(kek,clearAromaticFlags=True)
    except Exception: pass
    pi=sum({Chem.BondType.DOUBLE:1,Chem.BondType.TRIPLE:2}.get(b.GetBondType(),0) for b in kek.GetBonds())
    draw.text((25,14),name,font=font(30,True),fill=BLACK)
    draw.text((730,26),"Qualitative auto-generated study aid: verify resonance, conformations, and mechanisms.",font=font(14,True),fill=RED)
    shown_smiles=smiles if len(smiles)<=40 else smiles[:37]+"..."; stereo="  ".join(stereo_label(mol)) or "stereo: none assigned"
    subtitle=f"{rdMolDescriptors.CalcMolFormula(mol)}   {Descriptors.MolWt(mol):.1f} g/mol   σ {sigma}  π {pi:g}   {stereo}   {shown_smiles}"
    draw.text((25,58),subtitle,font=font(16),fill=GRAY)
    return image


@st.cache_data(show_spinner=False)
def get_card_image_bytes(name):
    image=make_card(name); buffer=io.BytesIO(); image.save(buffer,format="PNG"); return buffer.getvalue()


def safe_filename(name):
    cleaned=re.sub(r"[^A-Za-z0-9_.-]+","_",name.strip()).strip("._")
    return cleaned or "orgo_card"


def main():
    st.set_page_config(page_title="OrgoCard",layout="wide")
    st.title("OrgoCard Generator")
    st.caption("Structure annotations are automated study aids. Energy values are sampled force-field energies, not experimental thermodynamic data.")
    col1,col2=st.columns([3,1])
    with col1:
        name=st.text_input("Molecule name or SMILES:",value="cis-1,2-difluoroethene")
    with col2:
        st.write(""); st.write(""); requested=st.button("Draw",type="primary",use_container_width=True)
    if requested:
        with st.spinner("Generating conformers and study card..."):
            try:
                png=get_card_image_bytes(name); st.image(png,use_container_width=True)
                st.download_button("Download PNG",png,f"{safe_filename(name)}.png","image/png")
                st.info("Dipole values are Gasteiger point-charge estimates. Energy comparisons use sampled MMFF94s or UFF conformers. Orbital interactions are possibilities inferred from connectivity, not proof of favorable 3D overlap.")
            except Exception as error:
                st.error(f"Error generating molecule: {error}")


if __name__ == "__main__":
    main()
