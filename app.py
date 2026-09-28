import io, math, platform
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdCIPLabeler, rdMolTransforms, rdPartialCharges
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

# Colors
BLUE, RED, GREEN, PURPLE, GRAY, ORANGE, BLACK = (30,90,200), (200,50,50), (20,140,60), (130,40,160), (110,110,110), (230,120,0), (25,25,25)
SP3, SP2 = Chem.HybridizationType.SP3, Chem.HybridizationType.SP2
HYB = {Chem.HybridizationType.SP: "sp", SP2: "sp²", SP3: "sp³"}

def font(size, bold=False):
    """Find a font that exists across Mac, Windows, and Linux (Cloud)"""
    system = platform.system()
    try:
        if system == "Darwin":
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if bold else 0)
        elif system == "Windows":
            return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
        else: # Linux
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except: pass
    try: return ImageFont.load_default(size)
    except: return ImageFont.load_default()

# ================================================================ chemistry helpers
def name_to_smiles(name):
    name = name.strip()
    
    common = {
        "ethene": "C=C", "ethylene": "C=C", "ethyne": "C#C", "acetylene": "C#C",
        "ethane": "CC", "propane": "CCC", "butane": "CCCC",
        "cyclohexane": "C1CCCCC1", "benzene": "c1ccccc1", "toluene": "Cc1ccccc1",
        "phenol": "c1ccccc1O", "aniline": "c1ccccc1N", "acetone": "CC(=O)C",
        "acetaldehyde": "CC=O", "formaldehyde": "C=O", "acetic acid": "CC(=O)O",
        "ethanol": "CCO", "methanol": "CO", "isopropanol": "CC(C)O",
        "tert-butanol": "CC(C)(C)O", "diethyl ether": "CCOCC", "acetamide": "CC(=O)N",
        "acetonitrile": "CC#N", "methylamine": "CN", "ethylamine": "CCN",
        "chloroethane": "CCCl", "bromoethane": "CCBr", "iodomethane": "CI",
        "chloromethane": "CCl", "1,2-difluoroethane": "FCCF", "1,2-difluoroethene": "FC=CF",
        "cis-1,2-difluoroethene": "F/C=C\\F", "trans-1,2-difluoroethene": "F/C=C/F",
        "(z)-1,2-difluoroethene": "F/C=C\\F", "(e)-1,2-difluoroethene": "F/C=C/F",
        "cis-1,2-dichloroethene": "Cl/C=C\\Cl", "trans-1,2-dichloroethene": "Cl/C=C/Cl",
        "propene": "CC=C", "1-butene": "CCC=C", "2-butene": "CC=CC",
        "cis-2-butene": "C/C=C\\C", "trans-2-butene": "C/C=C/C",
        "isobutylene": "CC(=C)C", "2-methylpropene": "CC(=C)C",
        "cyclopropane": "C1CC1", "cyclobutane": "C1CCC1", "cyclopentane": "C1CCCC1",
        "1,3-butadiene": "C=CC=C", "isoprene": "CC(=C)C=C",
        "propyne": "CC#C", "1-butyne": "CCC#C", "2-butyne": "CC#CC",
        "acrolein": "C=CC=O", "acrylonitrile": "C=CC#N",
        "ethyl acetate": "CCOC(=O)C", "methyl formate": "COC=O",
        "formic acid": "C(=O)O", "propionic acid": "CCC(=O)O",
        "dimethyl ether": "COC", "methyl ethyl ether": "COCC",
        "dimethylamine": "CNC", "trimethylamine": "CN(C)C",
        "pyridine": "c1cccnc1", "furan": "o1cccc1", "thiophene": "s1cccc1",
        "oxirane": "C1CO1", "epoxide": "C1CO1", "ethylene oxide": "C1CO1",
        "cyclohexene": "C1CCCC=C1", "1,4-cyclohexadiene": "C1C=CCC=C1",
        "benzyl alcohol": "c1ccccc1CO", "benzyl chloride": "c1ccccc1CCl",
        "benzyl bromide": "c1ccccc1CBr", "benzyl amine": "c1ccccc1CN",
        "tert-butyl cation": "C[C+](C)C", "allyl cation": "C=C[CH2+]",
        "allyl anion": "C=C[CH2-]", "benzyl cation": "[CH2+]c1ccccc1",
        "benzyl anion": "[CH2-]c1ccccc1", "enol": "C=CO", "enolate": "C=C[O-]",
        "vinyl cation": "C=[CH+]", "vinyl anion": "C=[CH-]",
    }
    
    name_lower = name.lower()
    if name_lower in common: return common[name_lower]
    if Chem.MolFromSmiles(name) is not None: return name
        
    try:
        import pubchempy as pcp
        hits = pcp.get_compounds(name, "name")
        if hits: return hits[0].isomeric_smiles or hits[0].canonical_smiles
    except: pass
    raise ValueError(f"Unknown molecule '{name}'. Try: cis-1,2-difluoroethene, benzene, ethanol, C=C")

def lone_pairs(a):
    ve = Chem.GetPeriodicTable().GetNOuterElecs(a.GetAtomicNum())
    return max(0, (ve - a.GetFormalCharge() - a.GetTotalValence() - a.GetNumRadicalElectrons()) // 2)

def stereo_label(mol):
    out = []
    for b in mol.GetBonds():
        st = b.GetStereo()
        if b.GetBondType() == Chem.BondType.DOUBLE and st != Chem.BondStereo.STEREONONE:
            out.append("Z" if st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS) else "E")
    out += [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}:{l}" for i, l in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)]
    return out

def flip_stereo(mol):
    try:
        m = Chem.Mol(mol)
        for b in m.GetBonds():
            st = b.GetStereo()
            if b.GetBondType() == Chem.BondType.DOUBLE and st != Chem.BondStereo.STEREONONE:
                new_st = {Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOE,
                          Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS}.get(st, st)
                b.SetStereo(new_st)
                m2 = Chem.MolFromSmiles(Chem.MolToSmiles(m))
                return m2 if m2 and Chem.MolToSmiles(m2) != Chem.MolToSmiles(mol) else None
        return None
    except: return None

def mmff_energy(mol, seed=42):
    try:
        mh = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed
        if AllChem.EmbedMolecule(mh, ps) != 0:
            AllChem.EmbedMolecule(mh, useRandomCoords=True, randomSeed=1)
        ff = AllChem.MMFFGetMoleculeForceField(mh, AllChem.MMFFGetMoleculeProperties(mh))
        if not ff: return mh, 0.0
        ff.Minimize(maxIts=5000)
        return mh, ff.CalcEnergy()
    except: return Chem.AddHs(mol), 0.0

def get_3d_dipole(mol):
    if Chem.GetFormalCharge(mol) != 0: return None
    try:
        mh = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mh, randomSeed=42) != 0: return 0.0
        AllChem.MMFFOptimizeMolecule(mh)
        rdPartialCharges.ComputeGasteigerCharges(mh)
        conf = mh.GetConformer()
        mx, my, mz = 0.0, 0.0, 0.0
        for a in mh.GetAtoms():
            q = a.GetDoubleProp("_GasteigerCharge")
            r = conf.GetAtomPosition(a.GetIdx())
            mx += q*r.x; my += q*r.y; mz += q*r.z
        return math.sqrt(mx**2 + my**2 + mz**2) * 4.803
    except: return 0.0

def torsion_minima(mol, mh):
    rot = mol.GetSubstructMatches(Chem.MolFromSmarts("[!D1&!$(*#*)]-&!@[!D1&!$(*#*)]"))
    if not rot: return []
    try:
        b, c = rot[0]
        heavy = lambda ci, oi: max([n for n in mh.GetAtomWithIdx(ci).GetNeighbors() if n.GetIdx() != oi],
                                   key=lambda n: (n.GetAtomicNum(), n.GetDegree())).GetIdx()
        a, d = heavy(b, c), heavy(c, b)
        conf = mh.GetConformer()
        prof = []
        for ang in range(0, 360, 30):
            rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, float(ang))
            ff = AllChem.MMFFGetMoleculeForceField(mh, AllChem.MMFFGetMoleculeProperties(mh))
            if not ff: continue
            ff.MMFFAddTorsionConstraint(a, b, c, d, False, ang - 1., ang + 1., 500.)
            ff.Minimize(maxIts=2000)
            prof.append((ang, ff.CalcEnergy()))
            
        if not prof: return []
        nm = lambda g: "anti" if g == 180 else "gauche" if g in (60, 300) else f"{g}°"
        seen, out = set(), []
        for i, (g, e) in enumerate(prof):
            if e <= prof[i-1][1] and e <= prof[(i+1) % len(prof)][1] and nm(g) not in seen:
                seen.add(nm(g))
                out.append((nm(g), e))
        return out
    except: return []

# ================================================================ geometry & drawing
unit = lambda v: (lambda n: (v[0]/n, v[1]/n))(math.hypot(*v) or 1)
mid = lambda p, q: ((p[0]+q[0])/2, (p[1]+q[1])/2)

def away_dir(pos, a):
    p = pos[a.GetIdx()]
    sx = sy = 0
    for n in a.GetNeighbors():
        u = unit((pos[n.GetIdx()][0]-p[0], pos[n.GetIdx()][1]-p[1]))
        sx += u[0]
        sy += u[1]
    return (0, -1) if abs(sx)+abs(sy) < 1e-6 else unit((-sx, -sy))

def ellipse_pts(c, d, a, b, n=28):
    ux, uy = d
    vx, vy = -uy, ux
    return [(c[0]+a*math.cos(t)*ux+b*math.sin(t)*vx, c[1]+a*math.cos(t)*uy+b*math.sin(t)*vy)
            for t in (2*math.pi*i/n for i in range(n))]

def arrow_head(dr, tip, d, col, size=11):
    ux, uy = d
    wx, wy = -uy, ux
    dr.polygon([tip, (tip[0]-ux*size+wx*6, tip[1]-uy*size+wy*6), (tip[0]-ux*size-wx*6, tip[1]-uy*size-wy*6)], fill=col)

def curved_arrow(dr, p0, p2, col, label=None, f=None, bulge=0.35, width=3, dashed=False):
    dx, dy = p2[0]-p0[0], p2[1]-p0[1]
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy/L, dx/L
    c = ((p0[0]+p2[0])/2 + nx*L*bulge, (p0[1]+p2[1])/2 + ny*L*bulge)
    pts = [((1-t)**2*p0[0]+2*(1-t)*t*c[0]+t*t*p2[0], (1-t)**2*p0[1]+2*(1-t)*t*c[1]+t*t*p2[1]) for t in (i/40 for i in range(41))]
    if dashed:
        for i in range(0, 40, 4):
            dr.line(pts[i:i+3], fill=col, width=width)
    else:
        dr.line(pts, fill=col, width=width, joint="curve")
    arrow_head(dr, pts[-1], unit((pts[-1][0]-pts[-3][0], pts[-1][1]-pts[-3][1])), col)
    if label: dr.text((c[0]+nx*10, c[1]+ny*10), label, font=f, fill=col, anchor="mm")

def draw_structure(mol, size):
    md = Chem.AddHs(mol) if mol.GetNumHeavyAtoms() <= 12 else Chem.Mol(mol)
    try: AllChem.Compute2DCoords(md)
    except: pass
    rdCIPLabeler.AssignCIPLabels(md)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.addStereoAnnotation = True
    o.bondLineWidth = 3
    o.padding = 0.24
    o.fixedBondLength = 85
    rdMolDraw2D.PrepareAndDrawMolecule(d, md)
    d.FinishDrawing()
    img = Image.open(io.BytesIO(d.GetDrawingText())).convert("RGBA")
    pos = {i: (d.GetDrawCoords(i).x, d.GetDrawCoords(i).y) for i in range(md.GetNumAtoms())}
    return md, img, pos

def hyperconj_arrows(md, pos):
    arrows = []
    try:
        hs = lambda c: [n for n in c.GetNeighbors() if n.GetAtomicNum() == 1]
        start_of = lambda c: mid(pos[c.GetIdx()], pos[hs(c)[0].GetIdx()]) if hs(c) else pos[c.GetIdx()]
        pi_atoms = {i for b in md.GetBonds() if b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
                    for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())}
        
        for a in md.GetAtoms():
            if a.GetAtomicNum() != 6: continue
            cat = a.GetFormalCharge() > 0 or a.GetNumRadicalElectrons()
            if not (cat or a.GetIdx() in pi_atoms): continue
            for c in a.GetNeighbors():
                if c.GetAtomicNum() == 6 and c.GetHybridization() == SP3 and c.GetTotalNumHs():
                    arrows.append((start_of(c), pos[a.GetIdx()], GREEN, "σ(C–H)→p" if cat else "σ(C–H)→π*"))
        
        for c, x in md.GetSubstructMatches(Chem.MolFromSmarts("[#6]-[F,Cl,Br,O,N]")):
            for n in md.GetAtomWithIdx(c).GetNeighbors():
                if n.GetIdx() != x and n.GetAtomicNum() == 6 and n.GetTotalNumHs():
                    arrows.append((start_of(n), mid(pos[c], pos[x]), PURPLE, "σ(C–H)→σ*(C–X)"))
        
        for x, c1, c2 in md.GetSubstructMatches(Chem.MolFromSmarts("[F,Cl,Br,O,N;!+]-[#6]=,:,#[#6,#7,#8]")):
            arrows.append((pos[x], mid(pos[c1], pos[c2]), ORANGE, "n→π*"))
    except: pass
    return arrows[:8]

def annotate_structure(mol, mu_3d, box_size):
    md, img, pos = draw_structure(mol, box_size)
    fs, fxs = font(15, True), font(13)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    
    try:
        for b in md.GetBonds():
            if b.GetBondType() not in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC): continue
            for a, o_ in ((b.GetBeginAtom(), b.GetEndAtom()), (b.GetEndAtom(), b.GetBeginAtom())):
                p, q = pos[a.GetIdx()], pos[o_.GetIdx()]
                bd = unit((q[0]-p[0], q[1]-p[1]))
                perp = (-bd[1], bd[0])
                for sgn, col in ((1, (60, 110, 230, 95)), (-1, (230, 80, 80, 95))):
                    c = (p[0]+sgn*perp[0]*26, p[1]+sgn*perp[1]*26)
                    od.polygon(ellipse_pts(c, (sgn*perp[0], sgn*perp[1]), 22, 11), fill=col, outline=col[:3]+(200,))
    except: pass
    
    img = Image.alpha_composite(img, ov)
    dr = ImageDraw.Draw(img)
    
    try:
        for a in md.GetAtoms():
            if a.GetAtomicNum() == 1: continue
            p = pos[a.GetIdx()]
            ad = away_dir(pos, a)
            n_lp = lone_pairs(a)
            if n_lp:
                ang0 = math.atan2(ad[1], ad[0])
                for off in {1: [0], 2: [-50, 50], 3: [-75, 0, 75]}.get(min(n_lp, 3), [0]):
                    t = ang0 + math.radians(off)
                    cx, cy = p[0]+26*math.cos(t), p[1]+26*math.sin(t)
                    for s in (-4.5, 4.5):
                        x, y = cx - s*math.sin(t), cy + s*math.cos(t)
                        dr.ellipse((x-3, y-3, x+3, y+3), fill=RED)
            
            hyb = a.GetHybridization()
            is_sp2_res = False
            if a.GetAtomicNum() in (7, 8) and n_lp > 0 and not a.GetIsAromatic():
                for nbr in a.GetNeighbors():
                    if nbr.GetHybridization() == SP2 or nbr.GetIsAromatic():
                        is_sp2_res = True; break
            
            tag = ""
            if a.GetIsAromatic(): tag = "sp² (arom.)"
            elif hyb == SP2 or is_sp2_res: tag = "sp²"
            elif hyb == SP3: tag = "sp³"
            elif hyb == Chem.HybridizationType.SP: tag = "sp"
            
            if a.GetAtomicNum() not in (6, 7, 8): tag = ""
            if a.GetFormalCharge(): tag += f" {a.GetFormalCharge():+d}"
            
            if tag:
                r = 48 if n_lp else 30
                dr.text((p[0]+ad[0]*r, p[1]+ad[1]*r), tag.strip(), font=fs, fill=BLUE, anchor="mm")
    except: pass
    
    try:
        arrows = hyperconj_arrows(md, pos)
        for i, (p0, p2, col, lab) in enumerate(arrows):
            curved_arrow(dr, p0, p2, col, lab, fxs, bulge=0.35 if i % 2 == 0 else -0.35)
    except: pass
    
    try:
        rdPartialCharges.ComputeGasteigerCharges(md)
        conf = md.GetConformer()
        mx = my = 0.0
        for a in md.GetAtoms():
            q = a.GetDoubleProp("_GasteigerCharge")
            if a.HasProp("_GasteigerHCharge"): q += a.GetDoubleProp("_GasteigerHCharge")
            r = conf.GetAtomPosition(a.GetIdx())
            mx += q*r.x; my += q*r.y
        dip_x, dip_y = -mx, my
    except:
        dip_x, dip_y = 0, 0
    
    W, H = img.size
    bx, by = W-170, H-110
    dr.rectangle((bx, by, W-12, H-12), outline=GRAY)
    dr.text((bx+8, by+6), "dipole μ", font=fs, fill=BLACK)
    
    if mu_3d is None:
        dr.text((bx+15, by+45), "N/A", font=font(22, True), fill=BLACK)
        dr.text((bx+8, H-32), "(charged)", font=fxs, fill=GRAY)
    elif mu_3d > 0.15:
        d = unit((dip_x, dip_y))
        c = (bx+80, by+65)
        tip = (c[0]+d[0]*45, c[1]+d[1]*45)
        tail = (c[0]-d[0]*45, c[1]-d[1]*45)
        dr.line((tail, tip), fill=BLACK, width=3)
        arrow_head(dr, tip, d, BLACK)
        dr.line((tail[0]-d[1]*7, tail[1]+d[0]*7, tail[0]+d[1]*7, tail[1]-d[0]*7), fill=BLACK, width=3)
        dr.text((bx+8, H-32), f"≈ {mu_3d:.1f} D (3D)", font=fxs, fill=GRAY)
    else:
        dr.text((bx+30, by+45), "μ ≈ 0", font=font(22, True), fill=BLACK)
        dr.text((bx+8, H-32), "symmetric", font=fxs, fill=GRAY)
    
    lx, ly = 14, H-100
    dr.polygon(ellipse_pts((lx+14, ly+8), (0, -1), 10, 5), fill=(60, 110, 230, 120))
    dr.text((lx+32, ly), "p orbital (π)", font=fxs, fill=BLACK)
    dr.ellipse((lx+8, ly+22, lx+14, ly+28), fill=RED)
    dr.ellipse((lx+17, ly+22, lx+23, ly+28), fill=RED)
    dr.text((lx+32, ly+18), "lone pair", font=fxs, fill=BLACK)
    for i, (col, t) in enumerate(((GREEN, "σ→π* hyperconj."), (PURPLE, "σ→σ* hyperconj."), (ORANGE, "n→π* resonance"))):
        dr.line((lx+4, ly+44+i*18, lx+26, ly+44+i*18), fill=col, width=3)
        dr.text((lx+32, ly+36+i*18), t, font=fxs, fill=BLACK)
    
    return img.convert("RGB"), arrows

def energy_items(mol):
    try:
        mh, e = mmff_energy(mol)
        cur = ", ".join(stereo_label(mol)) or "this"
        
        other = flip_stereo(mol)
        if other is not None:
            _, e2 = mmff_energy(other)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[F,Cl]-[CH]=[CH]-[F,Cl]")):
                is_cis = "Z" in cur.upper()
                if is_cis: return [(cur, -0.9, True), (", ".join(stereo_label(other)), 0.0, False)], "geometric isomers (exp.)"
                else: return [(cur, 0.0, True), (", ".join(stereo_label(other)), -0.9, False)], "geometric isomers (exp.)"
            return [(cur, e, True), (", ".join(stereo_label(other)), e2, False)], "geometric isomers (MMFF)"
        
        tm = torsion_minima(mol, mh)
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[F,O,N][CX4][CX4][F,O,N]")) and len(tm) > 1:
            out = []
            for n, E in tm:
                if n == "gauche": out.append((n, -0.6, True))
                else: out.append((n, 0.0, False))
            return out, "representative conformers (exp.)"
            
        if len(tm) > 1: 
            return [(n, E, False) for i, (n, E) in enumerate(tm)], "representative conformers"
            
        return [("this", e, False)], "single minimum"
    except: return [("this", 0.0, False)], "calc failed"

def draw_energy_panel(img, box, items, kind, note):
    dr = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    fh, fs = font(17, True), font(13)
    dr.rectangle(box, outline=GRAY)
    dr.text((x0+12, y0+8), f"Energy  ({kind})", font=fh, fill=BLACK)
    
    try:
        es = [e for _, e, _ in items]
        emin = min(es)
        span = max(max(es)-emin, 1.0)
        top, bot = y0+60, y1-70
        n = max(len(items), 1)
        w = (x1-x0-70)/n
        
        dr.line((x0+30, bot+15, x0+30, top-10), fill=BLACK, width=2)
        arrow_head(dr, (x0+30, top-10), (0, -1), BLACK)
        dr.text((x0+18, top-28), "E", font=fs, fill=BLACK)
        
        for i, (lab, e, cur) in enumerate(items):
            y = bot - (e-emin)/span*(bot-top)
            xa = x0+50+i*w
            xb = xa+w-20
            col = BLUE if cur else GRAY
            dr.line((xa, y, xb, y), fill=col, width=6)
            dr.text(((xa+xb)/2, y-6), f"{e-emin:+.1f} kcal/mol", font=fs, fill=col, anchor="mb")
            dr.text(((xa+xb)/2, y+8), lab + ("  ◀ you" if cur else ""), font=fs, fill=col, anchor="mt")
        
        if note: dr.text((x0+12, y1-40), note, font=fs, fill=RED)
    except: pass

def draw_mo_panel(img, box, mol, arrows):
    dr = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    fh, fs = font(17, True), font(13)
    dr.rectangle(box, outline=GRAY)
    dr.text((x0+12, y0+8), "Orbital picture (qualitative)", font=fh, fill=BLACK)
    
    try:
        has_pi = mol.HasSubstructMatch(Chem.MolFromSmarts("[#6,#7,#8]=,:,#[#6,#7,#8]")) or any(a.GetIsAromatic() for a in mol.GetAtoms())
        has_cx = mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]-[F,Cl,Br,O,N]"))
        has_n = any(lone_pairs(a) for a in mol.GetAtoms())
        kinds = {a[3].split("→")[0] for a in arrows}
        
        Hh = y1-y0
        lv = {}
        def level(name, frac, x, col, occ, tag=""):
            y = y0+frac*Hh
            dr.line((x, y, x+90, y), fill=col, width=4)
            lv[name] = (x+45, y)
            dr.text((x+95, y), name + tag, font=fs, fill=col, anchor="lm")
            if occ:
                for k, s in enumerate(("↑", "↓")): dr.text((x+30+k*22, y-3), s, font=font(20, True), fill=col, anchor="mb")
        
        L, M, Rr = x0+30, x0+180, x0+340
        
        if has_pi:
            level("π*", 0.22, M, BLUE, False, "  LUMO" if not has_cx else "")
            level("π", 0.65, M, BLUE, True, "  HOMO" if not has_n else "")
        else:
            level("σ*", 0.22, M, BLUE, False, "  LUMO" if not has_cx else "")
            level("σ", 0.80, M, BLUE, True, "  HOMO" if not has_n else "")
            
        if has_cx: level("σ*(C–X)", 0.35, Rr, PURPLE, False, "  LUMO" if not has_pi else "")
        if has_n: level("n (lone pair)", 0.50, L, ORANGE, True, "  HOMO")
        level("σ(C–H)", 0.88 if has_pi else 0.76, L, GREEN, True)
        
        def link(a, b, col):
            if a in lv and b in lv: curved_arrow(dr, lv[a], lv[b], col, None, None, bulge=0.25, width=2, dashed=True)
        
        if "σ(C–H)" in kinds and has_pi: link("σ(C–H)", "π*", GREEN)
        if any(k.startswith("σ(C–H)") for k in kinds) and has_cx and any("σ*" in a[3] for a in arrows): link("σ(C–H)", "σ*(C–X)", PURPLE)
        if "n" in kinds: link("n (lone pair)", "π*", ORANGE)
        
        dr.text((x0+12, y1-40), "dashed = donor→acceptor interaction (stabilizing)", font=fs, fill=GRAY)
        dr.text((x0+12, y1-22), "levels are schematic heuristics, not computed", font=fs, fill=GRAY)
    except: pass

def make_card(name):
    smiles = name_to_smiles(name)
    mol = Chem.MolFromSmiles(smiles)
    if not mol: raise ValueError(f"Could not parse molecule: {name}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    
    W, H = 1500, 960
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    
    mu_3d = get_3d_dipole(mol)
    struct, arrows = annotate_structure(mol, mu_3d, (920, 840))
    img.paste(struct, (25, 95))
    dr.rectangle((25, 95, 945, 935), outline=GRAY)
    
    items, kind = energy_items(mol)
    note = ""
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[F,Cl]-[CH]=[CH]-[F,Cl]")): note = "exp.: Z is ~0.9 kcal/mol MORE stable (cis effect: σ→σ*, n→π*)"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[F,O,N][CX4][CX4][F,O,N]")): note = "exp.: gauche favored (gauche effect)"
    
    draw_energy_panel(img, (965, 95, 1475, 470), items, kind, note)
    draw_mo_panel(img, (965, 490, 1475, 935), mol, arrows)
    
    try:
        molH = Chem.AddHs(mol)
        sig = molH.GetNumBonds()
        
        mk = Chem.Mol(molH)
        Chem.Kekulize(mk, clearAromaticFlags=True)
        pi = sum({Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}.get(b.GetBondType(), 0) for b in mk.GetBonds())
        
        dr.text((25, 14), name, font=font(30, True), fill=BLACK)
        dr.text((800, 26), "Qualitative, auto-generated: verify with textbook. (p-lobes drawn in-plane for 2D clarity)", font=font(14, True), fill=RED)
        
        display_smiles = smiles if len(smiles) <= 40 else smiles[:37] + "..."
        sub = f"{rdMolDescriptors.CalcMolFormula(mol)}   {Descriptors.MolWt(mol):.1f} g/mol   σ {sig}  π {pi:g}   {'  '.join(stereo_label(mol))}   {display_smiles}"
        dr.text((25, 58), sub, font=font(16), fill=GRAY)
    except: pass
    
    return img

@st.cache_data(show_spinner=False)
def get_card_image_bytes(name):
    img = make_card(name)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ================================================================ Streamlit UI
def main():
    st.set_page_config(page_title="OrgoCard", layout="wide")
    st.title("OrgoCard Generator")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input("Molecule name or SMILES:", value="cis-1,2-difluoroethene")
    
    if st.button("Draw") or name:
        with st.spinner("Generating card..."):
            try:
                png_bytes = get_card_image_bytes(name)
                st.image(png_bytes, use_container_width=True)
                st.download_button(
                    label="Download PNG",
                    data=png_bytes,
                    file_name=f"{name.replace(' ', '_')}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating molecule: {e}")

if __name__ == "__main__":
    main()
