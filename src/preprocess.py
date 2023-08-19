import json
import os
import subprocess

def parse_dssp_line(line):
    idx = int(line[0:5])
    res = line[13]
    ss = line[16]
    acc = float(line[34:38])
    nho0p = float(line[39:45])
    nho0e = float(line[46:50])
    ohn0p = float(line[50:56])
    ohn0e = float(line[57:61])
    nho1p = float(line[61:67])
    nho1e = float(line[68:72])
    ohn1p = float(line[72:78])
    ohn1e = float(line[79:83])
    phi = float(line[103:109])
    psi = float(line[109:115])
    x = float(line[115:122])
    y = float(line[122:129])
    z = float(line[129:136])
    
    return {
        'idx': idx,
        'res': res,
        'ss': ss,
        'acc': acc,
        'nho0p': nho0p,
        'nho0e': nho0e,
        'ohn0p': ohn0p,
        'ohn0e': ohn0e,
        'nho1p': nho1p,
        'nho1e': nho1e,
        'ohn1p': ohn1p,
        'ohn1e': ohn1e,
        'phi': phi,
        'psi': psi,
        'x': x,
        'y': y,
        'z': z
    }


def process_file(inpath, outdir):
    filename = inpath
    pdbid = os.path.splitext(os.path.basename(filename))[0]
    dssp_filename = outdir + pdbid + '.dssp'
    
    subprocess.run(['../DSSP/dssp', '-i', filename, '-o', dssp_filename], check=True)

    data = {'model': {}}

    with open(dssp_filename, 'r') as input_file:
        lines = input_file.readlines()[28:]
        for line in lines:
            parsed_line = parse_dssp_line(line)
            data['model'][parsed_line['idx']] = parsed_line

    json_filename = outdir + pdbid + '.json'
    with open(json_filename, 'w') as output_file:
        output_file.write(json.dumps(data))

if __name__ == '__main__':
    test_ent_list = ['../dataset/pdbstyle-2.08/r1/d3r1pa_.ent', '../dataset/pdbstyle-2.08/jl/d6jl3a_.ent', '../dataset/pdbstyle-2.08/nd/d7ndyg_.ent']
    for fn in test_ent_list:
        sid = os.path.splitext(os.path.basename(fn))[0]
        print(sid)
        outdir = '../dataset/json/'
        process_file(fn, outdir)
