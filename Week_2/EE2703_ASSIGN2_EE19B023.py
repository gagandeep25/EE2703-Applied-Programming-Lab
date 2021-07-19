#Conventions used:
#V... n1 n2 value: Vn1 - Vn2 = value, current measured from n1 to n2
#I... n1 n2 value: current leaving n1 and entering n2
#All values are to be entered in scientific format. Arguments like 1k, 1m are not accepted

#Usage: python3 <file.py> <file.netlist>

from sys import argv
import numpy as np

circuit = '.circuit'; end = '.end'; ac = '.ac'
#function to parse the file and return a list of circuit specific lines, dictionary of nodes, voltage sources and freqency if ac
def parsefile(filename):
    try:
        with open(filename) as f:
            inckt = False; lines = []
            nodes = {'GND': 0}; n = 1
            k = 0; vol_srcs = {}
            freq = 0
            cleanlines = list(filter(None, [line.split('#')[0].strip() for line in f.readlines()]))
            if cleanlines.count(circuit) == 1 and cleanlines.count(end) == 1:
                if cleanlines.index(circuit) < cleanlines.index(end):
                    for l in cleanlines:
                        if l == circuit:
                            inckt = True
                        elif l == end:
                            inckt = False
                        elif l.split()[0] == ac:
                            freq = float(l.split()[-1])
                        elif inckt:
                            lines.append(l.split())
                            if l.split()[0].startswith('V'):
                                vol_srcs[l.split()[0]] = k
                                k += 1
                            try:
                                if l.split()[1] not in nodes.keys():
                                    nodes[l.split()[1]] = n
                                    n += 1
                                if l.split()[2] not in nodes.keys():
                                    nodes[l.split()[2]] = n
                                    n += 1
                            except Exception:
                                print("Error! Check the lines in the circuit block")
                                exit()
                else:
                    print("Invalid circuit definition")
            else:
                print("Invalid circuit definition")
        return vol_srcs, lines, nodes, freq
    except FileNotFoundError:
        print(f'File named {argv[1]} does not exist')
        exit()

if len(argv) != 2:
    print("Error! Please execute using the format:\npython3 <ee19b023_file1.py> <path to netlist file>")
elif '.netlist' not in argv[1]:
    print("Error! Please make sure the filename ends with .netlist")
else:
    vol_srcs, cktlines, nodes, freq = parsefile(argv[1])
    M = np.zeros((len(nodes) + len(vol_srcs), len(nodes) + len(vol_srcs)), dtype=complex)
    b = np.zeros(len(nodes) + len(vol_srcs), dtype=complex)
    for line in cktlines: #find the type of element and do the appropriate operation
        if line[0].startswith('R'):
            val = float(line[3])
            M[nodes[line[1]], nodes[line[1]]] += 1/val
            M[nodes[line[1]], nodes[line[2]]] -= 1/val
            M[nodes[line[2]], nodes[line[1]]] -= 1/val
            M[nodes[line[2]], nodes[line[2]]] += 1/val
        elif line[0].startswith('V'):
            if line[3] == 'dc':
                isac = False
                val = float(line[4])
            elif line[3] == 'ac':
                if freq == 0:
                    print("Frequency not specified")
                    exit()
                isac = True
                val = 0.5*float(line[4])*np.exp(1j*float(line[5])*np.pi/180)
            else:
                print("Invalid definition for voltage source")
                exit()
            M[len(nodes) + vol_srcs[line[0]], nodes[line[1]]] += 1
            M[len(nodes) + vol_srcs[line[0]], nodes[line[2]]] -= 1
            M[nodes[line[1]], len(nodes) + vol_srcs[line[0]]] += 1
            M[nodes[line[2]], len(nodes) + vol_srcs[line[0]]] -= 1
            b[len(nodes) + vol_srcs[line[0]]] += val
        elif line[0].startswith('I'):
            val = float(line[3])
            b[nodes[line[1]]] -= val
            b[nodes[line[2]]] += val
        elif line[0].startswith('L'):
            val = float(line[3])
            if freq != 0:
                imp = (2j*np.pi*freq*val)
            else:
                imp = 1e-6
            M[nodes[line[1]], nodes[line[1]]] += 1/imp
            M[nodes[line[1]], nodes[line[2]]] -= 1/imp
            M[nodes[line[2]], nodes[line[1]]] -= 1/imp
            M[nodes[line[2]], nodes[line[2]]] += 1/imp    
        elif line[0].startswith('C'):
            val = float(line[3])
            if freq != 0:
                imp = 1/(2j*np.pi*freq*val)
                M[nodes[line[1]], nodes[line[1]]] += 1/imp
                M[nodes[line[1]], nodes[line[2]]] -= 1/imp
                M[nodes[line[2]], nodes[line[1]]] -= 1/imp
                M[nodes[line[2]], nodes[line[2]]] += 1/imp
                
    M[0] = 0; b[0] = 0 #Setting the first equation as V0 = 0 (reference voltage)
    M[0, 0] = 1
    x = np.linalg.solve(M, b)
    for i in range(1, len(nodes) + len(vol_srcs)):
        if isac:
            if i < len(nodes):
                print(f"Node voltage of {list(nodes.keys())[i]} is Magnitude: %.2e V, Phase: %.4f degrees" %(np.abs(x[i]), np.angle(x[i])*180/np.pi))
            else:
                print(f"Current through {list(vol_srcs.keys())[i-len(nodes)]} is Magnitude: %.2e A, Phase: %.4f degrees" %(np.abs(x[i]), np.angle(x[i])*180/np.pi))
        else:    
            if i < len(nodes):
                print(f"Node voltage of {list(nodes.keys())[i]} is %.2e V" %(np.real(x[i])))
            else:
                print(f"Current through {list(vol_srcs.keys())[i-len(nodes)]} is %.2e A" %(np.real(x[i])))