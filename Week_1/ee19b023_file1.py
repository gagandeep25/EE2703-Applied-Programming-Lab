#Pseudo Code for the program:
#take file as commandline input
#throw errors for wrong execution, wrong filename and if file doens't exist
#open(file); data = read_each_line(file); close(file)
#remove all the comments and empty lines if any from the data
#reverse data, to loop from bottom to top
#throw errors if circuit definiton is invalid
#for line in data:
#    check if line is .end and set a flag
#    check if line is .circuit and set a flag
#    if line is neither of them:
#        use the flag to determine if it belongs to the circuit
#        reverse the elements and print

import sys

circuit = '.circuit'; end = '.end'
if len(sys.argv) != 2:
    print("Error! Please execute using the format:\npython3 <ee19b023_file1.py> <path to netlist file>")
elif '.netlist' not in sys.argv[1]:
    print("Error! Please make sure the filename ends with .netlist")
else:
    try:
        with open(sys.argv[1]) as f:
            inckt = False
            cleanlines = list(filter(None, [line.split('#')[0].strip() for line in f.readlines()[::-1]]))
            if cleanlines.count(circuit) == 1 and cleanlines.count(end) == 1:
                if cleanlines.index(circuit) > cleanlines.index(end):
                    for l in cleanlines:
                        if l == end:
                            inckt = True
                        elif l == circuit:
                            inckt = False
                        elif inckt:
                            print(' '.join(l.split()[::-1]))
                else:
                    print("Invalid circuit definition")
            else:
                print("Invalid circuit definition")
    except FileNotFoundError:
        print(f'File named {sys.argv[1]} does not exist')
