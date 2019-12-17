## Residue_E_Decomp.py
##  written by Lalith Perera for energy analysis
##  June 19 2009
##  modified by G. Andres Cisneros to f90
##  June 30 2009
##  modified by EML to python3.6
##  ?? ?? 2019

## Info on MPI4Py: https://mpi4py.readthedocs.io/en/stable/tutorial.html

#from mpi4py import MPI
import sys
import parmed as pmd
from datetime import datetime
import numpy as np
#import scipy

## Give the answers file as program input
## When operating the program, use "python prog_name.py answers_file"
ans = sys.argv[1]
#ans = "./answers_file"

## Let's start by defining some tasks

## Stuff for MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()


## Define the conversion
## coulomb_const_kcal_per_mole = 332.0538200
amber_q = 18.2223

## Read the answers file
def readans(ans):
	## The answers file contains:
	## L1: The name of the EDA input file
	## L2: The name of the prmtop file
	##
	## Read the file line-by-line
	## Store the lines to "ansl" list
	ansf = open(ans,"r")
	ansl = []
	for line in ansf:
		ansl.append(line)
	ansf.close()
	return ansl


## Read the EDA input for information
def readEDA(EDA_inp):
	## n_res = number of protein residues
	## n_files = number of files
	## nat_max = total number of atoms
	## nprotat = number of protein atoms
	## nres_max = number of total residues
	## ntype_max = max number of types
	parts = []
	with open(EDA_inp, "r") as EDAf:
		for index, line in enumerate(EDAf, start=1):
			parts.append(line.split(" !"))
	## Only take the numbers and mdcrd lines
	clean = []
	test = int(parts[1][0]) + 6 - 1
	test2 = int(parts[1][0]) + test
	## Remove newlines from mdcrd lines
	for x in range(test,test2):
		parts[x][0] = parts[x][0].rstrip()
	## Make the clean list
	for x in range(test2):
		clean.append(parts[x][0])
	n_res     = clean[0]
	n_files   = clean[1]
	nat_max   = clean[2]
	nprotat   = clean[3]
	nres_max  = clean[4]
	ntype_max = clean[5]
	##
	EDAf.close()
	##
	return n_res, n_files, nat_max, nprotat, nres_max, ntype_max, clean


## Read the prmtop for information
def readparm(prmfile):
	parm = pmd.load_file(prmfile)
	#
	natom    = parm.ptr('NATOM')
	ntypes   = parm.ptr('NTYPES')
	nbonh    = parm.ptr('NBONH')
	mbona    = parm.ptr('MBONA')
	ntheth   = parm.ptr('NTHETH')
	mtheta   = parm.ptr('MTHETA')
	nphih    = parm.ptr('NPHIH')
	mphia    = parm.ptr('MPHIA')
	nhparm   = parm.ptr('NHPARM')
	nparm    = parm.ptr('NPARM')
	nnb      = parm.ptr('NNB')
	nres     = parm.ptr('NRES')
	nbona    = parm.ptr('NBONA')
	ntheta   = parm.ptr('NTHETA')
	nphia    = parm.ptr('NPHIA')
	numbnd   = parm.ptr('NUMBND')
	numang   = parm.ptr('NUMANG')
	nptra    = parm.ptr('NPTRA')
	natyp    = parm.ptr('NATYP')
	nphb     = parm.ptr('NPHB')
	ifpert   = parm.ptr('IFPERT')
	nbper    = parm.ptr('NBPER')
	ngper    = parm.ptr('NGPER')
	ndper    = parm.ptr('NDPER')
	mbper    = parm.ptr('MBPER')
	mgper    = parm.ptr('MGPER')
	mdper    = parm.ptr('MDPER')
	ifbox    = parm.ptr('IFBOX')
	nmxrs    = parm.ptr('NMXRS')
	ifcap    = parm.ptr('IFCAP')
	numextra = parm.ptr('NUMEXTRA')
	#
	## NCOPY is sometimes missing, so handle that
	try:
		ncopy = parm.ptr('NCOPY')
	except KeyError:
		ncopy = 0
	#
	## print("Finished allocating arrays")
	print(" \n \
	NATOM  = %7s NTYPES = %7s NBONH = %7s MBONA  = %7s\n \
	NTHETH = %7s MTHETA = %7s NPHIH = %7s MPHIA  = %7s\n \
	NHPARM = %7s NPARM  = %7s NNB   = %7s NRES   = %7s\n \
	NBONA  = %7s NTHETA = %7s NPHIA = %7s NUMBND = %7s\n \
	NUMANG = %7s NPTRA  = %7s NATYP = %7s NPHB   = %7s\n \
	IFBOX  = %7s NMXRS  = %7s IFCAP = %7s NEXTRA = %7s\n \
	NCOPY  = %7s\n" %( \
	natom,  ntypes, nbonh, mbona, \
	ntheth, mtheta, nphih, mphia, \
	nhparm, nparm,  nnb,   nres, \
	nbona,  ntheta, nphia, numbnd, \
	numang, nptra,  natyp, nphb, \
	ifbox,  nmxrs,  ifcap, numextra, \
	ncopy))
	#
	return natom, ntypes, nres, parm


## Get atom charges from prmtop
def at_chg(parm, nprotat, natom):
	## Stopping at the number of protein atoms
	astop = int(nprotat)
	## List of unique atom types; list of all atom types
	charge_list = []
	all_charges_list = []
	for atom in parm.atoms:
		all_charges_list.append(atom.charge)
	#
	# AMBER uses q(internal units) = q(electron charge units)*18.2223
	# So you have to adjust parmed's adjustment...
	for x in range(0,astop):
		charge_list.append(all_charges_list[x] * amber_q)
	n_tot_charges = len(all_charges_list)
	if n_tot_charges != natom:
		sys.exit("Hey, buddy, something's wrong with the charges.")
	return charge_list


## Get atomic numbers from prmtop
def at_num_index(parm, nprotat, natom):
	## Stopping at the number of protein atoms
	astop = int(nprotat)
	## List of unique atom types; list of all atom types
	at_num_list = []
	all_at_num_list = []
	for atom in parm.atoms:
		all_at_num_list.append(atom.atomic_number)
	#
	for x in range(0,astop):
		at_num_list.append(all_at_num_list[x])
	n_tot_at_index = len(all_at_num_list)
	if n_tot_at_index != natom:
		sys.exit("Hey, friend, something's wrong with the atomic \
		 numbers.")
	return at_num_list


## Get Amber Atom Types (string) from prmtop
def am_at_type(parm, natom):
	## List of unique atom types; list of all atom types
	am_at_type_list = []
	all_atom_types = []
	for atom in parm.atoms:
		all_atom_types.append(atom.type)
	#
	for x in all_atom_types:
		if x not in am_at_type_list:
			am_at_type_list.append(x)
	natom_types = len(all_atom_types)
	if natom_types != natom:
		sys.exit("Hey, human, something's wrong with the atom types.")
	return all_atom_types, am_at_type_list


## Get Indexed Amber Atom Types (integer) from prmtop
def am_at_type_idx(parm, natom, all_atom_types):
	## List of unique atom types indices; list of all atom types indices
	## Locations of unique atom types for LJ-determination
	am_at_type_idx_list = []
	all_atom_idx_types = []
	for item in parm.parm_data['ATOM_TYPE_INDEX']:
		all_atom_idx_types.append(item)
	## Check against the Amber Atom Type (string) codes 
	k1 = 1
	k2 = 0
	for k2 in range(natom):
		if k1 == all_atom_idx_types[k2]:
			am_at_type_idx_list.append(all_atom_types[k2])
			k1 += 1
			k2 += 1
		else:
			k2 += 1
			continue
	natom_idx_types = len(all_atom_idx_types)
	if natom_idx_types != natom:
		sys.exit("Hey, peer, something's wrong with the indexed \
		 atom types.")
	return all_atom_idx_types, am_at_type_idx_list


## Get the van der Waals parameters in terms of combos
##
## You need interaction pairs, so use the A/B Coeff
## Thus, do not use something like
## sigma = parm.atoms[k1].sigma and epsilon = parm.atoms[k1].epsilon
## These are based on atom types!
##
def lj_pairwise(parm, ntypes):
	xlj_12 = parm.parm_data['LENNARD_JONES_ACOEF']
	xlj_6 = parm.parm_data['LENNARD_JONES_BCOEF']
	lj_param = []
	print('ntypes = {}'.format(ntypes))
	## Create the arrays containing LJ cross terms for later use
	mat_size = (ntypes,ntypes)
	lj_12_vals = np.zeros(mat_size)
	lj_6_vals = np.zeros(mat_size)
	k1 = 0
	for k2 in range(0,ntypes):
		## You need this value to go k2's "final" value of loop
		## So if iteration 1 for k2, need 1 for k3
		## In terms of range commands, this means k2+1
		for k3 in range(0,k2+1):
			lj_12_vals[k2][k3] = xlj_12[k1]
			lj_12_vals[k3][k2] = xlj_12[k1]
			lj_6_vals[k2][k3] = xlj_6[k1]
			lj_6_vals[k3][k2] = xlj_6[k1]
			if (abs(xlj_6[k1]) >= 0.000000001) \
			or (abs(xlj_12[k1]) >= 0.000000001):
				sigma = (xlj_12[k1]/xlj_6[k1])**(1.0/6.0)
				epsilon = (xlj_6[k1]**2)/(4.0*xlj_12[k1])
			else:
				sigma = 0.0
				epsilon = 0.0
			re = sigma * 2.0**(1.0/6.0)
			if k2 == k3:
				lj_param.extend([(k2+1), (k3+1), re/2.0, epsilon])
			k1 += 1
			if k2 == 0 and k3 == 0:
				print(k2,k3,lj_12_vals[k2][k3],lj_6_vals[k2][k3], k1-1, xlj_12[k1-1], xlj_6[k1-1])
			if k2 == (ntypes-1) and k3 == (ntypes-1):
				print(k2,k3,lj_12_vals[k2][k3],lj_6_vals[k2][k3], k1-1, xlj_12[k1-1], xlj_6[k1-1])
	the_LJ_k = k1
	return lj_param, lj_12_vals, lj_6_vals, the_LJ_k


## Get the total number of atoms in a residue
## Base this off of n_res from the EDA input
## NOT the nres in the prmtop
def get_res_point(parm, ntypes, n_res):
	i_res_point = parm.parm_data['RESIDUE_POINTER']
	nat_in_res = np.zeros(n_res,dtype=int)
#	print("n_res = " + str(n_res))
	for k1 in range(0,n_res):
		nat_in_res[k1] = (i_res_point[k1+1] - i_res_point[k1])
		k1 += 1
	return nat_in_res, i_res_point

## Perform the sanity check
def sanity_check(am_at_type_list, am_at_type_idx_list, lj_param, nat_in_res):
	## For printing the standard AMBER Atom Types
	first_check = []
	for x in range(len(am_at_type_list)):
		first_check.append( str((x+1)) + am_at_type_list[x] )
	#
	## For printing the "clean" AMBER Atom Types
	second_check_values = []
	second_check_index = []
	for x in range(len(am_at_type_idx_list)):
		second_check_index.append( str((x+1)) )
		second_check_values.append( str(am_at_type_idx_list[x]) )
	#
	## For printing the LJ parameters
	third_check = []
	x = 0
	while (x+3) < len(lj_param):
		third_check.append( ' {:>9} {:>9} {:9.4f} {:9.4f}'.format(lj_param[x], \
		 lj_param[x+1], lj_param[x+2], lj_param[x+3]) )
		x += 4
	#
	## For printing the number of atoms in a residue
	## Write 20 values per line
	fourth_check = []
	for item in range(0,len(nat_in_res)):
		x = item
		fourth_check.append( '{:>4}'.format( str(nat_in_res[item])) )
		if x % 20 == 19:
			fourth_check.append( "\n" )
	#
	now = datetime.now()
	dt_string = now.strftime("%b %d, %Y %H:%M:%S")
	#
	with open("EDA_sanity_check-py.txt", "w+") as san_out:
		san_out.write("\n")
		san_out.write("Energy Decomposition Analysis (EDA)\n")
		san_out.write("Python Version: TEST\n")
		san_out.write("I was run: ")
		san_out.write(dt_string)
		san_out.write("\n\n")
		san_out.write("\n".join(first_check))
		san_out.write("\n")
		san_out.write("Done writing the standard Amber atom types!\n")
		san_out.write("\n".join('{:>3} {:3}'.format(a,b) for a,b in \
        	 zip(second_check_index,second_check_values)))
		san_out.write("\n")
		san_out.write("Done writing the atom types!\n")
		san_out.write("\n".join(third_check))
		san_out.write("\n")
		san_out.write("Done writing the LJ parameters! \t")
		san_out.write(str(the_LJ_k))
		san_out.write("\n")
		san_out.write("".join(fourth_check))
		san_out.write("\n")
		san_out.write("Done writing the number of atoms in a residue!\n")
		san_out.close()

### FROM OTHER
## Assign the charges
def assn_charges(parm, n_res, nat_in_res, i_res_point):
	k_bin = 0
	charge_ij = np.zeros(int(natpair))
	vdw_12 = np.zeros(int(natpair))
	vdw_6 = np.zeros(int(natpair))
	iflag = np.zeros(int(natpair), dtype=int)
	at_index_i = np.zeros(int(natpair), dtype=int)
	at_index_j = np.zeros(int(natpair), dtype=int)
	#
	print_index = np.zeros(int(npairs), dtype=int)
	#
	k1 = 0
	## For residue k2
	for k2 in range(0,n_res-1):
		## For residue k4
		## *** WARNING: ***
		## k4 should start 2 res away from k2. Otherwise
		## 1-2, 1-3, & 1-4 interactions will result in
		## erroneous energies.
		for k4 in range(k2+1,n_res):
			for k3 in range(0,nat_in_res[k2]):
				## Initial i_res_point = 1
				## You need to get rid of that 1 for later
				at_id_i = int(i_res_point[k2] + k3 -1 )
				### List of atom idxx types starts with 1, not 0
				at_type_i = all_atom_idx_types[at_id_i] - 1
				## For atom k5 of residue k4
				for k5 in range(0,nat_in_res[k4]):
					at_id_j = int(i_res_point[k4] + k5 -1 )
					at_type_j = \
					 all_atom_idx_types[at_id_j] - 1
					## Assign the charge
					charge_ij[k1] = (charge_list[at_id_i] \
					 * charge_list[at_id_j] )
					## Assign LJ parameters for each atom
					vdw_12[k1] = ( \
					 lj_12_vals[at_type_i][at_type_j] )
					vdw_6[k1] = ( \
					 lj_6_vals[at_type_i][at_type_j] )
					iflag[k1] =  ( k_bin )
					at_index_i[k1]= ( at_id_i )
					at_index_j[k1]= ( at_id_j )
					k1 += 1
			k_bin += 1
			print_index[k_bin - 1] = (k_bin)
	k_bin_tot = k_bin
	print('k_bin_tot = {}'.format(k_bin_tot))
	return charge_ij, vdw_12, vdw_6, iflag, at_index_i, at_index_j, \
	 k_bin_tot, print_index


## Read in the trajectories
## Creates a dictionary of trajectories
## Either ASCII mdcrds or NetCDFs can be read in using the
## load_file feature, provided natom and hasbox are defined
## for the mdcrd use.
def read_traj_files(n_files, clean, natom):
	trajs = {}
	total_samples = np.zeros(int(n_files), dtype=int) 
	for x in range(int(n_files)):
		## Fix the string issue
		clean[x+6] = clean[x+6].replace("'","")
		trajs[x] = pmd.load_file(filename=clean[x+6], \
		 natom=natom, hasbox=False)
		total_samples[x] = ( len(trajs[x].coordinates) )
	#
	n_file_sam = total_samples.sum()
	print('number of samples in file: n_file_sam = ' + str(n_file_sam))
	#
	return trajs, n_file_sam

## Create a progress bar for the number of samples
def progressbar(it, prefix="", size=60, file=sys.stdout):
	count = len(it)
	def show(j):
		x = int(size*j/count)
		file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
		file.flush()        
	show(0)
	for i, item in enumerate(it):
		yield item
		show(i+1)
	file.write("\n")
	file.flush()


## Perform the calculation
def traj_coul_vdw(n_files, n_file_sam, trajs, at_index_i, at_index_j, charge_ij, \
 vdw_12, vdw_6, iflag, E_coul_list, E_vdW_list, SE_coul_list, SE_vdW_list):
	ResA = np.zeros([int(npairs)], dtype=int)
	ResB = np.zeros([int(npairs)], dtype=int)
#	for x in range(0,int(n_files)):
	for x in progressbar(range(0,int(n_files)), "File: ", 60):
		## More easily access trajs[x].coordinates
		file_coords = trajs[x].coordinates
#		for trial in range(0,n_file_sam):
		for trial in progressbar(range(0,n_file_sam), "Frame: ", 60):
			k1 = 0
			## Residue k2
			for k2 in range(0, n_res-1):
				## Residue k4
				for k4 in range(k2+1, n_res):
					## Atom k3 of residue k2
					for k3 in range(0,nat_in_res[k2]):
						## Atom k5 of residue k4
						for k5 in range(0,nat_in_res[k4]):
							m1 = at_index_i[k1]
							m2 = at_index_j[k1]
							#
							## Time comes from A,
							## atom comes from B
							## x/y/z comes from C
							## mdcrd[x].coordinates[A][B][C]
							xd = \
							 file_coords[trial][m1][0] \
							 - file_coords[trial][m2][0]
							yd = \
							 file_coords[trial][m1][1] \
							 - file_coords[trial][m2][1]
							zd = \
							 file_coords[trial][m1][2] \
							 - file_coords[trial][m2][2]
							#
							rd = xd**2 + yd**2 + zd**2
							rd6 = rd * rd * rd
							rd12 = rd6 * rd6
							#
							## Coulomb Energy
							E_coul = charge_ij[k1] / \
							 np.sqrt(rd)
							#
							## van der Waals Energy
							E_vdW = vdw_12[k1]/rd12 - \
							 vdw_6[k1]/rd6
							#
							## Store them away...
							k_bin = iflag[k1]
							E_coul_list[k_bin] += E_coul
							E_vdW_list[k_bin] += E_vdW
							SE_coul_list[k_bin] += E_coul**2
							SE_vdW_list[k_bin] += E_vdW**2
							k1 += 1
					## ResNames for printing
					ResA[k_bin] = (k2 + 1)
					ResB[k_bin] = (k4 + 1)
#			print("Glad to report sample " + str(trial+1) + " of " + \
#			 str(n_file_sam) + " is done.")
	return E_coul_list, SE_coul_list, E_vdW_list, SE_vdW_list, ResA, ResB



##-------------------------------------------------------------------------##
##----------------------------- START PROGRAM -----------------------------##
##-------------------------------------------------------------------------##

## Deal with MPI
'''
## Create an MPI logfile
with open("EDA_mpi-py.log", "w+") as mp_out:
	mp_out.write("\n")
	mp_out.write("Energy Decomposition Analysis (EDA)\n")
	mp_out.write("Python Version: TEST\n")
	mp_out.write("\n")
	mp_out.write("Program found ")
	mp_out.write(str(proc_num))
	mp_out.write(" available processors.\n")
	mp_out.write("Program found ")
	mp_out.write(str(thread_num))
	mp_out.write(" available threads.\n")
	mp_out.close()
'''

##--------------------------##
##---- Read the Answers ----##
##--------------------------##
## Read the answer file and get the input and prmtop names
## These are defined as "EDA_inp" and "prmfile"
ansl = readans(ans)
EDA_inp = ansl[0].rstrip()
prmfile = ansl[1].rstrip()

##----------------------------##
##---- Read the EDA input ----##
##----------------------------##

n_res, n_files, nat_max, nprotat, nres_max, ntype_max, clean = readEDA(EDA_inp)
	
n_res = int(n_res)
npairs = (n_res*(n_res-1))/2
natpair = (int(nprotat)*(int(nprotat)+1))/2

##-------------------------##
##---- Read the prmtop ----##
##-------------------------##
## Initial opening of file
natom, ntypes, nres, parm = readparm(prmfile)

## Get the charges
charge_list = at_chg(parm, nprotat, natom)

## Get the atomic numbers
at_num_list = at_num_index(parm, nprotat, natom)

## Get the standard AMBER atom types
all_atom_types, am_at_type_list = am_at_type(parm, natom)

## Get the "clean" AMBER atom types
all_atom_idx_types, am_at_type_idx_list = am_at_type_idx(parm, natom, \
 all_atom_types)

## Get the vdW pairing and LJ parameters
lj_param, lj_12_vals, lj_6_vals, the_LJ_k = lj_pairwise(parm, ntypes)

## Get the number of atoms per residue
nat_in_res, i_res_point = get_res_point(parm, ntypes, n_res)

##------------------------------##
##---- Run the Sanity Check ----##
##------------------------------##
sanity_check(am_at_type_list, am_at_type_idx_list, lj_param, nat_in_res)


##-----------------------------------------------##
##---- Set up empty energy and std err lists ----##
##-----------------------------------------------##
## Arrays of 1 x npairs
E_coul_list = np.zeros([int(npairs)])
SE_coul_list = np.zeros([int(npairs)])
E_vdW_list = np.zeros([int(npairs)])
SE_vdW_list = np.zeros([int(npairs)])

## Assign the charges
#charge_ij, vdw_12, vdw_6, iflag, at_index_i, at_index_j, k_bin_tot = \
# assn_charges(parm, n_res, nat_in_res, i_res_point)

## Assign the charges
charge_ij, vdw_12, vdw_6, iflag, at_index_i, at_index_j, k_bin_tot, \
 print_index = assn_charges(parm, n_res, nat_in_res, i_res_point)


##------------------------------------##
##---- Deal with the Trajectories ----##
##------------------------------------##
trajs, n_file_sam = read_traj_files(n_files, clean, natom)


##-----------------------------##
##---- Get to calculating! ----##
##-----------------------------##
E_coul_list, SE_coul_list, E_vdW_list, SE_vdW_list, ResA, ResB = \
 traj_coul_vdw(n_files, n_file_sam, trajs, at_index_i, at_index_j, charge_ij, \
 vdw_12, vdw_6, iflag, E_coul_list, E_vdW_list, SE_coul_list, SE_vdW_list)


##-----------------------------##
##---- Print the Coul File ----##
##-----------------------------##
## Write the Coulomb header. np.savetxt will add a "# " at the beginning
coul_head = '{:>8} {:>10} {:>10} {:>20} {:>20}'.format('Index', 'ResA', 'ResB', \
 'CoulEnergy', 'CoulStdErr')

with open("EDA_coul_results-inputvers-py.txt", "w+") as coul_out:
	np.savetxt(coul_out, np.c_[print_index,ResA,ResB,np.divide(E_coul_list,\
	int(n_file_sam)),np.divide(SE_coul_list,int(n_file_sam))], \
	header=coul_head,fmt='%10i %10i %10i %20.12E %20.12E')


##------------------------------##
##----- Print the vdW File -----##
##------------------------------##
## Write the vdW header. np.savetxt will add a "# " at the beginning
vdW_head = '{:>8} {:>10} {:>10} {:>20} {:>20}'.format('Index', 'ResA', 'ResB', \
 'vdWEnergy', 'vdWStdErr')

with open("EDA_vdW_results-inputvers-py.txt", "w+") as vdw_out:
	np.savetxt(vdw_out, np.c_[print_index,ResA,ResB,np.divide(E_vdW_list,\
	int(n_file_sam)),np.divide(SE_vdW_list,int(n_file_sam))], \
	header=vdW_head,fmt='%10i %10i %10i %20.12E %20.12E')

print("What, you wanted an end quote? Grad students don't have time for that.")
