from brian2 import *
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os

startbuild = time.time() #used to compute the time to run some part of the code

savedir = "brian_data/"

if not os.path.exists(savedir):
    os.makedirs(savedir)

runtime = 1000.0 * ms # How long you want the simulation to be
dt_sim=0.1 #ms         # Resolution of the simulation
G=5 #global constant to increase all the connections weights
Gl1=5 #global constant to increase all the connections weights in Layer 1
Ntot=5000 #Total number of neurons in the simulation

np.random.seed(200) #If you want the exact same simulation every time uncomment this

# External input to the neurons
Iext=np.loadtxt('import_files/Iext0.txt') #File that contain the external input for layer 2/3,4,5,6
Iext_l1= 0 #External input to layer 1
Iext1=np.loadtxt('import_files/Iext0.txt') #File that contain the external input (if you want a second input to come at a later time you also need this)


# Background noise to the neurons
nu_ext_file='import_files/nu_ext.txt' #This is the file that contains the background noise to the neurons
nu_ext=np.loadtxt(nu_ext_file) #I upload the file containing the backround noise to the neurons
nu_extl1= 650 *Hz

# If you want different receptor weight for each particular group receiving the external background noise- you will also need this:
#wext=np.loadtxt('import_files/wext.txt')
# I chose to have the same weight for all the receptors of the different neurons type receiving the external noise

##### ----------------------------------------------#####
##### Percentage of neurons in the simulation       #####
##### --------------------------------------------- #####

# Percentage of neurons in each layer based on experimental data 
N1=int(0.0192574218*Ntot) # N1 is not included in the calculation of Ntot, Ntot is just the sum of 4 layers.
N2_3=int(0.291088453*Ntot) # This percentage are computed looking at the numbers of the Allen institute
N4=int(0.237625904*Ntot)
N5=int(0.17425693*Ntot)
N6= Ntot-N2_3-N4-N5
#N6=int(0.297031276*Ntot)
#print(N2_3+N4+N5+N6)

perc_tot=np.loadtxt('import_files/perc_tot.txt') #Matrix containing the percentage of excitatory and inhib neurons
#print(perc_tot)
perc=np.loadtxt('import_files/perc.txt') #Matrix containing the percentage of neurons for each type in each layer
#print(perc)
n_tot= np.array([[N2_3,N2_3,N2_3,N2_3],[N4,N4,N4,N4],[N5,N5,N5,N5],[N6,N6,N6,N6]]) # total number of neurons in each layer,
# I need this n_tot to then be able to create the matrix N which contains the exact number of each neurons for this simulation
#print(n_tot)

N=perc*perc_tot*n_tot #Matrix containing numbers of neurons for each type in each layer
N=np.matrix.round(N)  #I round it to the nearest integer value
N=N.astype(int) # Number of neurons should be of type int
# Now I correct the matrix I obtained:
# the sum of each layer should return the total number of neurons in that layer
for k in range(0,4):
    N[k][0]+= n_tot[k][0]-sum(N[k]) #sum(N[k]) is the total number of neurons in that layer I have in the matrix
                                    #n_tot[k][0] is the number of neuorons in that layer from the percentages
                                    #If the two numbers don't match the N[k][0] (excitaotry in each layer) gets updated

##### ----------------------------------------------#####
#####               Synapse model                   #####
##### --------------------------------------------- #####

# Percentage of AMPA and NMDA receptors in excitatory and inhibitory neurons
e_ampa=0.8
e_nmda=0.2
i_ampa=0.8
i_nmda=0.2

w_ext=1                                 #Weight for the external background noise going to AMPA receptor. Is the same value for every population.
                                        #This is the weight in front of s_tot. (see below in eq. ampa ext)
                                        #If you don't want it to be the same for every population:
                                        #wext is also in the equations of AMPA ext (see below in eq. ampa ext), if needed just uncomment it there.

gext=1                                  #How much you affect s_ampa with 1 spike from the Poisson generator.

#V_I = -70 * mV  each group has his own!!  # Reversal potential of inhibitory synapses
V_E = 0 * mV                               # Reversal potential of excitatory synapses
tau_AMPA = 2.0 * ms                        # Decay constant of AMPA-type conductances
tau_GABA = 5.0 * ms                        # Decay constant of GABA-type conductances
tau_NMDA_decay = 80.0 * ms                 # Decay constant of NMDA-type conductances
tau_NMDA_rise = 2.0 * ms                   # Rise constant of NMDA-type conductances
alpha_NMDA = 0.5 * kHz                     # Saturation constant of NMDA-type conductances
Mg2 = 1.                                   # Magnesiumn concentration
d = 2 * ms                                 # Transmission delay of recurrent excitatory and inhibitory connections

# print("Importing the data")
# Matrices containing all the connections (probabilities and strenght) 
# This are data from the Allen database
Cp = np.loadtxt('import_files/connectionsPro_final.txt') #connenctions probabilities between the 16 groups in the 4 layers (not VIP1)
Cs=np.loadtxt('import_files/connectionsStren.txt') #connenctions strenghts between the 16 groups in the 4 layers (not VIP1)

Cpl1 = np.loadtxt('import_files/Cpl1_final.txt') #connenctions probabilities from each of the 16 groups and VIP1
Csl1=np.loadtxt('import_files/Csl1.txt') #connenctions strenghts from each of the 16 groups and VIP1
Cp_tol1 = np.loadtxt('import_files/Cptol1_final.txt') #connenctions probabilities from VIP1 to each of the 16 groups
Cs_tol1 = np.loadtxt('import_files/Cstol1.txt') #connenctions strengths from VIP1 to each of the 16 groups
Cs_l1_l1=1.73 #connenctions strengths from VIP1 to VIP1
Cp_l1_l1=0.7371*0.656 #connenctions probability from VIP1 to VIP1

# Parameters of the neurons for each layer
# row: layer in this order from top to bottom: 2_3,4,5,6
# column: populations in this order: e, pv, sst, vip
Cm=np.loadtxt('import_files/Cm.txt') #pF
gl=np.loadtxt('import_files/gl.txt') #nS
Vl=np.loadtxt('import_files/Vl.txt') #mV
Vr=np.loadtxt('import_files/Vr.txt') #mV
Vt=np.loadtxt('import_files/Vt.txt') #mV
tau_ref=np.loadtxt('import_files/tau_ref.txt') #ms

#Parameters of VIP1
Nl1= N1
Vt_l1= -40.20
Vr_l1= -65.5
Cm_l1= 37.11
gl_l1= 4.07
Vl_l1= -65.5
tau_ref_l1= 3.5

#To compute the time of import of all the files
end_import= time.time() 

# Check if everything is correct 
# print('------------------Check--------------------------------')
# print('Cm')
# print(Cm)
# print('gl')
# print(gl)
# print(Vl)
# print(Vt)
# print(type(Vt[0][1]))
# print(tau_ref)

##### ----------------------------------------------#####
#####             Equations of synapse model        #####
##### --------------------------------------------- #####

# Equations of the model. 
# Each neuron is governed by this equations
eqs='''
        dv / dt = (- g_m * (v - V_L) - I_syn) / C_m : volt (unless refractory)
        I_syn = I_AMPA_rec + I_AMPA_ext + I_GABA + I_NMDA + I_external: amp

        # Parameters that can differ for each type of neuron, they are internal variable of the neuron.
        # This way I can then set their value later when I build the population. Each population can have a different value
        C_m : farad
        g_m: siemens
        V_L : volt
        V_rest : volt
        Vth: volt
        g_AMPA_ext: siemens
        g_AMPA_rec : siemens
        g_NMDA : siemens
        g_GABA :siemens

        # Here I define the external inputs. 
        # Depending on what I want to study I can use one of this equations. 
        # Just uncomment the one you want.

        # If I want same input for the entire simulation use this:
        I_external = I_ext: amp

        # When I want no input and then input activated use this:
        #I_external= (abs(t-t1*ms+0.001*ms)/(t-t1*ms+0.001*ms) + 1)* (I_ext/2) : amp #at the beginnig is 0 then the input is activated at t1
            
        # If I want: at the beginnig I_ext is 0 then the input is activated at t1 then deactivated at t2 then activated again at t3 use this:
        #I_external= (abs(t-t1*ms+0.001*ms)/(t-t1*ms+0.001*ms) + 1) * (I_ext/2)-(abs(t-t2*ms+0.001*ms)/(t-t2*ms+0.001*ms) + 1) * (I_ext/2) + (abs(t-t3*ms+0.001*ms)/(t-t3*ms+0.001*ms) + 1) * (I_ext/2)- (abs(t-t4*ms+0.001*ms)/(t-t4*ms+0.001*ms) + 1) * (I_ext/2) : amp

        # When I want 2 inputs at different times going to different layers use this:
        # I need Iext and Iext1
        #I_external= (abs(t-t1*ms+0.001*ms)/(t-t1*ms+0.001*ms) + 1)* (I_ext/2) + (abs(t-t2*ms+0.001*ms)/(t-t2*ms+0.001*ms) + 1)* (I_ext1/2) : amp #at the beginnig is 0 then the input is activated

        # These are also variable of each neuron, I can later set the value I want when I build them
        I_ext : amp
        I_ext1 : amp #this is the second input to the other layer


        # Equations for AMPA receiving the inputs from the background (Poisson genetors:)

        I_AMPA_ext= g_AMPA_ext * (v - V_E) * w_ext * s_AMPA_ext : amp
        ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
        #w_ext: 1 (If you want to have different weight fo each group, use this and uncomment later in pop.w_ext to set the desired value)
        # Here I don't need the summed variable because the neuron receive inputs from only one Poisson generator.
        # Each neuron need only one s.

        # Equations for AMPA receiving the inputs from other neurons:

        I_AMPA_rec = g_AMPA_rec * (v - V_E) * 1 * s_AMPA_tot : amp
        s_AMPA_tot=s_AMPA_tot0+s_AMPA_tot1+s_AMPA_tot2+s_AMPA_tot3 : 1
        s_AMPA_tot0 : 1
        s_AMPA_tot1 : 1
        s_AMPA_tot2 : 1
        s_AMPA_tot3 : 1
        # the eqs_ampa solve many s and sum them and give the summed value here
        # Each neuron receives inputs from many neurons. Each of them has his own differential equation s_AMPA (where I have the deltas with the spikes)
        # I then sum all the solutions s of the differential equations and I obtain s_AMPA_tot_post
        # One s_AMPA_tot from each group of neurons sending excitation (each neuron is receiving from 4 groups)


        # Equations for GABA receiving the inputs from other neurons:

        I_GABA= g_GABA * (v - V_I) * s_GABA_tot : amp
        V_I : volt

        s_GABA_tot=s_GABA_tot0+s_GABA_tot1+s_GABA_tot2+s_GABA_tot3+s_GABA_tot4+s_GABA_tot5
                    +s_GABA_tot6+s_GABA_tot7+s_GABA_tot8+s_GABA_tot9+s_GABA_tot10+s_GABA_tot11+s_GABA_tot12: 1
        s_GABA_tot0 : 1
        s_GABA_tot1 : 1
        s_GABA_tot2 : 1
        s_GABA_tot3 : 1
        s_GABA_tot4 : 1
        s_GABA_tot5 : 1
        s_GABA_tot6 : 1
        s_GABA_tot7 : 1
        s_GABA_tot8 : 1
        s_GABA_tot9 : 1
        s_GABA_tot10 : 1
        s_GABA_tot11 : 1
        s_GABA_tot12: 1

        # Equations for NMDA receiving the inputs from other neurons:

        I_NMDA  = g_NMDA * (v - V_E) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) * s_NMDA_tot : amp
        s_NMDA_tot=s_NMDA_tot0+s_NMDA_tot1+s_NMDA_tot2+s_NMDA_tot3 : 1
        s_NMDA_tot0 : 1
        s_NMDA_tot1 : 1
        s_NMDA_tot2 : 1
        s_NMDA_tot3 : 1

     '''

# This is the general ampa equation for each neuron type
eqs_ampa_base='''
            s_AMPA_tot_post= w_AMPA* s_AMPA : 1 (summed) #sum all the s, one for each synapse
            ds_AMPA / dt = - s_AMPA / tau_AMPA : 1 (clock-driven)
            w_AMPA: 1
        '''
# I need that each neuron group has his own AMPA equation,
# each group in fact has his own s_AMPA_tot, I create a list of equations
eqs_ampa=[]

for k in range (4):
    eqs_ampa.append(eqs_ampa_base.replace('s_AMPA_tot_post','s_AMPA_tot'+str(k)+'_post'))

# This is the general nmda equation for each neuron type
eqs_nmda_base='''s_NMDA_tot_post = w_NMDA * s_NMDA : 1 (summed)
    ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha_NMDA * x * (1 - s_NMDA) : 1 (clock-driven)
    dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
    w_NMDA : 1
'''
# I need that each neuron group has his own NMDA equation,
# each group in fact has his own s_NMDA_tot, I create a list of equations
eqs_nmda=[]
for k in range (4):
    eqs_nmda.append(eqs_nmda_base.replace('s_NMDA_tot_post','s_NMDA_tot'+str(k)+'_post'))

# This is the general gaba equation for each neuron type
eqs_gaba_base='''
    s_GABA_tot_post= w_GABA* s_GABA : 1 (summed)
    ds_GABA/ dt = - s_GABA/ tau_GABA : 1 (clock-driven)
    w_GABA: 1
'''
# I need that each neuron group has his own GABA equation,
# each group in fact has his own s_GABA_tot, I create a list of equations
eqs_gaba=[]
for k in range (12):
    eqs_gaba.append(eqs_gaba_base.replace('s_GABA_tot_post','s_GABA_tot'+str(k)+'_post'))

# Eqs I need to use for connections coming from L1, I only need GABA because VIP1 is inhibitory
eqs_gaba_l1= '''s_GABA_tot12_post= w_GABA* s_GABA : 1 (summed)
    ds_GABA/ dt = - s_GABA/ tau_GABA : 1 (clock-driven)
    w_GABA: 1
'''

##### ----------------------------------------------#####
#####   Create the populations in the model         #####
##### --------------------------------------------- #####

# To compute the time to build the populations 
start_populations= time.time()

# I create the cortical column model with the groups 
# I am creating all the populations in each layer
pops=[[],[],[],[]]
for h in range(0,4):
    for z in range(0,4):

        # I create a neuron population, number of neurons and parameters differ for each group
        pop = NeuronGroup(N[h][z], model=eqs, threshold='v > Vth', reset='v = V_rest', refractory=tau_ref[h][z]*ms, method='rk4')
        
        # The values are taken from the matrices with specific values
        pop.C_m = Cm[h][z]* pF
        pop.g_m= gl[h][z]*nS
        pop.V_L = Vl[h][z] *mV
        pop.V_I= Vl[h][z] *mV
        pop.V_rest= Vr[h][z] *mV
        pop.Vth=Vt[h][z]*mV
        pop.g_AMPA_ext= 1*nS #I am using the same value for every population
        #pop.w_ext= 1 #wext[h][z] # I chose the same for everyone now, if you want a different one uncomment this 
        pop.g_AMPA_rec = 1*nS #0.95*nS #I am using the same value for every population
        pop.g_NMDA = 1*nS #0.05*nS #I am using the same value for every population
        pop.g_GABA = 1*nS #I am using the same value for every population

        pop.I_ext= 0. * pA
        pop.I_ext1= 0. * pA

        #I initialize the starting value of the membrane potential
        for k in range(0,int(N[h][z])):
            pop[k].v[0]=Vr[h][z] *mV

        pops[h].append(pop) #I append the population to the matrix with all the populations
        del (pop)


#IMPORTANT:
# pops: each row is a layer, each column a different subpopulation group (e,pv,sst,vip)
# rows: 0=layer2/3, 1=layer4, 2=layer5, 3=layer6
# columns 0=e 1=pv 2=sst 3=vip
# For example: pop[1][0] is excitatory neurons in layer 4

#I create the population in layer 1 (this is not in the previous for loop)
Vth_l1= Vt_l1*mV
Vrest_l1=Vr_l1*mV
popl1 = NeuronGroup(Nl1, model=eqs, threshold='v > Vth_l1', reset='v = Vrest_l1', refractory=tau_ref_l1*ms, method='rk4')

popl1.C_m = Cm_l1* pF
popl1.g_m= gl_l1*nS
popl1.V_L = Vl_l1 *mV
popl1.V_I = Vl_l1 *mV

popl1.g_AMPA_ext= 1*nS
#popl1.wext= 1
popl1.g_AMPA_rec = 1*nS
popl1.g_NMDA = 1*nS
popl1.g_GABA = 1*nS
popl1.I_ext= Iext_l1* pA

for k in range(0,int(Nl1)):
    popl1[k].v[0]=Vrest_l1

# Compute the time to build the populations
end_populations= time.time() 

##### ----------------------------------------------#####
#####   Create background noise to the column       #####
##### --------------------------------------------- #####

# I create a poisson generator for each neuron in the population, 
# all the neurons infact are receiving the background inputs

# Function to connect each group to the noise generator
def input_layer_connect(Num,pop,gext,nu_ext): #nu_ext must be in Hz!!
    extinput = PoissonInput(pop, "s_AMPA_ext", N=1000, rate=nu_ext / 1000, weight=gext) #External noise generator
    return extinput

#LAYER 2/3
#nu_ext=np.loadtxt('import_files/nu_ext.txt') #Is at the beginning of the code in the definitions of parameters!
#gext=1 is at the beginnig of the code in the definitions of parameters!
extinputs = []
extinputs.append(input_layer_connect(N[0][0],pops[0][0],gext,nu_ext[0][0]* Hz)) #Connecting e populations of layer 2/3 to noise
extinputs.append(input_layer_connect(N[0][1],pops[0][1],gext,nu_ext[0][1]* Hz)) #Connecting pv populations of layer 2/3 to noise
extinputs.append(input_layer_connect(N[0][2],pops[0][2],gext,nu_ext[0][2]* Hz)) #Connecting sst populations of layer 2/3 to noise
extinputs.append(input_layer_connect(N[0][3],pops[0][3],gext,nu_ext[0][3]* Hz)) #Connecting vip populations of layer 2/3 to noise

#LAYER 4
# Connecting all populations of layer 4 to noise
extinputs.append(input_layer_connect(N[1][0],pops[1][0],gext,nu_ext[1][0]* Hz))
extinputs.append(input_layer_connect(N[1][1],pops[1][1],gext,nu_ext[1][1]* Hz))
extinputs.append(input_layer_connect(N[1][2],pops[1][2],gext,nu_ext[1][2]* Hz))
extinputs.append(input_layer_connect(N[1][3],pops[1][3],gext,nu_ext[1][3]* Hz))

#LAYER 5
# Connecting all populations of layer 5 to noise
extinputs.append(input_layer_connect(N[2][0],pops[2][0],gext,nu_ext[2][0]* Hz))
extinputs.append(input_layer_connect(N[2][1],pops[2][1],gext,nu_ext[2][1]* Hz))
extinputs.append(input_layer_connect(N[2][2],pops[2][2],gext,nu_ext[2][2]* Hz))
extinputs.append(input_layer_connect(N[2][3],pops[2][3],gext,nu_ext[2][3]* Hz))

#LAYER 6
# Connecting all populations of layer 6 to noise
extinputs.append(input_layer_connect(N[3][0],pops[3][0],gext,nu_ext[3][0]* Hz))
extinputs.append(input_layer_connect(N[3][1],pops[3][1],gext,nu_ext[3][1]* Hz))
extinputs.append(input_layer_connect(N[3][2],pops[3][2],gext,nu_ext[3][2]* Hz))
extinputs.append(input_layer_connect(N[3][3],pops[3][3],gext,nu_ext[3][3]* Hz))

#Connect L1 to noise
extinputs.append(input_layer_connect(Nl1,popl1,gext,nu_extl1)) #Connecting vip1 to noise

##### ----------------------------------------------#####
#####   Function to connect the populations         #####
##### --------------------------------------------- #####

# IMPORTANT!
# THIS IS THE STRUCTURE OF THE FOLLOWING VERY LONG FUNCTION:
# Read this to understand what is happening:

# def connect_populations(sources list, targets list, flag nmda, weights matrix, propbability matrix, N matrix, populations):
#             for loop on the sources
#                 for loop on the targets
#                     activate AMPA or GABA depending on the neuron type
#                     set the parameters
#                     activate NMDA

#sources=[[layer,cell_type],[layer,cell_type]]  #sources[h][0] is the layer
                                                #sources[h][1] is the cell type

# Function to connect the populations (used for alla layers except L1)
def connect_populations(sources,targets,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):

# Percentage of different receptors
# I multiply the prob of connection by this 
# Now is passed in the function no need to have it here
    # e_ampa=0.8
    # e_nmda=0.2
    # i_ampa=0.8
    # i_nmda=0.2

    All_C=[] #I will store all the connections here

    # In the future I can have connections within same population stronger/weaker than the one between different populations
    wp_p=1  #multyply factor for connections within the same populations
    wp_m=1  #multyply factor for connections between different populations

    for h in range(len(sources)):
        for k in range(len(targets)):
            s_layer = sources[h][0] #sending layer
            s_cell_type = sources[h][1] #population type in the sending layer
            t_layer = targets[k][0] #target layer
            t_cell_type = targets[k][1] #population type in the target layer

            if s_cell_type==0: # sendind is excitatory neuron

                if t_cell_type==0: # target is excitatory neuron

                    # sending is excitaotry receiving is excitaotry then they are connected trought AMPA receptors:
                    conn= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_ampa[s_layer],on_pre='s_AMPA+=1', method='rk4') #I am connectiong 2 populations, with ampa equation
                    conn.connect(condition='i != j',p=e_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]) #how to connect the neurons in the two populations, with probability p 
                    #conn.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1)# when NMDA off use this

                    if s_layer==t_layer and s_cell_type==t_cell_type: #within the same population
                        wp=wp_p
                    else:  #between different populations
                        wp=wp_m
                    #print("Printing the connections")
                    #print(conn.N_outgoing_pre)
                    if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an AMPA that has a weight with 0 effect
                        conn.w_AMPA= 0
                    else:
                        conn.w_AMPA=wp* G*Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(e_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                    conn.delay='d'
                    All_C.append(conn) #I append the connections to the list containing all of them
                    del conn #I delete it to save memory

                    if nmda_on==True: #I need to create the NMDA connections
                        conn1= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_nmda[s_layer],on_pre='x+=1', method='rk4')
                        conn1.connect(condition='i != j',p=e_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])

                        if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an NMDA that has a weight with 0 effect
                            conn1.w_NMDA= 0
                        else:
                            conn1.w_NMDA=wp* G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(e_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                        conn1.delay='d'
                        All_C.append(conn1)
                        del conn1

                if t_cell_type!=0: # target is inhibitory neuron
                                    # Note: is the same as before but in the future if the % of AMPA is different I have already everything in place

                    conn= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_ampa[s_layer],on_pre='s_AMPA+=1', method='rk4')
                    conn.connect(condition='i != j',p=i_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                    #conn.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1)# when NMDA off use this

                    if s_layer==t_layer and s_cell_type==t_cell_type: #within the same population
                        wp=wp_p
                    else: #between different populations
                        wp=wp_m
                    #print("Printing the connections")
                    #print(conn.N_outgoing_pre)
                    if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an AMPA that has a weight with 0 effect
                        conn.w_AMPA= 0
                    else:
                        conn.w_AMPA=wp*G*Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(i_ampa*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                    conn.delay='d'
                    All_C.append(conn) #I append the connections to the list containing all of them
                    del conn #I delete it to save memory

                    if nmda_on==True: #I need to create the NMDA connections
                        conn1= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_nmda[s_layer],on_pre='x+=1', method='rk4')
                        conn1.connect(condition='i != j',p=i_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])
                        #conn1.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*1) # when I try weights instead
                        if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0: #If the probability of connections is 0 I still need to create an AMPA that has a weight with 0 effect
                            conn1.w_NMDA= 0
                        else:
                            conn1.w_NMDA=wp*G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(i_nmda*Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])

                        conn1.delay='d'
                        All_C.append(conn1) #I append the connections to the list containing all of them
                        del conn1 #I delete it to save memory

            else: # sendind is inhibitory neuron, the connections goes to GABA receptors
                conn2= Synapses(pops[s_layer][s_cell_type],pops[t_layer][t_cell_type],model=eqs_gaba[3*s_layer+s_cell_type-1],on_pre='s_GABA+=1', method='rk4')
                conn2.connect(condition='i != j',p=Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type])

                if s_layer==t_layer and s_cell_type==t_cell_type: #within the same population
                    wp=wp_p
                else: #between different populations
                    wp=wp_m

                if Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]==0:  #If the probability of connections is 0 I still need to create an GABA that has a weight with 0 effect
                    conn2.w_GABA= 0
                else:
                    conn2.w_GABA=wp*G* Cs[4*s_layer+s_cell_type][4*t_layer+t_cell_type]/(Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]*N[s_layer][s_cell_type])
#                 print(" Sending: layer=%s, type=%s"%(s_layer,s_cell_type))
#                 print(" Receiving: layer=%s, type=%s"%(t_layer,t_cell_type))
#                 print("Printing the connections sent")
#                 print(conn2.N_outgoing_pre)
#                 print("Printing the connections arriving")
#                 print(conn2.N_incoming_post)
                conn2.delay='d'
                All_C.append(conn2)
                del conn2

    return All_C


# Function to connect l1 to populations
def connect_l1to_target(targets,Gl1,Csl1,Cpl1,Nl1,pops,popl1,d):
    All_C_l1=[]
    for k in range(len(targets)):
            t_layer = targets[k][0]
            t_cell_type = targets[k][1]

            conn2= Synapses(popl1,pops[t_layer][t_cell_type],model=eqs_gaba_l1,on_pre='s_GABA+=1', method='rk4')
            conn2.connect(condition='i != j',p=Cpl1[4*t_layer+t_cell_type])

            if Csl1[4*t_layer+t_cell_type]==0 or Cpl1[4*t_layer+t_cell_type]==0:
                conn2.w_GABA= 0
            else:
                conn2.w_GABA=Gl1* Csl1[4*t_layer+t_cell_type]/(Cpl1[4*t_layer+t_cell_type]*Nl1)
            conn2.delay='d'
            All_C_l1.append(conn2)
            del conn2
    return All_C_l1

# Function to connect l1 to l1
def connect_l1_l1(Gl1,Cs_l1_l1,Cp_l1_l1,Nl1,popl1,d):
    conn2= Synapses(popl1,popl1,model=eqs_gaba_l1,on_pre='s_GABA+=1', method='rk4')
    conn2.connect(condition='i != j',p=Cp_l1_l1)
    #conn2.w_GABA= Cs_l1_l1
    conn2.w_GABA= Gl1* Cs_l1_l1/(Cp_l1_l1*Nl1)
    conn2.delay='d'
    return conn2

# Function to connect populations to l1
def connect_source_tol1(sources,Gl1,Cs_tol1,Cp_tol1,N,pops,popl1,d,i_ampa,i_nmda,nmda_on=True):
    All_C=[]
    for h in range(len(sources)):
        s_layer = sources[h][0]
        s_cell_type = sources[h][1]

        if s_cell_type==0: #0 is excitatory neuron
            conn= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_ampa[s_layer],on_pre='s_AMPA+=1', method='rk4')
            conn.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type]*i_ampa)

            #print(conn.N_outgoing_pre)
            if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]==0:
                conn.w_AMPA= 0
            else:
                conn.w_AMPA=Gl1* Cs_tol1[4*s_layer+s_cell_type]/(i_ampa*Cp_tol1[4*s_layer+s_cell_type]*N[s_layer][s_cell_type])

            conn.delay='d'
            All_C.append(conn)
            del conn

            if nmda_on==True:
                conn1= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_nmda[s_layer],on_pre='x+=1', method='rk4')
                conn1.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type]*i_nmda)
                if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]==0:
                    conn1.w_NMDA= 0
                else:
                    conn1.w_NMDA=Gl1*  Cs_tol1[4*s_layer+s_cell_type]/(i_nmda*Cp_tol1[4*s_layer+s_cell_type]*N[s_layer][s_cell_type])

                conn1.delay='d'
                All_C.append(conn1)
                del conn1
        else:
            conn2= Synapses(pops[s_layer][s_cell_type],popl1,model=eqs_gaba[3*s_layer+s_cell_type-1],on_pre='s_GABA+=1', method='rk4')
            conn2.connect(condition='i != j',p=Cp_tol1[4*s_layer+s_cell_type])

            if Cs_tol1[4*s_layer+s_cell_type]==0 or Cp_tol1[4*s_layer+s_cell_type]==0:
                conn2.w_GABA= 0
            else:
                conn2.w_GABA=Gl1* Cs_tol1[4*s_layer+s_cell_type]/(Cp_tol1[4*s_layer+s_cell_type]*N[s_layer][s_cell_type])

            conn2.delay='d'
            All_C.append(conn2)
            del conn2

    return All_C
# I have to assign to each source his own equation bijectively. This eqs_gaba[3*s_layer+s_cell_type-1]
# trasform the pair [layer][cell_type] into a number corresponding to one of the 11 gaba equations
# I want a correspondace between my matrix 4x4 (layer,cell type) and the matrix 16x16 where all the values of the connections are stored.
# This is why I need #Cp[4*s_layer+s_cell_type][4*t_layer+t_cell_type]



##### ------------------------------------------------#####
#####   Function to connect many populations at once  #####
##### ------------------------------------------------#####

#Here I build different functions to connect the layers

# CONNECTING ALL LAYERS (layer 2/3, 4, 5, 6)
def connect_all_layers(Cs,Cp,G,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):
    targets=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]
    sources=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]
    conn_all=connect_populations(sources,targets,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
    return conn_all

# CONNECTING only 2 LAYERS
def connect_layers(layer_s,layer_t,G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True):
    targets=[[[0,0],[0,1],[0,2],[0,3]],
            [[1,0],[1,1],[1,2],[1,3]],
            [[2,0],[2,1],[2,2],[2,3]],
            [[3,0],[3,1],[3,2],[3,3]]]
    sources=[[[0,0],[0,1],[0,2],[0,3]],
            [[1,0],[1,1],[1,2],[1,3]],
            [[2,0],[2,1],[2,2],[2,3]],
            [[3,0],[3,1],[3,2],[3,3]]]
    conn=connect_populations(sources[layer_s],targets[layer_t],G,Cs,Cp,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
    return conn

# Connection L1 to all populations & all to L1
def connect_l1_all(Gl1,Csl1,Cpl1,Cs_tol1,Cp_tol1,Cs_l1_l1,Cp_l1_l1,N,Nl1,pops,popl1,d,i_ampa,i_nmda,nmda_on=True):
    targets=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]
    sources=[[0,0],[0,1],[0,2],[0,3],
            [1,0],[1,1],[1,2],[1,3],
            [2,0],[2,1],[2,2],[2,3],
            [3,0],[3,1],[3,2],[3,3]]

    conn_l1_to_all=connect_l1to_target(targets,Gl1,Csl1,Cpl1,Nl1,pops,popl1,d)
    conn_all_to_l1=connect_source_tol1(sources,Gl1,Cs_tol1,Cp_tol1,N,pops,popl1,d,i_ampa,i_nmda,nmda_on=True)
    conn_l1_l1=[connect_l1_l1(Gl1,Cs_l1_l1,Cp_l1_l1,Nl1,popl1,d)]
    conn= conn_l1_to_all+ conn_all_to_l1 + conn_l1_l1
    return conn



##### ------------------------------------------------------#####
#####   Connecting all the layers by calling the functions  #####
##### ------------------------------------------------------#####

# Compute the time to connect the populations between them
start_connecting=time.time()
# print('--------------------------------------------------')
# print('Connecting layers')
# print('--------------------------------------------------')

# # I connect all the layers by calling the functions to connect
conn_all_l1=connect_l1_all(Gl1,Csl1,Cpl1,Cs_tol1,Cp_tol1,Cs_l1_l1,Cp_l1_l1,N,Nl1,pops,popl1,d,i_ampa,i_nmda,nmda_on=True)
conn_all=connect_all_layers(Cs,Cp,G,N,pops,d,e_ampa,e_nmda,i_ampa,i_nmda,nmda_on=True)
# print('All layers now connected')
end_connecting=time.time() # Compute the time it took to connect all populations
connections = conn_all + conn_all_l1
##### -----------------------------------------------------------------------------------#####
##### Create the functions for the recording devices: Spike detectors and rate detectors #####
##### -----------------------------------------------------------------------------------#####

# Function to monitor
# Spike detectors
def spike_det(pops,layer,rec=True):
    e_spikes = SpikeMonitor(pops[layer][0],record=rec)  #create the spike detector for e
    pv_spikes= SpikeMonitor(pops[layer][1],record=rec)  #create the spike detector for group pv
    sst_spikes= SpikeMonitor(pops[layer][2],record=rec) #create the spike detector for group sst
    vip_spikes= SpikeMonitor(pops[layer][3],record=rec) #create the spike detector for group vip

    return e_spikes,pv_spikes,sst_spikes,vip_spikes

# Subgroup where from each group I record only a number n_activity of neurons
def spike_det_n(pops,layer,n_activity):
    e_spikes = SpikeMonitor(pops[layer][0][:n_activity])  #create the spike detector for e
    pv_spikes= SpikeMonitor(pops[layer][1][:n_activity])  #create the spike detector for subgroup pv
    sst_spikes= SpikeMonitor(pops[layer][2][:n_activity]) #create the spike detector for subgroup sst
    vip_spikes= SpikeMonitor(pops[layer][3][:n_activity]) #create the spike detector for subgroup vip

    return e_spikes,pv_spikes,sst_spikes,vip_spikes

# Rate detectors
def rate_det(pops,layer):
    e_rate = PopulationRateMonitor(pops[layer][0]) #create the rate det for e
    pv_rate= PopulationRateMonitor(pops[layer][1]) #create the rate detector for subgroup pv
    sst_rate= PopulationRateMonitor(pops[layer][2]) #create the rate detector for subgroup sst
    vip_rate= PopulationRateMonitor(pops[layer][3])#create the rate detector for subgroup vip

    return e_rate,pv_rate,sst_rate,vip_rate

##### -----------------------------------------------------------------------------------------#####
##### Create the recording devices: Spike detectors and rate detectors (calling the functions) #####
##### -----------------------------------------------------------------------------------------#####

# To compute the time to create detectors 
start_detectors=time.time()


SM_ampa_ext = [StateMonitor(popl1, "s_AMPA_ext", record=True, dt=1*ms)] + [StateMonitor(pop, "s_AMPA_ext", record=True, dt=1*ms) for layer in pops for pop in layer]
SM_v = [StateMonitor(popl1, "v", record=True, dt=1*ms)] + [StateMonitor(pop, "v", record=True, dt=1*ms) for layer in pops for pop in layer]

spike_monitors = [SpikeMonitor(popl1[:],record=True)]
for i in range(4):
    spike_monitors.extend(spike_det(pops, i, True))

rate_monitors = [PopulationRateMonitor(popl1)]
for i in range(4):
    rate_monitors.extend(rate_det(pops, i))


# Other recorders devices (not used)
# mE=StateMonitor(pops[0][0][10], 'v',record=True) #check membrane potential of one neuron

defaultclock.dt = dt_sim*ms #time step of simulations

# Construct network
# I have to add here all the populations, inputs, connections, monitor devices
net = Network(pops[:],
              popl1,
              connections[:],
              extinputs,
              spike_monitors,
              rate_monitors,
#               SM_ampa_ext,
#               SM_v,
              )

net.store("initialized")
population_rates = []
# s_AMPA_exts = []
# vms = []
for i in range(5):
    net.restore("initialized")
    np.random.seed(i+1)
    net.run(runtime)
    population_rates.append([rm.smooth_rate(window="flat", width=1*ms).mean() for rm in rate_monitors])
#     s_AMPA_exts.append(SM_ampa_ext[1].s_AMPA_ext)
#     vms.append(SM_v[1].v)
population_rates = np.array(population_rates)
# s_AMPA_exts = np.array(s_AMPA_exts)
# vms = np.array(vms)

## 

# def plot_network(variables):
#     gs = GridSpec(ncols=4, nrows=5)
#     index = 0
#     fig = plt.figure()
#     for var in variables:
#         i1, i2 = index // 4, index % 4
#         ax = fig.add_subplot(gs[i1, i2])
#         ax.plot(var, color="black")
#         if index == 0:
#             index += 4
#         else:
#             index += 1
# 
# plot_network(population_rates)
# plt.show()


# a = np.array([0.97916667, 0.87363931, 3.81538462, 3.65957447, 7.83177570, 1.76363636, 5.51020408, 2.52830189, 0.37037037, 8.36240952, 4.34920635, 4.14285714, 5.63636364, 1.67408047, 2.93761141, 6.60784314, 3.68421053])
# 
# b = np.array([0.70833333, 0.68887908, 2.67692308, 3.40425532, 5.90654206, 1.38811881, 3.30241187, 1.54716981, 0.37037037, 2.57931542, 3.71428571, 3.71428571, 5.45454545, 0.97315195, 2.09803922, 6.03921569, 3.15789474])
# 
# c = np.array([650, 930., 1460., 870., 1405., 890., 1980., 2105., 240., 4740., 930., 530., 870., 1770., 1170., 885., 1620.])

