# =================================================================================================================================================
#                                       Import modules
##import sys
##sys.path.append('/home/ercelik/opt1/nest/lib/python3.4/site-packages/')
##import nest
import pickle
import random
import numpy as np
from numpy import linalg as LA

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def ActFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def NeuronFunc(x):
    return (1.+np.tanh(x-1.6))*.5 # 4.=9 input

def InputFunc(x,dim,nAct,maxAcc,minAcc):
    rangeAcc=maxAcc-minAcc
    off=np.arange(minAcc+rangeAcc/(2.*nAct),maxAcc,rangeAcc/nAct)
    res=np.array([[ActFunc(nAct,maxAcc,minAcc,x[j],off[i]) for i in range(len(off))] for j in range(len(x))])
    res=res.reshape(res.size,1)
    for i in range(res.size):
        if res[i]<1e-2:
            res[i]=.0
    return res


bpy.context.scene.game_settings.fps=60.
dt=1000./bpy.context.scene.game_settings.fps


#nest.sli_func('synapsedict info')
# =================================================================================================================================================
#                                       Creating muscles


#~ servo_ids = {}
#~ servo_ids["forearm.L"] = setVelocityServo(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxV = 10.0)
PP=110.

servo_ids = {}
servo_ids["wrist.L"]      = setPositionServo(reference_object_name = "obj_wrist.L",      attached_object_name = "obj_forearm.L", P = PP)
servo_ids["wrist.R"]      = setPositionServo(reference_object_name = "obj_wrist.R",      attached_object_name = "obj_forearm.R", P = PP)
servo_ids["forearm.L"]    = setPositionServo(reference_object_name = "obj_forearm.L",    attached_object_name = "obj_upper_arm.L", P = PP)
servo_ids["forearm.R"]    = setPositionServo(reference_object_name = "obj_forearm.R",    attached_object_name = "obj_upper_arm.R", P = PP)

servo_ids["upper_arm.L"]  = setPositionServo(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L", P = PP)
servo_ids["upper_arm.R"]  = setPositionServo(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R", P = PP)
servo_ids["shin_lower.L"] = setPositionServo(reference_object_name = "obj_shin_lower.L", attached_object_name = "obj_shin.L", P = PP)
servo_ids["shin_lower.R"] = setPositionServo(reference_object_name = "obj_shin_lower.R", attached_object_name = "obj_shin.R", P = PP)

servo_ids["shin.L"]       = setPositionServo(reference_object_name = "obj_shin.L",       attached_object_name = "obj_thigh.L", P = PP)
servo_ids["shin.R"]       = setPositionServo(reference_object_name = "obj_shin.R",       attached_object_name = "obj_thigh.R", P = PP)
servo_ids["thigh.L"]       = setPositionServo(reference_object_name = "obj_thigh.L",     attached_object_name = "obj_hips", P = PP)
servo_ids["thigh.R"]       = setPositionServo(reference_object_name = "obj_thigh.R",     attached_object_name = "obj_hips", P = PP)


# =================================================================================================================================================
#                                       Network creation

#np.random.seed(np.random.randint(0,10000))

ax_avg=0.1 #Global parameter to calculate avg acceleration
ay_avg=0.1
az_avg=0.1

ax=.1 #Global parameter to record acceleration
ay=.1
az=.1

# PID Parameters
nJoint=12 #number of joints that is controlled
kP=np.array([.1 for i in range(nJoint)]) #coefficients of PID controller
kI=np.array([.03 for i in range(nJoint)])
kD=np.array([.1 for i in range(nJoint)])

e_old=np.array([0. for i in range(nJoint)]) #The error one step before
E=np.array([0. for i in range(nJoint)]) #Integrated error
uu=np.array([0. for i in range(nJoint)]) # Control input initialization

inpp=[] # Record inputs of servos to be plotted
outp=[] # Record outputs of servos to be plotted 
cont_inp=[] # Reecord control inputs to servos to be plotted

# Joints
#Joint names
joints=["wrist.L","wrist.R","forearm.L","forearm.R","upper_arm.L","upper_arm.R",\
        "shin_lower.L","shin_lower.R","shin.L","shin.R","thigh.L","thigh.R"]
numRec=3 # The number of joint to be plotted

# =================================================================================================================================================
#                                       Evolve function
def evolve():
    # Global variable definitions
    global ax,ay,az,ax_avg,ay_avg,az_avg
    global inpp,outp,PP
    global kP,kI,kD,e_old,E,uu
    global joints,nJoint,numRec
    
    print("Step:", i_bl, "  Time:{0:.2f}".format(t_bl),'   Acc:{0:8.2f}  {1:8.2f}  {2:8.2f}'.format(ax,ay,az),\
          'Acc:{0:8.2f} {1:8.2f} {2:8.2f}'.format(ax_avg,ay_avg,az_avg))
    # ------------------------------------- Visual ------------------------------------------------------------------------------------------------
    #visual_array     = getVisual(camera_name = "Meye", max_dimensions = [256,256])
    #scipy.misc.imsave("test_"+('%05d' % (i_bl+1))+".png", visual_array)
    # ------------------------------------- Olfactory ---------------------------------------------------------------------------------------------
    olfactory_array  = getOlfactory(olfactory_object_name = "obj_nose", receptor_names = ["smell1", "plastic1"])
    # ------------------------------------- Taste -------------------------------------------------------------------------------------------------
    taste_array      = getTaste(    taste_object_name =     "obj_mouth", receptor_names = ["smell1", "plastic1"], distance_to_object = 1.0)
    # ------------------------------------- Vestibular --------------------------------------------------------------------------------------------
    vestibular_array = getVestibular(vestibular_object_name = "obj_head")
    #print (vestibular_array)
    # ------------------------------------- Sensory -----------------------------------------------------------------------------------------------
    # ------------------------------------- Proprioception ----------------------------------------------------------------------------------------
    #~ spindle_FLEX = getMuscleSpindle(control_id = muscle_ids["forearm.L_FLEX"])
    #~ spindle_EXT  = getMuscleSpindle(control_id = muscle_ids["forearm.L_EXT"])
    # ------------------------------------- Neural Simulation -------------------------------------------------------------------------------------
    
    
    ax=vestibular_array[3] # Get instant accelerations
    ay=vestibular_array[4]
    az=vestibular_array[5]

    ax_avg=(ax_avg*(i_bl)+ax)/(i_bl+1) # Calculate avg accelerations
    ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
    az_avg=(az_avg*(i_bl)+az)/(i_bl+1)


    # ------------------------------------- Muscle Activation -------------------------------------------------------------------------------------
    
    speed_ = 6.0 # Speed of the mouse (ang. freq of joint patterns)

    # Joint signals to be applied
    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
    anti_act_tmp    = 1.0 - act_tmp
    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
    anti_act_tmp_p1 = 1.0 - act_tmp_p1
    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
    anti_act_tmp_p2 = 1.0 - act_tmp_p2

    # Reference value of joints
    r=np.array([0.4,0.4,0.8*act_tmp,0.8*anti_act_tmp,1.0*act_tmp_p1,1.0*anti_act_tmp_p1,0.8*anti_act_tmp,\
                0.8*act_tmp,0.5*anti_act_tmp_p1,0.5*act_tmp_p1,0.5*anti_act_tmp,0.5*act_tmp])

    if i_bl<=0:
        for i in range(nJoint):
            controlActivity(control_id = servo_ids[joints[i]], control_activity = r[i])
            # Apply the reference at the first step to the servos
    else:
        for i in range(nJoint):
            controlActivity(control_id = servo_ids[joints[i]], control_activity = uu[i])
            # Apply the control inputs to the servos 

    # Get actual positions of joints after the control input
    positions=np.array([getMuscleSpindle(control_id = servo_ids[joints[i]])[0] for i in range(len(joints))])
    

    e=r-positions # Calculateinstant errors for all joints
    E=E+e # Integrate errors for all joints
    uu=kP*e+kI*E+kD*(e-e_old) # Calculate control inputs for all joints
    e_old=e # Replace old error
    
    
    inpp.append(r[numRec]) # Record reference
    outp.append(positions[numRec]) # record positions of servos
    cont_inp.append(uu[numRec]) # Reecord control inputs

    if i_bl==750:
        data=np.hstack((np.array(inpp),np.array(outp),np.array(cont_inp),joints[numRec],PP,speed_))
        PickleIt(data,'inpOut') # Save the data to be plotted
