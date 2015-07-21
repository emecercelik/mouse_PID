import matplotlib.pyplot as plt
import numpy as np
import pickle

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

fps=50
aaa=GetPickle('inpOut')

length=aaa.shape[0]-3
inp_len=length/3

inp=aaa[0:inp_len]
out=aaa[inp_len:int(2*inp_len)]
cont_inp=aaa[int(2*inp_len):length]

jointName=aaa[-3]
PP=float(aaa[-2])
speed=float(aaa[-1])

time=np.arange(0,inp_len/fps,1/fps)

plt.axis([0,(2*np.pi)*7/speed,-0.1,1.1])

plt.plot(time,inp,'bo')
plt.plot(time,inp,'b',label='input servo position')
plt.plot(time,out,'go')
plt.plot(time,out,'g',label='actual servo position')
plt.plot(time,cont_inp,'ro')
plt.plot(time,cont_inp,'r',label='control input')

plt.xlabel('Time (sec)')
plt.ylabel('Position of Servos [0,1]')
plt.legend()
plt.title('{0:s} while P={1:.0f} and speed={2:.0f}'.format(jointName,PP,speed))
plt.savefig('input_output_servo.png')
plt.show()
