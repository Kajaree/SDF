# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:10:21 2019

@author: kajaree das
"""
import numpy as np
import matplotlib.pyplot as plt

dt = 0.5
sigma = 100
I = np.diag([1., 1.])
O = np.diag([0., 0.])
H = np.block([I, O, O])
R = sigma**2*I

def plotter(measurements, title):
    '''
    input: measurements, title 
    output: plot of the measurement
    '''
        plt.rcParams['legend.fontsize'] = 10
        plt.plot(measurements[:, 0], measurements[:, 1])
        plt.title(title)
        plt.savefig(title, bbox_inches='tight')
        plt.show()

class KalmanFilter(object):
    def __init__(self, F=None, Q=None, R=None, P0=None, x0=None):
        self.n = F.shape[1]
        self.F = F
        self.Q = np.diag([1., 1., 1., 1., 1., 1.])
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P0 is None else P0
        self.x = np.zeros((self.n, 1)) if x0 is None else x0.T
        self.i = 0
        self.x = np.reshape(self.x, (6,1))
        
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        
    def update(self, z):
        v = np.subtract(np.reshape(z, (2,1)), np.dot(H, self.x))
        S = np.dot(H, np.dot(self.P, H.T)) + R
        W = np.dot(self.P, np.dot(H.T, np.linalg.inv(S))) 
        self.x = np.add(self.x, np.dot(W, v))
        self.P -= np.dot(W, np.dot(S, W.T))   
            
class Measurement():
    def __init__(self, sigma=10):
        self.R = R
        self.mean = [0, 0]
        
    def getNoise(self):
        return np.random.multivariate_normal(self.mean, self.R)
    
    

class Sensor(object):
    def __init__(self, position, target, dt = 0.1/60, sigma=1.0):
        self.position = position
        self.measurements = None
        self.target = target
        self.sigma = sigma
        self.R = sigma**2*I
        self.filteredMeasurement = None 
        self.covariances = None
        self.x = None
        self.P = None
        
    def getMeasurements(self):
        measurements = []
        measure = Measurement(self.sigma)
        for track in self.target.tracks:
            t = np.array([track[0], track[1]])
            u = measure.getNoise()
            measurement = np.add(t, u)
            measurement = np.subtract(measurement, self.position)
            measurements.append(measurement)
        self.measurements = np.array(measurements)
        
    def filterMeasurements(self):
        x0 = np.block([np.array([self.measurements[0][0], self.measurements[0][1]]).T,np.array([0, 0]).T,np.array([0, 0]).T]).T
        self.target.getP()
        P0 = self.target.P0
        self.target.x0 = x0
        sigmak = (self.target.vmax/self.target.qmax)/3.0
        F = np.matrix([[1.0, 0.0, dt, 0.0, 1/2.0*dt**2, 0.0],
                        [0.0, 1.0, 0.0,  dt, 0.0, 1/2.0*dt**2 ],
                        [0.0, 0.0, 1.0, 0.0,  dt, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0,  dt],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        Q = sigmak**2*np.block([[dt**4*I/4, dt**3*I/2, dt**2*I/2],[dt**3*I/2, dt**2*I, dt*I],[dt**2*I/2, dt*I, I]])
        kf = KalmanFilter(F, Q, self.R, P0, x0)
        self.covariances = np.zeros((len(self.measurements), F.shape[1], F.shape[1]))
        self.filteredMeasurement = np.zeros((len(self.measurements), F.shape[1], 1))
        for i, measurement in enumerate(self.measurements):
            kf.predict()
            kf.update(measurement)
            self.filteredMeasurement[i, :] = kf.x
            self.P = kf.P
            self.covariances[i, :, :] = self.P
        
    def plotMeasurements(self, title):
        plotter(self.measurements, title)
        
    def plotFilteredMeasurements(self, title):
        plotter(self.filteredMeasurement, title)
    
    
class Target(object):
    def __init__(self, vin, qin, timePeriod):
        self.x0 = np.array([0., 0., 0., 0., 0., 0.]).T
        self.v0 = vin
        self.q0 = qin
        self.T = timePeriod
        self.state = None
        self.P0 = None
        self.tracks = None
        self.vmax = 0
        self.qmax = 0
    
    def getAngularFreq(self):
        return float(self.q0)/(2*self.v0)

    def getAmplitude(self):
        return float(self.v0**2)/self.q0
    
    def getMaxVelocity(self):
        vx = self.tracks[:, 2]
        vy = self.tracks[:, 3]
        v = np.sqrt(np.add(np.square(vx), np.square(vy)))
        return np.max(v)
    
    def getMaxAccelaration(self):
        ax = self.tracks[:, 4]
        ay = self.tracks[:, 5]
        a = np.sqrt(np.add(np.square(ax), np.square(ay)))
        return np.max(a)
    
    def getP(self):
        vmax = self.getMaxVelocity()
        qmax = self.getMaxVelocity()
        self.P0 = np.vstack([[ sigma**2 ,     0    ,    0    ,    0    ,    0      ,     0   ],
                             [    0     , sigma**2 ,    0    ,    0    ,    0      ,     0   ],
                             [    0     ,    0     , vmax**2 ,    0    ,    0      ,     0   ],
                             [    0     ,    0     ,    0    , vmax**2 ,    0      ,     0   ],
                             [    0     ,    0     ,    0    ,    0    ,  qmax**2  ,     0   ],
                             [    0     ,    0     ,    0    ,    0    ,    0      ,  qmax**2]])
        self.vmax = vmax
        self.qmax = qmax
    def trajectory_tracker(self):
        A = self.getAmplitude()
        w = self.getAngularFreq()
        track = []
        track.append(np.block([np.array([0, 0]).T,np.array([self.v0, self.v0]).T,np.array([self.q0, self.q0]).T]).T)
        for t in self.T:
            rx = A*np.sin(w*t)
            ry = A*np.sin(2*w*t)
            vx = self.v0 * np.cos(w*t)/2
            vy = self.v0* np.cos(2*w*t)
            ax = (-1./4)*self.q0*np.sin(w*t)
            ay = (-1)*self.q0*np.sin(2*w*t)
            x = np.block([np.array([rx, ry]).T,np.array([vx, vy]).T,np.array([ax, ay]).T]).T
            track.append(x)
        self.tracks = np.array(track)
        
    def plot_trajectory(self, title='Ground Truth'):
        plotter(self.tracks, title)
        
class FusionCenter(object):
    def __init__(self, n, target):
        self.numSensors = n
        self.sensors = []
        self.target = target
        self.P = None
        self.x = None
        self.estimates = None
        self.covariances = None
        
    def initiateSensors(self):
        for i in range(self.numSensors):
            position = Measurement().getNoise()
            sensor = Sensor(position, self.target)
            sensor.getMeasurements()
            self.sensors.append(sensor)
            sensor.filterMeasurements()
            
    def naiveFusion(self):
        self.covariances = np.zeros((len(self.sensors[0].measurements), 6, 6))
        self.estimates = np.zeros((len(self.sensors[0].measurements), 6, 1))
        for i in range(len(self.sensors[0].covariances)):
            tempP = np.zeros((6, 6))
            tempX = np.zeros((6,1))
            for j in range(len(self.sensors)):
                tempP = np.add(tempP, np.linalg.inv(self.sensors[j].covariances[i, :, :]))
                tempX = np.add(tempX, np.dot(np.linalg.inv(self.sensors[j].covariances[i, :, :]), self.sensors[j].filteredMeasurement[i, :, :]))
            self.P = np.linalg.inv(tempP)
            self.covariances[i, :, :] = np.linalg.inv(tempP)
            self.estimates[i, :, :] = np.dot(self.covariances[i, :, :], tempX)
            
            
    def trackletFusion(self):
        self.P = self.target.P0
        self.x = self.target.x0
        self.x = np.reshape(self.x, (6,1))
        self.covariances = np.zeros((len(self.sensors[0].measurements), 6, 6))
        self.estimates = np.zeros((len(self.sensors[0].measurements), 6, 1))
        self.covariances[0, :, :] = self.P
        self.estimates[0, :, :] = self.x
        for i in range(1, len(self.sensors[0].covariances)):
            sumIks = np.zeros((6, 6))
            sumiks = np.zeros((6, 1))
            tempP = np.linalg.pinv(self.P)
            tempX = np.dot(tempP, self.x)
            for j in range(len(self.sensors)):
                Iks = np.subtract(np.linalg.inv(self.sensors[j].covariances[i, :, :]), np.linalg.inv(self.sensors[j].covariances[i-1, :, :]))
                iks = np.subtract(np.dot(np.linalg.inv(self.sensors[j].covariances[i, :, :]), self.sensors[j].filteredMeasurement[i, :]), 
                                  np.dot(np.linalg.inv(self.sensors[j].covariances[i-1, :, :]), self.sensors[j].filteredMeasurement[i-1, :]))
                sumIks += Iks
                sumiks += iks
            tempP += sumIks
            tempX += sumiks
            self.P = np.linalg.pinv(tempP)
            self.x = np.dot(self.P, tempX)
            self.covariances[i, :, :] = self.P
            self.estimates[i, :, :] = self.x
            
            
span = np.arange(0.,1000., dt)
target = Target(300, 9, span)
target.trajectory_tracker()
target.plot_trajectory()
fs = FusionCenter(4, target)
fs.initiateSensors()
for i in range(fs.numSensors):
    fs.sensors[i].plotMeasurements('Measurement by Sensor - {}'.format(i+1))
    fs.sensors[i].plotFilteredMeasurements('Filtered measurement')
fs.naiveFusion()
plotter(fs.estimates, 'Naive Fusion Number of Sensors = {}'.format(fs.numSensors))
fs.trackletFusion()
plotter(fs.estimates, 'Tracklet Fusion Number of Sensors = {}'.format(fs.numSensors))

