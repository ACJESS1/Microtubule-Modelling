#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:20:27 2023

@author: ale
"""

import matplotlib.pyplot as plt
import itertools
import hoomd
import math

import xarray as xr
import numpy
import gsd.hoomd
import sphere_fres as sf
import model_tools as mt

def patch_code(sx,sz,sy,sa,sb,sc):
    
    code = f"""

quat<float> q_rel = q_j*conj(q_i);
vec3<float> angles(0,0,0);

// roll (x-axis rotation)
//    double sinr_cosp = 2 * (q_rel.s * q_rel.v.x + q_rel.v.y * q_rel.v.z);
//    double cosr_cosp = 1 - 2 * (q_rel.v.x * q_rel.v.x + q_rel.v.y * q_rel.v.y);
//    angles.x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
//    double sinp = std::sqrt(1 + 2 * (q_rel.s * q_rel.v.y - q_rel.v.x * q_rel.v.z));
//   double cosp = std::sqrt(1 - 2 * (q_rel.s * q_rel.v.y - q_rel.v.x * q_rel.v.z));
//    angles.y = 2 * std::atan2(sinp, cosp) - M_PI / 2;

    // yaw (z-axis rotation)
//    double siny_cosp = 2 * (q_rel.s * q_rel.v.z + q_rel.v.x * q_rel.v.y);
//    double cosy_cosp = 1 - 2 * (q_rel.v.y * q_rel.v.y + q_rel.v.z * q_rel.v.z);
//    angles.z = std::atan2(siny_cosp, cosy_cosp);

const int sx={sx},sy={sy},sz={sz},sa={sa},sb={sb},sc={sc};

vec3<float> direction = rotate(conj(q_i),r_ij);

int x = round(direction.x+10);
int y = round(direction.y+10);
int z = round(direction.z+10);

//int a=0,b=0,c=0;

int a=round((-abs(angles.x)/M_PI)+1);
int b=round((-(2*abs(angles.y))/M_PI)+1) ;
int c=round((-abs(angles.z)/M_PI)+1);

const float kT = param_array[sx*sy*sz*sa*sb*sc];
//+31661.46

return (param_array[x*sy*sz*sa*sb*sc+y*sz*sa*sb*sc+z*sa*sb*sc+a*sb*sc+b*sc+c])/kT;


"""

    return code

def patch_code_map3(sx,sz,sy,sa,sb,sc):
    
    code = f"""

quat<float> q_rel = q_j*conj(q_i);
vec3<float> angles(0,0,0);

// roll (x-axis rotation)
//    double sinr_cosp = 2 * (q_rel.s * q_rel.v.x + q_rel.v.y * q_rel.v.z);
//    double cosr_cosp = 1 - 2 * (q_rel.v.x * q_rel.v.x + q_rel.v.y * q_rel.v.y);
//    angles.x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
//    double sinp = std::sqrt(1 + 2 * (q_rel.s * q_rel.v.y - q_rel.v.x * q_rel.v.z));
//   double cosp = std::sqrt(1 - 2 * (q_rel.s * q_rel.v.y - q_rel.v.x * q_rel.v.z));
//    angles.y = 2 * std::atan2(sinp, cosp) - M_PI / 2;

    // yaw (z-axis rotation)
//    double siny_cosp = 2 * (q_rel.s * q_rel.v.z + q_rel.v.x * q_rel.v.y);
//    double cosy_cosp = 1 - 2 * (q_rel.v.y * q_rel.v.y + q_rel.v.z * q_rel.v.z);
//    angles.z = std::atan2(siny_cosp, cosy_cosp);

const int sx={sx},sy={sy},sz={sz},sa={sa},sb={sb},sc={sc};

vec3<float> direction = rotate(conj(q_i),r_ij);

int x = round((direction.x)/2+5);
int y = round((direction.y)/2+5);
int z = round((direction.z)/2+5);

//int a=0,b=0,c=0;

int a=round((2*angles.x)/M_PI+2);
int b=round((4*angles.x)/M_PI+2);
int c=round((2*angles.x)/M_PI+2);

a=a%4,b=b%4,c=c%4;

const float kT = param_array[sx*sy*sz*sa*sb*sc];
//+31661.46

return (param_array[x*sy*sz*sa*sb*sc+y*sz*sa*sb*sc+z*sa*sb*sc+a*sb*sc+b*sc+c])/kT;


"""

    return code

def graphs(sim):

    
    
    data = gsd.hoomd.read_log('log.gsd')
    step = data['configuration/step']

    beta_potential_energy = data['log/hpmc/pair/user/CPPPotential/energy']
    kT = data['log/kT']
    potential_energy = beta_potential_energy * kT


    fig1 = plt.figure(1)
    ax= fig1.add_subplot()
    ax.plot(step[potential_energy<0],potential_energy[potential_energy<0])
    ax2 = ax.twinx()
    ax2.plot(step,kT,'r')



    with sim.state.cpu_local_snapshot as sn:
        #print(sn.particles.position)
        N_particles = max(sn.particles.tag)
        distlog = numpy.array([])
        neighborlist = numpy.array([])
        for i in range(N_particles):
            for j in range(i+1,N_particles):
                displacement = sn.particles.position[i,:]- sn.particles.position[j,:]
                temp_dist = math.sqrt(numpy.dot(displacement, displacement))
                distlog = numpy.append(distlog,temp_dist)
                if temp_dist <= 10:
                    neighborlist = numpy.append(neighborlist,[int(i),int(j)])
                    dist =sn.particles.position[0,:]- sn.particles.position[1,:]
                    #print(math.sqrt(numpy.dot(dist, dist)))
                    
    plt.figure(2)
    plt.hist(distlog,bins=int(numpy.max(distlog)*0.8))
                    
    pticks = numpy.arange(0,max(distlog),10)
    plt.xticks(pticks)
    plt.xlabel('Distance / 10^(-10)m')
    
    
    neighborlist=numpy.reshape(neighborlist,(int(numpy.size(neighborlist)/2),2) ) 
                    
                    
    neighborlist[int(numpy.size(neighborlist)/2)-1,:]
                    
                    
                    
    with sim.state.cpu_local_snapshot as sn:
        angle_distlog = numpy.array([])
        for i in range(0,int(numpy.size(neighborlist)/2)):
            P1 = int(neighborlist[i,0])
            P2 = int(neighborlist[i,1])
            Q1 = numpy.array(sn.particles.orientation[P1,:])
            Q2 = numpy.array(sn.particles.orientation[P2,:])
            in_prod = numpy.sum(Q1*Q2)
            Angle_dist = math.acos( 2*(in_prod)**2-1)
            angle_distlog = numpy.append(angle_distlog,Angle_dist)

        

    plt.figure(23)
    plt.hist(angle_distlog*(180/math.pi),bins=12)

    import quat_mult as qm

    with sim.state.cpu_local_snapshot as sn:
        #angle_distlog = numpy.array([])
        indexlog = numpy.array([])
        for i in range(0,int(numpy.size(neighborlist)/2)):
            P1 = int(neighborlist[i,0])
            P2 = int(neighborlist[i,1])
            Q1 = numpy.array(sn.particles.orientation[P1,:])
            Q2 = numpy.array(sn.particles.orientation[P2,:])
            in_prod = numpy.sum(Q1*Q2)

            Q1_conj = numpy.array([Q1[0], -Q1[1],-Q1[2],-Q1[3]])
        
            Q_rel = qm.quaternion_multiply(Q2,Q1_conj)
        
            angles=numpy.array([0.,1.,2.])

            sinr_cosp = 2 * (Q_rel[0] * Q_rel[1] + Q_rel[2] * Q_rel[3]);
            cosr_cosp = 1 - 2 * (Q_rel[1] * Q_rel[1] + Q_rel[2] * Q_rel[2]);
            angles[0] = math.atan2(sinr_cosp, cosr_cosp);

            sinp = math.sqrt(1 + 2 * (Q_rel[0] * Q_rel[2] - Q_rel[1] * Q_rel[3]));
            cosp = math.sqrt(1 - 2 * (Q_rel[0] * Q_rel[2] - Q_rel[1] * Q_rel[3]));
            angles[1] = 2 * math.atan2(sinp, cosp) - math.pi / 2;
            
            
            siny_cosp = 2 * (Q_rel[0] * Q_rel[3] + Q_rel[1] * Q_rel[2]);
            cosy_cosp = 1 - 2 * (Q_rel[2] * Q_rel[2] + Q_rel[3] * Q_rel[3]);
            angles[2] = math.atan2(siny_cosp, cosy_cosp);
            
            a=round((-abs(angles[0])/math.pi)+1);
            b=round((-(2*abs(angles[1]))/math.pi)+1) ;
            c=round((-abs(angles[2])/math.pi)+1);
        
            indexlog=numpy.append(indexlog,[a*4+b*2+c])

    plt.figure(4)
    plt.hist(indexlog,bins=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])

    neighborlist.astype(int)

    #create triplets
    triplet=numpy.array([])
    for i in range(int(numpy.size(neighborlist)/2)):
            for j in range(i+1,int(numpy.size(neighborlist)/2)):
                tpair1 = numpy.array(neighborlist[i,:])
                tpair2 = numpy.array(neighborlist[j,:])
                temp_triplet = numpy.unique( numpy.concatenate((tpair1,tpair2 )) )
                if numpy.size(temp_triplet) == 3:
                    triplet=numpy.append(triplet, temp_triplet)


    triplet = numpy.reshape(triplet,(int(numpy.size(triplet)/3),3) ) 
    triplet = numpy.unique(triplet,axis=0) #remove double counted triplets
    with sim.state.cpu_local_snapshot as sn:
        v_angle_log = numpy.array([])
        for i in range(0,int(numpy.size(triplet)/3)):
            I1 = int(triplet[i,0])
            I2 = int(triplet[i,1])
            I3 = int(triplet[i,2])
            Pos1 = numpy.array(sn.particles.position[I1,:])
            Pos2 = numpy.array(sn.particles.position[I2,:])
            Pos3 = numpy.array(sn.particles.position[I3,:])
            V1 = Pos1-Pos2
            V2 = Pos2-Pos3
            V3 = Pos3-Pos1
            sz1 = math.sqrt(abs(numpy.dot(V1,V1)))
            sz2 = math.sqrt(abs(numpy.dot(V2,V2)))
            sz3 = math.sqrt(abs(numpy.dot(V3,V3)))
            if sz1>sz2 and sz1>sz3:
                v_angle_log = numpy.append( v_angle_log , math.acos(numpy.dot(V2,V3)/(sz2*sz3)) )
            elif sz2>sz1 and sz2>sz3:
                v_angle_log = numpy.append( v_angle_log , math.acos(numpy.dot(V1,V3)/(sz1*sz3)) )
            elif sz3>sz2 and sz3>sz1:
                v_angle_log = numpy.append( v_angle_log , math.acos(numpy.dot(V1,V2)/(sz1*sz2)) )

    plt.figure(5)
    plt.hist(v_angle_log*(180/math.pi),bins=20)
    #Spike at 2 radians indicates equilateral triangles
    return

def get_data(file):
    
    #The terrible things done to the data set

    ds_disk = xr.open_dataarray(file)

    sx = len(ds_disk.x) 
    sy = len(ds_disk.y)
    sz = len(ds_disk.z) 
    sa = len(ds_disk.a)
    sb = len(ds_disk.b) 
    sc = len(ds_disk.c)

    tarrt = numpy.reshape(ds_disk.data,(sx*sz*sy*sa*sb*sc))
    
    return sx,sy,sz,sa,sb,sc,tarrt