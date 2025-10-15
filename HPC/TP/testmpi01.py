from mpi4py import MPI
import numpy as np
import sys

#1. Communication point à point
def exchange_message():
    comm = MPI.COMM_WORLD
    nprocc=comm.Get_size()

    my_rank = comm.Get_rank()
    if my_rank !=nprocc-1:
        message=100+my_rank
        comm.send(message,dest=my_rank+1)
    if my_rank!=0:
        rece=comm.recv(source=my_rank-1)
        print(f" I am {my_rank} and my message from {my_rank-1} is {rece}")

# 2. Communication collective

def pick(n):

    count_inside = 0
    for i in range(n):
        x, y = np.random.random(2) * 2 - 1
    if x**2 + y**2 <= 1: count_inside += 1
    return count_inside

def par_picks(n):
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    size=comm.Get_size()
    nlocal=int(n/size)
    localcount=pick(nlocal)
    count=localcount
    if rank!=0:
        comm.send(localcount,dest=0)
    if rank==0:
        for irank in range(1,size):
            rcount=comm.recv(source=irank)
            count+=rcount
    return count

if __name__=="__main__":

   exchange_message()
   n=int(sys.argv[1])
   print(par_picks(n))