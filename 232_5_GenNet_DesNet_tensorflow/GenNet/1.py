#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:01:45 2018

@author: alice
"""

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()


### 1/4 for all available directions, 1-j/4 for stop
k = 0
M=int(1e7)
for i in range(M):
    state = [0,0]
    path = []
    n = 0
    while (state >= [0,0]) and (state <= [10,10]) and state not in path:
        # np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        last = np.copy(state)
        path.append(list(last))
        move_d = np.random.choice([0,1], 1)
        move_l = np.random.choice([-1,1], 1)
        state[move_d[0]] += move_l[0]
        n += 1
    end = path[n-1]
    r = (end[0]+1) <= 0 or (end[0]+1) >= 10 or [end[0]+1,end[1]] in path
    l = (end[0]-1) <= 0 or (end[0]-1) >= 10 or [end[0]-1,end[1]] in path
    u = (end[1]+1) <= 0 or (end[1]+1) >= 10 or [end[0],end[1]+1] in path
    d = (end[1]-1) <= 0 or (end[1]-1) >= 10 or [end[0],end[1]-1] in path
    p = int(r)+int(l)+int(u)+int(d)/4
    k += 1/((1/4)**(n-1)*p)
k/M

M = int(1e10)
M1 = int(1e8)
eps = 0.1

def saw2(M, eps, M1):
    k_all = 0
    k_end = 0
    n = 0
    n_end = 0
    cache_all = []
    cache_end = []
    for i in range(M):
        state = np.array([0,0])
        path = []
        p = 1
        r, l, u, d = np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])
        while all(state >= 0) and all(state <= 10) and list(state) not in path:
            last = np.copy(state)
            path.append(list(last))        
            moves = [np.array([0,0])]
            if all(state+r >= 0) and all(state+r <= 10) and list(state+r) not in path:
                moves.append(r)
            if all(state+l >= 0) and all(state+l <= 10) and list(state+l) not in path:
                moves.append(l)
            if all(state+u >= 0) and all(state+u <= 10) and list(state+u) not in path:
                moves.append(u)
            if all(state+d >= 0) and all(state+d <= 10) and list(state+d) not in path:
                moves.append(d)
            if len(moves) == 1:
                break
            else:
                prob = [eps] + list(np.repeat((1-eps)/(len(moves)-1),len(moves)-1))
                idx = np.random.choice(len(moves), 1, p = prob)
                move = moves[idx[0]]
                p_n = prob[idx[0]]
            if all(move == np.array([0,0])):
                break
            state += move
            p = p*p_n
            
        if i < M1:
            k_all += 1/p
            n += 1
        if all(state == np.array([10,10])):
            k_end += 1/p
            n_end += 1
            if np.log10(n_end) in list(range(1,int(np.log10(M))+1)):
                cache_end.append(k_end/n_end)
                print("Sample[%.3e] Sample_end[%5d] K_all: %.3e K_end: %.3e" % (n, n_end, k_all/n, k_end/n_end))
        if np.log10(n) in list(range(1,int(np.log10(M))+1)): 
            cache_all.append(k_all/n) 
            if n_end>0:
                print("Sample[%.0e] Sample_end[%5d] K_all: %.3e K_end: %.3e" % (n, n_end, k_all/n, k_end/n_end))
    
    return k_all/n, k_end/n_end, cache_all, cache_end

M = int(1e10)
M1 = int(1e8)
eps = 0.1
saw2(M, eps, M1)



def saw(i, eps, k_all, k_end, n, n_end, cache_all, cache_end):
    state = np.array([0,0])
    path = []
    p = 1
    r, l, u, d = np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])
    while all(state >= 0) and all(state <= 10) and list(state) not in path:
        last = np.copy(state)
        path.append(list(last))        
        moves = [np.array([0,0])]
        if all(state+r >= 0) and all(state+r <= 10) and list(state+r) not in path:
            moves.append(r)
        if all(state+l >= 0) and all(state+l <= 10) and list(state+l) not in path:
            moves.append(l)
        if all(state+u >= 0) and all(state+u <= 10) and list(state+u) not in path:
            moves.append(u)
        if all(state+d >= 0) and all(state+d <= 10) and list(state+d) not in path:
            moves.append(d)
        if len(moves) == 1:
            break
        else:
            prob = [eps] + list(np.repeat((1-eps)/(len(moves)-1),len(moves)-1))
            idx = np.random.choice(len(moves), 1, p = prob)
            move = moves[idx[0]]
            p_n = prob[idx[0]]
        if all(move == np.array([0,0])):
            break
        state += move
        p = p*p_n

    if i < M1-1:
        k_all += 1/p
        n += 1
    if all(state == np.array([10,10])):
        k_end += 1/p
        n_end += 1
        if np.log10(n_end) in list(range(1,int(np.log10(M))+1)):
            cache_end.append(k_end/n_end)
            print("Sample[%.3e] Sample_end[%5d] K_all: %.3e K_end: %.3e" % (n, n_end, k_all/n, k_end/n_end))
    if np.log10(n) in list(range(1,int(np.log10(M))+1)): 
        cache_all.append(k_all/n) 
        if n_end>0:
            print("Sample[%.0e] Sample_end[%5d] K_all: %.3e K_end: %.3e" % (n, n_end, k_all/n, k_end/n_end))
    return k_all, k_end, n, n_end, cache_all, cache_end

M = int(1e15)
M1 = int(1e10)
eps = 0.1
k_all = 0
k_end = 0
n = 0
n_end = 0
cache_all = []
cache_end = []
for i in range(M):
    k_all, k_end, n, n_end, cache_all, cache_end = saw(i, eps, k_all, k_end, n, n_end, cache_all, cache_end)


def saw(i):
    state = np.array([0,0])
    path = []
    p = 1
    end = False
    r, l, u, d = np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])
    while list(state) not in path:
        last = np.copy(state)
        path.append(list(last))        
        moves = []
        if all(state+r >= 0) and all(state+r <= 10) and list(state+r) not in path:
            moves.append(r)
        if all(state+l >= 0) and all(state+l <= 10) and list(state+l) not in path:
            moves.append(l)
        if all(state+u >= 0) and all(state+u <= 10) and list(state+u) not in path:
            moves.append(u)
        if all(state+d >= 0) and all(state+d <= 10) and list(state+d) not in path:
            moves.append(d)
        d = len(moves)
        if d == 0:
            break
        else:
            prob = np.repeat(1/d,d)
            idx = np.random.choice(d, 1, p = prob)
            move = moves[idx[0]]
            p_n = prob[idx[0]]
        state += move
        p = p*p_n
        if all(move == np.array([0,0])):
            break
        if all(state == np.array([10,10])):
            end = True
    design.append((end, p))
    s = len(design)
    if np.mod(s,1e3) == 0: 
        K_all = np.mean([1/x[1] for x in design])
        ws_end = np.array([1/x[1] for x in design if x[0]])
        temp = np.argwhere(design[0])
        us = np.append(temp[0],np.diff(temp))
        ps_end = ws_end/us
        K_end = np.mean(ps_end)
        print("Design Sample[%.1e] Sample_end[%5d] K_all: %.3e K_end: %.3e" % (s, len(ps_end), K_all, K_end))

design = []
M = int(1e7)
# pool = ThreadPool()
pool.map(saw, np.repeat(eps, M))
pool.wait_completion()

### Design2: eps to stop
def saw2(eps):
    state = np.array([0,0])
    path = []
    p = 1
    end = False
    r, l, u, d = np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])
    while list(state) not in path:
        last = np.copy(state)
        path.append(list(last))        
        moves = [np.array([0,0])]
        if all(state+r >= 0) and all(state+r <= 10) and list(state+r) not in path:
            moves.append(r)
        if all(state+l >= 0) and all(state+l <= 10) and list(state+l) not in path:
            moves.append(l)
        if all(state+u >= 0) and all(state+u <= 10) and list(state+u) not in path:
            moves.append(u)
        if all(state+d >= 0) and all(state+d <= 10) and list(state+d) not in path:
            moves.append(d)
        if len(moves) == 1:
            break
        else:
            prob = [eps] + list(np.repeat((1-eps)/(len(moves)-1),len(moves)-1))
            idx = np.random.choice(len(moves), 1, p = prob)
            move = moves[idx[0]]
            p_n = prob[idx[0]]
        state += move
        p = p*p_n
        if all(move == np.array([0,0])):
            break
        if all(state == np.array([10,10])):
            end = True
    design2.append((end, p))
    s = len(design2)
    if np.mod(s,1e5) == 0: 
        K_all = np.mean([1/x[1] for x in design2])
        ws_end = np.array([1/x[1] for x in design2 if x[0]])
        temp = np.argwhere(design2[0])
        us = np.append(temp[0],np.diff(temp))
        ps_end = ws_end/us
        K_end = np.mean(ps_end)
        print("Design2 Sample[%.1e] Sample_end[%5d] K_all: %.3e K_end: %.3e" % (s, len(ps_end), K_all, K_end))
        
# design2 = []
eps = 0.1
M = int(1e7)
# pool = ThreadPool()
pool.map(saw2, np.repeat(eps, M))
pool.wait_completion()

def saw1(i):
    state = np.array([0,0])
    path = []
    p = 1
    end = False
    length = 0
    r, l, u, d = np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])
    while list(state) not in path:
        last = np.copy(state)
        path.append(list(last))        
        moves = []
        if all(state+r >= 0) and all(state+r <= 10) and list(state+r) not in path:
            moves.append(r)
        if all(state+l >= 0) and all(state+l <= 10) and list(state+l) not in path:
            moves.append(l)
        if all(state+u >= 0) and all(state+u <= 10) and list(state+u) not in path:
            moves.append(u)
        if all(state+d >= 0) and all(state+d <= 10) and list(state+d) not in path:
            moves.append(d)
        d = len(moves)
        if d == 0:
            break
        else:
            prob = np.repeat(1/d,d)
            idx = np.random.choice(d, 1, p = prob)
            move = moves[idx[0]]
            p_n = prob[idx[0]]
            length += 1
        state += move
        p = p*p_n
        if all(state == np.array([10,10])):
            end = True
    design1.append((end, p, length))
    s = len(design1)
#    if np.log10(len(design1)) in list(range(1,int(np.log10(M))+1)) or np.mod(len(design1),1e5) == 0: 
    if np.mod(s,1e5) == 0: 
        K_all1 = np.mean([1/x[1] for x in design1])
        ws_end = np.array([1/x[1] for x in design1 if x[0]])
        temp = np.argwhere(design1[0])
        us = np.append(temp[0],np.diff(temp))
        ps_end = ws_end/us
        K_end1 = np.mean(ps_end)
        print("Sample[%.1e] Sample_end[%5d] Design1: %.3e, %.3e" % (s, len(ps_end), K_all1, K_end1))

# design1 = []
M = int(1e7)
i = 1

pool = ThreadPool(50)
pool.map(saw1, np.repeat(i, M))
pool.wait_completion()


K_all1 = np.array([1/x[1] for x in design1])
ws_end = np.array([1/x[1] for x in design1 if x[0]])*eps
temp = np.argwhere(design1[0])
us = np.append(temp[0],np.diff(temp))
ps_end = ws_end/us
K_end1 = np.mean(ps_end)

K_all2 = np.mean([0.9**x[2]/x[1] for x in design1])
ws_end = np.array([0.9**x[2]/x[1] for x in design1 if x[0]])
temp = np.argwhere(design1[0])
us = np.append(temp[0],np.diff(temp))
ps_end = ws_end/us
K_end2 = np.mean(ps_end)

ws = np.array([0.9**x[2]/x[1] for x in design1])
ws[x[3]>50] = 
K_all3 = np.mean()
ws_end = np.array([0.9**x[2]/x[1] for x in design1 if x[0]])
temp = np.argwhere(design1[0])
us = np.append(temp[0],np.diff(temp))
ps_end = ws_end/us
K_end3 = np.mean(ps_end)

def saw(i):
    state = np.array([0,0])
    path = []
    p = 1
    length = 0
    end = False
    r, l, u, d = np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])
    while all(state >= 0) and all(state <= 10) and state not in path:
        last = np.copy(state)
        path.append(last)        
        moves = []
        if all(state+r >= 0) and all(state+r <= 10) and (state+r) not in path:
            moves.append(r)
        if all(state+l >= 0) and all(state+l <= 10) and (state+l) not in path:
            moves.append(l)
        if all(state+u >= 0) and all(state+u <= 10) and (state+u) not in path:
            moves.append(u)
        if all(state+d >= 0) and all(state+d <= 10) and (state+d) not in path:
            moves.append(d)
        if len(moves) == 0:
            break
        else:
            directs = len(moves)
            pr = 1/directs
            idx = np.random.choice(directs, 1, p = np.repeat(pr,directs))
            move = moves[idx[0]]
            length += 1
        state += move
        p = p*pr
        if all(state == np.array([10,10])):
            end = True
    design1.append((end, p, length))
#    if np.log10(len(design1)) in list(range(1,int(np.log10(M))+1)) or np.mod(len(design1),1e5) == 0: 
    if np.mod(len(design1),1e5) == 0: 
        K_all1 = np.mean([1/x[1] for x in design1])
        ws_end = np.array([1/x[1] for x in design1 if x[0]])
        temp = np.argwhere(design1[0])
        us = np.append(temp[0],np.diff(temp))
        ps_end = ws_end/us
        K_end1 = np.mean(ps_end)
        K_all2 = np.mean([1/x[1]/0.99**x[2] for x in design1])
        ws_end = np.array([1/x[1]/0.99**x[2] for x in design1 if x[0]])
        temp = np.argwhere(design1[0])
        us = np.append(temp[0],np.diff(temp))
        ps_end = ws_end/us
        K_end2 = np.mean(ps_end)
        print("Sample[%.1e] Sample_end[%5d] Design1: %.3e, %.3e Design2: %.3e, %.3e" % (len(design1), len(ps_end), K_all1, K_end1, K_all2, K_end2))

pool.map(saw, np.repeat(i, M))
pool.wait_completion()
saw(1)