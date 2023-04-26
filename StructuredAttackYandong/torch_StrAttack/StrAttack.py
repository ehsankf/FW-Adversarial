## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
## modified by Kaidi Xu <xu.kaid@husky.neu.edu> for ICLR 2019 paper:
## 'Structured Adversarial Attack: Towards General Implementation and Better Interpretability'
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import time
import sys
import numpy as np
import torch
# from multiprocessing.pool import ThreadPool


BINARY_SEARCH_STEPS = 8  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-3  # larger values converge faster to less accurate results 1e-2 for MNIST, 1e-3 for cifar and imagenet
TARGETED = False  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1  # the initial constant c to pick as a first guess
RO = 15
RETRAIN = False

# pool = ThreadPool()

import pdb

class LADMMSTL2:
    def __init__(self, model, batch_size, image_size, num_channels, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY, initial_const = INITIAL_CONST,print_every = 100,
                 ro=RO, retrain=RETRAIN):
        """
        The L_2 optimized attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        self.model = model
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.ro = ro
        self.retrain = retrain
        self.model = model

    def compare(self, x, y):
        if not isinstance(x, (float, int, np.int64)):
            x = np.copy(x)
            if self.TARGETED:
                x[y] -= self.CONFIDENCE
            else:
                x[y] += self.CONFIDENCE
            x = np.argmax(x)
        if self.TARGETED:
            return x == y
        else:
            return x != y

    def grad(self, imgs, labs, tz, const):

        batch_size = self.batch_size
        shape = (batch_size, self.image_size, self.image_size, self.num_channels)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.model = self.model.to(device)

        timg = torch.from_numpy(imgs).to(dtype=torch.float32, device=device).requires_grad_(True)
        tlab = torch.from_numpy(labs).to(dtype=torch.float32, device=device).requires_grad_(True)
        tz = torch.from_numpy(tz).to(dtype=torch.float32, device=device).requires_grad_(True)
        const = torch.from_numpy(const).to(dtype=torch.float32, device=device).requires_grad_(True)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        newimg = tz + timg
        l2dist_real = torch.norm(tz, dim=(1, 2, 3), p=2)
        output = self.model(newimg)

        real = torch.sum(tlab * output, dim=1)
        other, _ = torch.max((1 - tlab) * output - (tlab * 10000), dim=1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1, _ = torch.max(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.CONFIDENCE, 0.0)

        loss1 = torch.sum(loss1)
        loss1.backward()
        gradtz = tz.grad
        # print("Linf norm: ", torch.norm(tz, dim=(1,2,3), p=float('inf')))
        l2s, scores, nimg, z_grads = l2dist_real.detach().cpu().numpy(), output.detach().cpu().numpy(),\
                                        newimg.detach().cpu().numpy(), gradtz.detach().cpu().numpy()

        return l2s, scores, nimg, np.array(z_grads)

        def doit(imgs, labs, z, CONST):

            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            sess.run(setup, {assign_timg: batch, assign_tlab: batchlab, assign_tz: z, assign_const: CONST, })

            l2s, scores, nimg, z_grads = sess.run([l2dist_real, output, newimg, gradtz])

            return l2s, scores, nimg, np.array(z_grads)

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        rv = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            r1, r2 = self.attack_batch(imgs[i:i + self.batch_size], targets[i:i + self.batch_size])
            r.extend(r1)
            rv = np.append(rv, r2)
        rv = rv.reshape([-1,3])
        rv = np.mean(rv, axis = 0)

        print("\nnone zeros group:", rv[0], "\nl2 mean:", rv[1], "\nli mean", rv[2], "\n")
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(imgs) # [np.zeros(imgs[0].shape)] * batch_size
        o_besty = np.ones(imgs.shape)
        
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * INITIAL_CONST # 1 for imgnet
        upper_bound = np.ones(batch_size)*1e10

        """
        alpha = 5
        tau = 3e-4
        gamma = 2 #0
        """
        """
        alpha = 5
        tau = 3
        gamma = 2
        """
        alpha = 5
        tau = 8
        gamma = 2

        if self.image_size>32: #imagenet
            filterSize = 13
            stride = 13
        else: # cifar mnist
            filterSize = 2
            stride = 2
        print('grid size:', filterSize)
        n = self.image_size * self.image_size * self.num_channels

        P = np.floor((self.image_size - filterSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P
        
        z = 0.0 * np.ones(imgs.shape)
        v = 0.0 * np.ones(imgs.shape)
        u = 0.0 * np.ones(imgs.shape)
        s = 0.0 * np.ones(imgs.shape)
        ep = 0.30

        index = np.ones([P*Q,filterSize * filterSize * self.num_channels],dtype=int)
        
        tmpidx = 0
        for q in range(Q):
            # plus = 0
            plus1 = q * stride * self.image_size * self.num_channels
            for p in range(P):
                index_ = np.array([], dtype=int)
                #index2_ = np.array([], dtype=int)
                for i in range(filterSize):
                    index_ = np.append(index_,
                                      np.arange(p * stride * self.num_channels + i * self.image_size * self.num_channels + plus1,
                                                p * stride * self.num_channels + i * self.image_size * self.num_channels + plus1 + filterSize * self.num_channels,
                                                dtype=int))
                index[tmpidx] = index_
                tmpidx += 1
        index = np.tile(index, (batch_size,1,1))

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # print(outer_step, o_bestl2, CONST)
            
            #prev = 1e6
            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size
            
            z = 0.0 * np.ones(imgs.shape)
            # z1 = 0.0*np.ones(imgs.shape)
            v = 0.0 * np.ones(imgs.shape)
            u = 0.0 * np.ones(imgs.shape)
            s = 0.0 * np.ones(imgs.shape)
            
            for iteration in range(self.MAX_ITERATIONS + outer_step * 100 ):
                if iteration % 2000 == -1:
                    print(iteration, 'best l2square:', o_bestl2, 'when l0:', np.count_nonzero((np.array(o_bestattack) - imgs).reshape([batch_size,-1]), axis = 1))
    
                # delta step
                # l2
                delt = self.ro / (self.ro + 2 * gamma) * (z - u)
                # l1
                #tmp = z - u - gamma / self.ro
                #tmp = np.where(tmp > 0, tmp, 0)
                #tmp1 = u - z - gamma / self.ro
                #tmp1 = np.where(tmp1 > 0, tmp1, 0)
                #delt = tmp - tmp1
                # w step
                temp = z - s
                temp1 = np.where(temp > np.minimum(1.0 - imgs, ep), np.minimum(1.0 - imgs, ep), temp)
                w = np.where(temp1 < np.maximum(0.0 - imgs, -ep), np.maximum(0.0 - imgs, -ep), temp1)
    
                # y step
                
                y0 = (z - v).reshape(batch_size,-1)
                #y0 = (z - v)
                
                y = y0[:]
    
                #timestart = time.time()
                #@jit 
                def findIndx(b):
                #for b in range(batch_size):
                    tmpc = tau / self.ro
                    
                    y0Ds = np.take(y0[b], index[b])
                    y0Dns = np.linalg.norm(y0Ds, axis=1)
                    #print(np.mean(y0Dns[y0Dns != 0]))
                    tmpy = np.zeros_like(y0Dns)
                    tmpy[y0Dns != 0] = 1 - tmpc / y0Dns[y0Dns != 0]
                    tmpy_ = np.zeros_like(y0Ds)
                    tmpy = np.transpose(np.tile(tmpy, [y0Ds.shape[1],1]))
                    tmpy_[tmpy > 0] = tmpy[tmpy > 0]  * y0Ds[tmpy > 0]           
                    #tmpy_[tmpy > 0] = np.transpose(np.tile(tmpy[tmpy > 0], [y0Ds.shape[1],1])) * y0Ds[tmpy > 0]
                    np.put(y[b], index[b], tmpy_)

       
                list(map(findIndx, range(batch_size)))

                y = y.reshape(imgs.shape)
                
                # z step
                l2s, scores, nimg, z_grads = self.grad(imgs, labs, z, CONST)
    

                Sc = y + v
    
                #eta = 1
                eta = 1/np.sqrt(iteration+1)
                z = 1 / (alpha / eta + 2 * self.ro + 2*self.ro) * \
                    (alpha / eta * z + 2*self.ro * (delt + u) + self.ro * (w + s) + self.ro * Sc - z_grads)
                # print(Sc.mean(),w.mean(),y.mean(),delt.mean(),z.mean())
    
                u = u + delt - z
                
                v = v + y - z
    
                s = s + w - z
    
                #yt = yt.reshape(imgs.shape)
                #np.count_nonzero(o_besty)/batch_size
                l2s, scores, nimg, y_grads = self.grad(imgs, labs, y, CONST)
    
                for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                        #print("change", e, o_bestl2[e] - l2)
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc) 
                        o_bestattack[e] = ii
                        o_besty[e] = y[e]
                    
            for e in range(batch_size):
                if self.compare(bestscore[e], np.argmax(labs[e])) and bestscore[e] != -1 and bestl2[e] == o_bestl2[e]:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 5
    
        print('Finally', o_bestl2)
        # np.save("img",o_besty[8].squeeze())
        if self.retrain:
            lower_bound = np.zeros(batch_size)
            CONST = np.ones(batch_size) * 5 # 5 for imgnet
            upper_bound = np.ones(batch_size)*1e10
            for tmpi in range(8):
                print("retrain C:", CONST)
                bestl2 = [1e10]*batch_size
                bestscore = [-1]*batch_size
                
                Nz = o_besty[np.nonzero(o_besty)]
                Nz = np.abs(Nz)
                e0 = np.percentile(Nz, 3)
                #e0 = 0
                # e0 = 0.00001
                #randm = -1 + 2*np.random.random((o_besty.shape))
                #z1 =  np.where(np.abs(o_besty) <= e0, 0, randm)
                A2 = np.where(np.abs(o_besty) <= e0, 0, 1)
                #randm = -1 + 2*np.random.random((o_besty.shape))
                z1 = o_besty
                u1 = 0.0 * np.ones(imgs.shape)
                tmpC = self.ro / (self.ro + gamma/100)
                for outer_step in range(400):
                    if outer_step % 200 == 0:
                        print("retrain", tmpi, outer_step, o_bestl2)
                    
                    tempA = (z1 - u1) * tmpC
                    tempA1 = np.where(np.abs(o_besty) <= e0, 0, tempA)
                    tempA2 = np.where(np.logical_and(tempA > np.minimum(1.0 - imgs, ep), (np.abs(o_besty) > e0)),
                                      np.minimum(1.0 - imgs, ep), tempA1)
                    deltA = np.where(np.logical_and(tempA < np.maximum(0.0 - imgs, -ep), (np.abs(o_besty) > e0)),
                                     np.maximum(0.0 - imgs, -ep), tempA2)
                    l2s, scores, nimg, z_grads = self.grad(imgs, labs, deltA, CONST)
                    z1 = 1 / (alpha + 2 * self.ro) * (alpha * z1 + self.ro * (deltA + u1) - np.multiply(z_grads[0],A2))
    
                    u1 = u1 + deltA - z1
    
                    #l2s, scores, nimg, z_grads = self.grad(imgs, labs, deltA)
                    for e, (l2, sc, ii,) in enumerate(zip(l2s, scores, nimg)):
                        if l2 < bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                            bestl2[e] = l2
                            bestscore[e] = np.argmax(sc)
                        if l2 < o_bestl2[e] and self.compare(sc, np.argmax(labs[e])):
                            o_bestl2[e] = l2
                            o_bestscore[e] = np.argmax(sc)
                            o_bestattack[e] = ii
                            o_besty[e] = deltA[e]
                            
                for e in range(batch_size):
                    if self.compare(bestscore[e], np.argmax(labs[e])) and bestscore[e] != -1:
                        # success, divide const by two
                        upper_bound[e] = min(upper_bound[e],CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        # failure, either multiply by 10 if no solution found yet
                        #          or do binary search with the known upper bound
                        lower_bound[e] = max(lower_bound[e],CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e])/2
                        else:
                            CONST[e] *= 5
                                   
        rVector = [0, 0, 0]             
        resultl2 = np.array([])  
        resultli = np.array([])  
        o_besty = o_besty.reshape(batch_size, -1)
        for b in (range(batch_size)):
            for k in range(index.shape[1]):
                ry0D = np.take(o_besty[b], index[b,k])
                ry0D2 = np.linalg.norm(ry0D)
                if ry0D2 != 0:
                    resultl2 = np.append(resultl2, ry0D2)
                    resultli = np.append(resultli, np.max(np.abs(ry0D)))
        
        rVector[0] =  len(resultl2)/batch_size
        rVector[1] =  np.mean(resultl2)
        rVector[2] =  np.mean(resultli)
        
        print("ro", self.ro, "gamma", gamma, "tau", tau, "alpha", alpha)
        print("\ntotal groups:", P*Q)
        return o_bestattack, rVector

