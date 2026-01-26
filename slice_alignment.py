import argparse
import torch
# torch.concatenate=torch.cat # compatibility
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import join,basename,splitext
from scipy.interpolate import interpn
from pathlib import Path

def main():
    """
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument('fnames', nargs = '*', type = str, help = 'List of file names in alignment order')
    parser.add_argument('outdir', type = Path, help = 'The directory where all intermediate and final outputs should be stored')
    parser.add_argument('--npad', default = 0, type = int, help = 'Default = 0; The scalar with which to pad the first and last image in the sequence')
    parser.add_argument('--addPads', action = 'store_true', help = 'If present, pad the first and last image in the sequence using \'npad\'')
    parser.add_argument('--device', default = 'cpu', help = 'Default - cpu; The device where PyTorch computations should occur during registration')
    parser.add_argument('--dtype', default = torch.float32, help = 'Default - torch.float32; The dtype to be used during PyTorch computation')

    args = parser.parse_args()
    fnames = args.fnames
    outdir = args.outdir
    npad = args.npad
    addPads = args.addPads
    device = args.device
    dtype = args.dtype

    # ===============================================
    # ===== (0) Load and clean the input images =====
    # ===============================================

    # Load the files in sequential order
    for fname in fnames:
        Ji = plt.imread(fname)
    
        if Ji.dtype == np.uint8:
            Ji = Ji / 255.0
        if Ji.shape[-1] == 4:
            Ji = Ji[...,:3]
        
        # find any rows or colums that are all ones
        if Ji.ndim == 2:
            Ji = Ji[...,None].repeat(3,axis=-1)
        rowones = np.all(Ji>=0.95,(0,-1))
        colones = np.all(Ji>=0.95,(1,-1))
        Wi = (1-rowones[None,:])*(1-colones[:,None])
        
        J_.append(Ji)
        W_.append(Wi)
    
    # Interpolate every image onto a grid of the same size
    nJ = [Ji.shape for Ji in J_]
    nJ = np.max(nJ,0)
    nJ = [len(J_),nJ[0],nJ[1]]
    down = 1
    x2d = [np.arange(n)*down - (n-1)*down/2 for n in nJ[1:]]
    X2d = np.stack(np.meshgrid(*x2d,indexing='ij'),-1)
    fig,ax = plt.subplots()
    hfig = display(fig,display_id=True)
    J__ = []
    W__ = []
    for Ji,Wi in zip(J_,W_):
        x = [np.arange(n)*down - (n-1)*down/2 for n in Ji.shape[:2]]
        Ji_ = interpn(x,Ji,X2d,bounds_error=False,method='nearest')
        Wi_ = interpn(x,Wi,X2d,bounds_error=False,method='nearest')
        
        Wi_ = (1.0 - np.isnan(Ji_[...,0]))*Wi_
        Ji_[np.isnan(Ji_)] = 0
        Wi_[np.isnan(Wi_)] = 0
        J__.append(Ji_)
        W__.append(Wi_)
        ax.cla()
        ax.imshow(Ji_)
        hfig.update(fig)
    
    # Note our convention is to use 
    J = np.stack(J__,0).transpose(-1,0,1,2)
    W = np.stack(W__)
    xJ = [np.arange(nJ[0])-(nJ[0]-1)/2,x2d[0],x2d[1]]

    # Optionally, pad the first and last slices
    if addPads:
        npad = 0
        J = np.pad(J,((0,0),(npad,npad),(0,0),(0,0)),mode='reflect')
        W = np.pad(W,((npad,npad),(0,0),(0,0)),mode='reflect')
        nJ = J.shape[1:]
        xJ = [np.arange(nJ[0])-(nJ[0]-1)/2,x2d[0],x2d[1]]

    # Convert data to pytorch objects
    J = torch.tensor(J,dtype=dtype,device=device)
    W = torch.tensor(W,dtype=dtype,device=device)
    xJ = [torch.tensor(x,dtype=dtype,device=device) for x in xJ]

    # ===================================================================
    # ===== (1) Generate necessary data structures for registration =====
    # ===================================================================
    # In order to apply a linear transform, we need to generate a sequence of 2D affine matrices
    XJ = torch.stack(torch.meshgrid(xJ,indexing='ij'),-1)
    A = torch.eye(3)
    A[0,0] = 1.1
    A[1,1] = 0.9
    A = A[None].repeat(nJ[0],1,1)
    A = A2DtoA3D(A)
    Ai = torch.linalg.inv(A)
    Xs = AX(Ai,XJ)
    AJ = interp(xJ,J,XJ)

    fig,ax = plt.subplots()
    ax.imshow(J[:,AJ.shape[1]//2].permute(1,2,0))
    plt.savefig(os.path.join(outdir, 'fig0.png'))

    fig,ax = plt.subplots()
    ax.imshow(AJ[:,AJ.shape[1]//2].permute(1,2,0))
    plt.savefig(os.path.join(outdir, 'fig1.png'))

    AJ = interp(xJ,J,Xs)
    fig,ax = plt.subplots()
    ax.imshow(AJ[:,AJ.shape[1]//2].permute(1,2,0))
    plt.savefig(os.path.join(outdir, 'fig2.png'))

    fig,ax = plt.subplots()
    ax.imshow(AJ[:,:,AJ.shape[2]//2].permute(1,2,0),aspect='auto')
    plt.savefig(os.path.join(outdir, 'fig3.png'))

    # get a set of sample points for v
    extendv = 1.1 # i.e. make it 10% bigger than the domain of J, to avoid wraparound
    dv = down*2
    vmin1 = torch.amin(xJ[1])
    vmin2 = torch.amin(xJ[2])
    vmax1 = torch.amax(xJ[1])
    vmax2 = torch.amax(xJ[2])
    vc1 = (vmin1 + vmax1)/2
    vc2 = (vmin2 + vmax2)/2
    vr1 = (vmax1-vmin1)/2*extendv
    vr2 = (vmax2-vmin2)/2*extendv
    v1 = torch.arange(vc1-vr1,vc1+vr1,dv,device=device,dtype=dtype)
    v2 = torch.arange(vc2-vr2,vc2+vr2,dv,device=device,dtype=dtype)
    xv = [xJ[0],v1,v2]
    XV = torch.stack( torch.meshgrid(*xv,indexing='ij') , -1)
    XV2d = XV[...,1:]
    v2d = torch.zeros_like(XV2d) 
    v2d = torch.randn(v2d.shape,dtype=v2d.dtype)

    # get highpass and lowpass operators for 2d reg
    a = 10.0
    p = 2.0
    L = L_from_xv_a_p(xv,a,p)
    LL = L**2
    K = 1.0/LL
    
    fig,ax = plt.subplots()
    ax.imshow(L)
    plt.savefig(os.path.join(outdir, 'fig4_L.png'))

    fig,ax = plt.subplots()
    ax.imshow(K)
    plt.savefig(os.path.join(outdir, 'fig5_K.png'))
    
    v2d = torch.fft.ifftn( torch.fft.fftn(v2d,dim=(1,2))*K[...,None] , dim=(1,2),).real
    v3d = v2DToV3D(v2d)
    v3d /= torch.std(v3d)
    v3d *= 20
    v = v2d # for later

    phi = exp(xv,v3d)
    fig,ax = plt.subplots()
    ax.contour(xv[2],xv[1],phi[phi.shape[0]//2,...,1])
    ax.contour(xv[2],xv[1],phi[phi.shape[0]//2,...,2])
    ax.set_title('Example deformation')
    plt.savefig(os.path.join(outdir, 'fig6_exdef.png'))

    # initial guess
    I = (torch.sum(J*W,1,keepdims=True)/(1e-6 + torch.sum(W,0,keepdims=True))).repeat(1,J.shape[1],1,1)
    xI = [x.clone() for x in xJ]
    fig,ax = plt.subplots()
    ax.imshow(I[:,I.shape[1]//2].permute(1,2,0))
    plt.savefig(os.path.join(outdir, 'fig7.png'))

    fig,ax = plt.subplots()
    ax.imshow(I[:,:,I.shape[2]//2].permute(1,2,0),aspect='auto')
    plt.savefig(os.path.join(outdir, 'fig8.png'))

    # transform an image with phi
    phiI = interp(xI,I,phi)
    fig,ax = plt.subplots()
    ax.imshow(phiI[:,AJ.shape[1]//2].permute(1,2,0))
    plt.savefig(os.path.join(outdir, 'fig9.png'))
    
    fig,ax = plt.subplots()
    ax.imshow(phiI[:,:,phiI.shape[2]//2].permute(1,2,0),aspect='auto')
    plt.savefig(os.path.join(outdir, 'fig10.png'))

    # Get the jacobian weights
    Wdetjac = detjac(xv,vnew)
    fig,ax = plt.subplots()
    mappable = ax.imshow(Wdetjac[Wdetjac.shape[0]//2])
    plt.colorbar(mappable)
    plt.savefig(os.path.join(outdir, 'fig11.png'))

    fig,ax = plt.subplots()
    ax.imshow(W[W.shape[0]//2],interpolation='none')
    plt.savefig(os.path.join(outdir, 'fig12.png'))

    RphiI = transform_image(xI,I,xv,vnew,Anew,xJ)
    fig,ax = plt.subplots()
    ax.imshow(RphiI[:,:,RphiI.shape[2]//2].permute(1,2,0),aspect='auto',interpolation='none')
    plt.savefig(os.path.join(outdir, 'fig13.png'))

    L,WR = robust_loss(RphiI,xJ,J,W,c, return_weights=True)
    fig,ax = plt.subplots()
    ax.imshow(WR[WR.shape[0]//2])
    plt.savefig(os.path.join(outdir, 'fig14.png'))

    # get phiiRiJ
    phiiRiJ = inverse_transform_image(xJ,J,xv,vnew,Anew,xI,padding_mode='border')
    phiiRiW = inverse_transform_image(xJ,W[None]*WR,xv,vnew,Anew,xI,padding_mode='zeros',mode='nearest')[0]
    XI = torch.stack(torch.meshgrid(xI,indexing='ij'),-1)
    Wdetjacs = interp(xv,Wdetjac[None],XI)[0]

    fig,ax = plt.subplots()
    ax.imshow((phiiRiJ)[:,I.shape[1]//2].permute(1,2,0))
    plt.savefig(os.path.join(outdir, 'fig15.png'))

    fig,ax = plt.subplots()
    ax.imshow((phiiRiW)[I.shape[1]//2])
    plt.savefig(os.path.join(outdir, 'fig16.png'))

    Inew = I.clone()

    asquare = 2.0**2
    anisotropy_factor = 1.0
    Inew,Eat,ERat = atlas_from_aligned_slices_and_weights(xI,Inew*0,phiiRiJ,phiiRiW*Wdetjacs,asquare,niter=2,draw=True,anisotropy_factor=anisotropy_factor)

    fig,ax = plt.subplots()
    ax.imshow(Inew[:,I.shape[1]//2].permute(1,2,0))
    plt.savefig(os.path.join(outdir, 'fig17.png'))

    fig,ax = plt.subplots()
    ax.imshow(Inew[:,:,I.shape[2]//2].permute(1,2,0),aspect='auto')
    plt.savefig(os.path.join(outdir, 'fig18.png'))

    # let's test the FWHM
    asquare = 3.5**2
    anisotropy_factor = 0.3**2
    fwhm = atlas_from_aligned_slices_and_weights(xI,Inew*0,phiiRiJ,phiiRiW*Wdetjacs,asquare,niter=2,draw=True,return_fwhm=True,anisotropy_factor=anisotropy_factor)

    # ====================================
    # ===== (2) Perform registration =====
    # ====================================

    # this cell should be the main function
    niter_big_loop = 400
    niter_reg = 5
    niter_atlas = 5
    asquare = 0.25**2
    asquare0=0.25**2
    asquare0 = 3.5**2
    anisotropy_factor = 0.1**2
    
    epT = 1e-2*2*2*0
    epL = 1e-6*2*2*0
    epv = 1e1*2
    epv = 1e3
    epv = 1e2*5
    c = 1.0 # bigger c means less robustness
    c = 2.0
    
    sigmaM = 1.0 # this should always be 1
    sigmaR = 1e5 # should be smaller
    sigmaR = 1e4
    sigmaR = 2e2
    sigmaR = 5e2
    a = 8.0 # was 10
    a = 6.0
    
    
    fig_at,ax_at = plt.subplots(2,3)
    ax_at = ax_at.ravel()
    hfig_at = display(fig_at,display_id=True)
    
    fig_at_estimate = plt.figure()
    hfig_at_estimate = display(fig_at_estimate,display_id=True)
    fig_reg = plt.figure()
    hfig_reg = display(fig_reg,display_id=True)
    fig_E,ax_E = plt.subplots(1,1)
    if isinstance(ax_E,np.ndarray):
        ax_E = ax_E.ravel()
    else:
        ax_E = [ax_E]
    hfig_E = display(fig_E,display_id=True)
    # we want xI bigger than xJ so ther are no boundary issues
    bigger = 20
    x2dI = [torch.arange(n+bigger,dtype=dtype)*down - (n+bigger-1)*down/2 for n in nJ[1:]]
    xI = [torch.arange(nJ[0],dtype=dtype)-(nJ[0]-1)/2,x2dI[0],x2dI[1]]
    XI = torch.stack(torch.meshgrid(xI,indexing='ij'),-1)
    
    # this is the loss we want to report, not WSEE loss
    Esave = []
    v = torch.zeros_like(v)
    A = torch.eye(3)
    A = A[None].repeat(nJ[0],1,1)
    
    
    # initialize with mean
    I = torch.zeros((J.shape[0],XI.shape[0],XI.shape[1],XI.shape[2])) + (torch.sum(J*W,dim=(1,2,3))/torch.sum(W,dim=(0,1,2)))[...,None,None,None]    
    # first get the loss and the weights, using current guesses
    RphiI = transform_image(xI,I,xv,v,A,xJ)    
    rloss, W_robust_loss = robust_loss(RphiI,xJ,J,W,c, return_weights=True)    
    
    for it_big_loop in range(niter_big_loop):
        
        if it_big_loop == 0:
            asquare = 4.0**2*asquare0
        elif it_big_loop == 20:
            asquare = 2.0**2*asquare0
        elif it_big_loop == 40:
            asquare = 1.0**2*asquare0
        asquare = asquare0
        
        
        # now register, but wait until I've estimated a reasonable atlas
        if it_big_loop > 0:
            A,v,Eregistration,Ereg = weighted_see_registration(xI,I,xJ,J,W*W_robust_loss,xv,v,A,a,p,sigmaM,sigmaR,niter_reg,epT,epL,epv,draw=5,fig=fig_reg,hfig=hfig_reg)
        else:
            Ereg = 0.0
        # and we want to add Ereg to our loss
        
        # now get jacobians
        Wdetjac = detjac(xv,v)
        
        # now update atlas
        phiiRiJ = inverse_transform_image(xJ,J,xv,v,A,xI,padding_mode='border',mode='nearest')
        phiiRiW = inverse_transform_image(xJ,W[None]*W_robust_loss,xv,v,A,xI,padding_mode='zeros',mode='nearest')[0]    
        Wdetjacs = interp(xv,Wdetjac[None],XI)[0]
        I,Eat,ERat = atlas_from_aligned_slices_and_weights(xI,I,phiiRiJ,phiiRiW*Wdetjacs,asquare,niter=niter_atlas,fig=fig_at_estimate,hfig=hfig_at_estimate,draw=True,anisotropy_factor=anisotropy_factor)
        # and we want to add ERat to the loss
        
        # get the loss and the weights, using current guesses
        RphiI = transform_image(xI,I,xv,v,A,xJ)    
        rloss, W_robust_loss = robust_loss(RphiI,xJ,J,W,c, return_weights=True)    
        # this is the loss we want to report, it's the loss with the current parameters
        
        ax_at[0].cla()
        ax_at[0].imshow(I[:,I.shape[1]//2].permute(1,2,0))
        ax_at[1].cla()
        ax_at[1].imshow(I[:,:,I.shape[2]//2].permute(1,2,0),aspect='auto',interpolation='none')
        ax_at[2].cla()
        ax_at[2].imshow(I[:,:,:,I.shape[3]//2].permute(1,2,0),aspect='auto',interpolation='none')
        
        
        Wshow = (phiiRiW*Wdetjacs)
        ax_at[3].cla()
        ax_at[3].imshow(Wshow[I.shape[1]//2])
        ax_at[4].cla()
        ax_at[4].imshow(Wshow[:,I.shape[2]//2],aspect='auto',interpolation='none')
        ax_at[5].cla()
        ax_at[5].imshow(Wshow[:,:,I.shape[3]//2],aspect='auto',interpolation='none')
        
        
        # is this the right error? yes I think so
        Esave.append([rloss.item()+Ereg+ERat,rloss.item(),Ereg,ERat])
        ax_E[0].cla()
        ax_E[0].plot(Esave)
        ax_E[0].legend(['total', 'robust matching', 'registration reg', 'atlas reg'])
        
        hfig_at.update(fig_at)
        hfig_E.update(fig_E)

        fig_at_estimate.savefig(os.path.join(outdir,f'atlas_{it_big_loop:06d}.png'))
    
    return