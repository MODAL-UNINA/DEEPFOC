      subroutine strikes(xs,ys,xc,yc,phi,phir)
      implicit none
      real xc,yc,xs,ys,ang,r,phiep,pi,phir,phis
      real x,y,x1,x2,y1,y2,num,den,phi,phisr,phi1
cf2py intent(in) :: xs,ys,xc,yc,phi
cf2py intent(out) :: phir
      pi=3.141592654
 
      phiep=phi*pi/180.
 
      phis=atan2(xs-xc,ys-yc)
 
      phisr=phis-pi*(-1+sign(1.0,phis))
 
      phir=phisr-phiep
 
      if(phir.lt.0.) phir=2*pi+phir
      if(phir.gt.2*pi) phir=phir-2*pi
 
      return
      end
 