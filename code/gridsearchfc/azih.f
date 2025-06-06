      subroutine azih(v,d,nl,x,y,z,xs,ys,nstaz,depi,ih,tt,az)
      implicit none
      integer l,nl,nstaz,nst,nsrc
      real x,y,z,xs(nstaz),ys(nstaz),ih(nstaz)
      real depi(nstaz),tt(nstaz),az(nstaz),zak(1)
      real pi,v(nl),d(nl)
      real ihs(1,1),tvt(1,1),distm(1,1),phir
cf2py intent(in) v,d,nl,x,y,z,xs,ys,nstaz
cf2py intent(out) depi,ih,tt,az
cf2py depend(nstaz) depi,tt,az
cf2py depend(nl) v,d
      pi=3.141592654

      nst=1
      nsrc=1

      do l=1,nstaz
         depi(l)=sqrt((x-xs(l))**2.+(y-ys(l))**2.) 
         zak(1)=z
         distm(1,1)=depi(l)
         call takeoff(V,D,nl,nst,nsrc,zak,distm,ihs,tvt)
         ih(l)=ihs(1,1)
         tt(l)=tvt(1,1)
         call strikes(xs(l),ys(l),x,y,0.0,phir)
         az(l)=phir*180./pi
      enddo

      return
      end
