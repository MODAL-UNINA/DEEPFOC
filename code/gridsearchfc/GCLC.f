      SUBROUTINE GCLC(ZLATDEG,ZLONDEG,XLATDEG,YLONDEG,XP,YP)
      implicit none
      real xp,yp,rho,xlat,ylat,zlat,cdist,crlat
      real a,b,c,e,zgcl,fr,ylondeg,xlatdeg,T,r
      real zlon,ylon,po180,zlondeg,zlatdeg
Cf2py intent(in) zlatdeg,zlondeg,xlatdeg,ylondeg
Cf2py intent(out) xp,yp
      external fr

      DATA R,E,PO180/6378.388,0.0033670033,0.017453293/


c      FR(T) = 1.-E*SIN(T)**2.
      ZLAT = ZLATDEG*PO180
      ZLON = ZLONDEG*PO180
      XLAT = XLATDEG*PO180
      YLON = YLONDEG*PO180
      CRLAT = ATAN(E*SIN(2.*ZLAT)/FR(ZLAT))
      ZGCL = ZLAT-CRLAT
      A = FR(ZGCL)
      RHO = R*A
      B = (E*SIN(2.*ZGCL)/A)**2.+1.
      C = 2.*E*COS(2.*ZGCL)*A+(E*SIN(2.*ZGCL))**2.
      CDIST = C/(A**2.*B)+1.
      YP = -RHO*(ZLON-YLON)*COS(XLAT-CRLAT)
      XP = RHO*(XLAT-ZLAT)/CDIST

      RETURN
      END

c-----------------------------------------------------------------

      function fr(t)
      implicit none
      real fr,t,e
      data e/0.0033670033/

      fr=1.-e*sin(t)**2.

      return
      end