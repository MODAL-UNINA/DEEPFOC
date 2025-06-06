      subroutine takeoff(VI,DI,nli,nsti,nsrci,zak,disti,ihs,tvt)
      implicit none
      integer NLMAX,NSTAZ,NLAY
      PARAMETER (NLMAX=87900,NSTAZ=50,NLAY=119)
      integer nli,n1,l,li,j,m,m1,nsti,nsrci,k,i,ki
      real VI(119),DI(119),DEPTHI(119),VSQ(119),SQT,TIM
      real THK(119),H(119),G(4,119)
      real T(202),TID(119,119),DID(119,119),F(119,119),distmp(nsti)
      real ihs(nsrci,nsti),ain(202),anin(202),Z,ZSQ,DIM
      real tvt(nsrci,nsti),disti(nsrci,nsti),zak(nsrci)
c      real ihs(NLMAX,NSTAZ),tvt(NLMAX,NSTAZ),disti(NLMAX,NSTAZ)
c      real zak(NLMAX)
      integer printonce !
cf2py intent(in) VI,DI,zak,disti
cf2py intent(out) ihs,tvt
cf2py depend(nli) VI,DI
cf2py depend(nsrci, nsti) disti
cf2py depend(nsrci) zak
      printonce = 1 !

      do i=1,nli
         DEPTHI(i)=Di(i)
      enddo

      n1=nli-1
c     NL numero di strati modello di velocita'
c     Creating matrix for TRVDRV
       do L=1,n1
          VSQ(L)=Vi(l)**2.
          THK(L)=Di(L+1)-Di(L)
          H(L)=THK(L)
c          write(*,*) L,Vi(l), VSQ(L),THK(L),H(L)
       enddo

C---- COMPUTE TID AND DID
        DO 150 J=1,NLi
        G(1,J)=SQRT(ABS(VSQ(J)-VSQ(1)))/(Vi(1)*Vi(J))
        G(2,J)=SQRT(ABS(VSQ(J)-VSQ(2)))/(Vi(2)*Vi(J))
        G(3,J)=Vi(1)/SQRT(ABS(VSQ(J)-VSQ(1))+0.000001)
        G(4,J)=Vi(2)/SQRT(ABS(VSQ(J)-VSQ(2))+0.000001)
        IF (J .LE. 1) G(1,J)=0.
        IF (J .LE. 2) G(2,J)=0.
        IF (J .LE. 1) G(3,J)=0.
        IF (J .LE. 2) G(4,J)=0.
        DO 150 L=1,NLi
        F(L,J)=1.
        IF (L .GE. J) F(L,J)=2.
  150 CONTINUE
       DO J=1,NLi
          DO M=1,N1
             TID(J,M)=0.
             DID(J,M)=0.
          enddo
       enddo

       DO 165 J=1,NLi
       DO 165 M=1,N1
       TID(J,M)=0.
  165  DID(J,M)=0.
       DO 170 J=1,NLi
       DO 170 M=J,N1
       IF (M .EQ. 1) GO TO 170
       M1=M-1
       DO 160 L=1,M1
       if (vsq(m) .lt. vsq(l)) then ! 
          if (printonce .eq. 1) then
          write(*,*) 'ERROR',j,M,L,VSQ(M),VSQ(L) !
            printonce = 0
          endif
       endif
       SQT=SQRT(VSQ(M)-VSQ(L))
       TIM=THK(L)*SQT/(Vi(L)*Vi(M))
       DIM=THK(L)*Vi(L)/SQT
c       write(*,*) 'L MVSQM VSQ',L,M,VSQ(M),VSQ(L),VSQ(M)-VSQ(L)
c       write(*,*) 'SQT=',SQT,'TIM=',TIM,'DIM=',DIM
c       write(*,*) 'M,L,M1,THK,M,L,THK(L),VSQ(L),VSQ(M),Vi(L),Vi(M)'
c       write(*,*) M,L,M1,THK(L),VSQ(L),VSQ(M),Vi(L),Vi(M)
       TID(J,M)=TID(J,M)+F(L,J)*TIM
  160  DID(J,M)=DID(J,M)+F(L,J)*DIM
  170  CONTINUE

      do li=1,nsrci
         z=abs(zak(li))
         zsq=z**2.
         do ki=1,nsti
            distmp(ki)=disti(li,ki)
         enddo
c
         CALL TRVDRV(VI,DI,DEPTHI,VSQ,N1,THK,H,G,F,TID,DID,
     &distmp,Z,ZSQ,nsti,T,ANIN)
c
         do i=1,nsti
            AIN(I)=ASIN(ANIN(I))*57.29578
            IF (AIN(I) .LT. 0.) AIN(I)=180.+AIN(I)
            AIN(I)=180.-AIN(I)
            ihs(li,i)=AIN(I)
            tvt(li,i)=T(I)
c            ihs(li,i)=AIN(I)
c            write(*,*) '************************************'
c            write(*,*) 'SOURCE=',li,'STAZ=',i,'ihs =',AIN(I)
c            write(*,*) 'DIST =',distmp(i)
c            write(*,*) 'TRAVEL TIME =',T(i)
c            write(*,*) 'SOURCE DEPTH =',z
c            write(*,*) '************************************'
c            write(888,*) distmp(i),T(i),AIN(I),Z
         enddo
      enddo

      return
      end
