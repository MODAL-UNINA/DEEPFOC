      subroutine radpar(azimuth,strike,dip,rake,ih,cp,csv,csh)
      implicit none
c     calcolo la radiazione cp cs e cp/cs per valori fissati nel programma
      real azimuth,strike,dip,rake,ih,cp,csv,csh
      real pig,rd,str,slp,azi,aih,dipr
      real F1,F2,F3,F4,F5,F6
c      COMMON/TRIGOC/ F1,F2,F3,F4,F5,F6,P1,P2,P3,Q1,Q2,Q3,SST,CST,
c     * SDI,CDI,R1,R2
cf2py intent(out) cp, csv, csh

      pig=2.*asin(1.)
      rd=pig/180.


c      write(*,*) '-----------------------'
c      write(*,*) '----- IN RAD PAR ------'
c      write(*,*) '-----------------------'
c      write(*,*) 'Strike =',strike
c      write(*,*) 'Dip =',dip 
c      write(*,*) 'Rake =',rake 
c      write(*,*) '-----------------------'

      str=strike*rd
      dipr=dip*rd
      slp=rake*rd
      azi=azimuth*rd
      aih=ih*rd

      

      CALL TRIGO(STR,DIPR,SLP,F1,F2,F3,F4,F5,F6)
      CALL RADGAM(AZI,AIH,STR,DIPR,F1,F2,F3,F4,F5,F6,CP,CSV,CSH)

c      write(*,*) '-----------------------'
c      write(*,*) '----- IN RAD PAR ------'
c      write(*,*) '-----------------------'
c      write(*,*) 'Strike =',strike
c      write(*,*) 'Dip =',dip 
c      write(*,*) 'Rake =',rake 
c      write(*,*) 'ih =',ih 
c      write(*,*) 'Azimuth =',azimuth
c      write(*,*) 'CP =',CP  
c      write(*,*) 'CSV =',CSV
c      write(*,*) 'CSH =',CSH
c      write(*,*) '-----------------------'

      return
      end

!       subroutine radpar(azimuth,strike,dip,rake,ih,cp,csv,csh)
!       implicit none
! c     calcolo la radiazione cp cs e cp/cs per valori fissati nel programma
!       real azimuth,strike,dip,rake,ih,cp,csv,csh
!       real pig,rd,str,slp,azi,aih,dipr
! c      COMMON/TRIGOC/ F1,F2,F3,F4,F5,F6,P1,P2,P3,Q1,Q2,Q3,SST,CST,
! c     * SDI,CDI,R1,R2
! cf2py intent(out) cp, csv, csh

!       pig=2.*asin(1.)
!       rd=pig/180.


! c      write(*,*) '-----------------------'
! c      write(*,*) '----- IN RAD PAR ------'
! c      write(*,*) '-----------------------'
! c      write(*,*) 'Strike =',strike
! c      write(*,*) 'Dip =',dip 
! c      write(*,*) 'Rake =',rake 
! c      write(*,*) '-----------------------'

!       str=strike*rd
!       dipr=dip*rd
!       slp=rake*rd
!       azi=azimuth*rd
!       aih=ih*rd

!       CALL TRIGO(STR,DIPR,SLP)
!       CALL RADGAM(AZI,AIH,STR,DIPR,CP,CSV,CSH)

! c      write(*,*) '-----------------------'
! c      write(*,*) '----- IN RAD PAR ------'
! c      write(*,*) '-----------------------'
! c      write(*,*) 'Strike =',strike
! c      write(*,*) 'Dip =',dip 
! c      write(*,*) 'Rake =',rake 
! c      write(*,*) 'ih =',ih 
! c      write(*,*) 'Azimuth =',azimuth
! c      write(*,*) 'CP =',CP  
! c      write(*,*) 'CSV =',CSV
! c      write(*,*) 'CSH =',CSH
! c      write(*,*) '-----------------------'

!       return
!       end
