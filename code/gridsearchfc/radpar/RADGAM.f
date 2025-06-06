C**************************************************************
      SUBROUTINE RADGAM(AZI,AIH,STR,DIP,F1,F2,F3,F4,F5,F6,CP,CSV,CSH)
C**************************************************************
C     CALCUL LES COEFFICIENTS DE RADIATION POUR UN MECANISME
C     AU FOYER DONNE (STR-DIP-SLP) POUR UNE DIRECTION DEFINI
C     PAR L'AZIMUT AZI ET L'INCIDENCE AIH (CF AKI ET RICHARDS
C     1980 PAGES 114-115)

      IMPLICIT NONE
      REAL AZI,AIH,STR,DIP,CP,CSV,CSH
      REAL SAZ,CAZ,SAI,CAI,A,B,SACAI,SAI2,SAI22,CAI2,DFI,SDF,CDF
      REAL F1,F2,F3,F4,F5,F6,R1,R2
      REAL SDF2,SDF22,CDF2,G1,G2
      REAL PI

      R1=1.0
      R2=2.0
!      INCLUDE 'TRIGOC.inc'
      DATA PI/3.1415926/
      SAZ=SIN(AZI)
      CAZ=COS(AZI)
      SAI=SIN(AIH)
         CAI=COS(AIH)
      A=SAI*CAZ
      B=SAI*SAZ
      SACAI=SAI*CAI
      SAI2=R2*SACAI
      SAI22=SAI*SAI
      CAI2=R1-R2*SAI22
      DFI=AZI-STR
      SDF=SIN(DFI)
         CDF=COS(DFI)
      SDF2=R2*SDF*CDF
      SDF22=SDF*SDF
      CDF2=R1-R2*SDF22
      G1=F1*SDF2+F4*SDF22
      G2=F2*CDF+F5*SDF
      CP=SAI22*G1+SAI2*G2+F3*CAI*CAI
      CSV=CAI2*G2+SACAI*(G1-F3)
      CSH=CAI*(F5*CDF-F2*SDF)+SAI*(F1*CDF2+F6*SDF2)
      RETURN
      END

! C**************************************************************
!       SUBROUTINE RADGAM(AZI,AIH,STR,DIP,CP,CSV,CSH)
! C**************************************************************
! C     CALCUL LES COEFFICIENTS DE RADIATION POUR UN MECANISME
! C     AU FOYER DONNE (STR-DIP-SLP) POUR UNE DIRECTION DEFINI
! C     PAR L'AZIMUT AZI ET L'INCIDENCE AIH (CF AKI ET RICHARDS
! C     1980 PAGES 114-115)
   
!             INCLUDE 'TRIGOC.inc'
!          DATA PI/3.1415926/
!          SAZ=SIN(AZI)
!          CAZ=COS(AZI)
!          SAI=SIN(AIH)
!             CAI=COS(AIH)
!          A=SAI*CAZ
!          B=SAI*SAZ
!          SACAI=SAI*CAI
!          SAI2=R2*SACAI
!          SAI22=SAI*SAI
!          CAI2=R1-R2*SAI22
!          DFI=AZI-STR
!          SDF=SIN(DFI)
!             CDF=COS(DFI)
!          SDF2=R2*SDF*CDF
!          SDF22=SDF*SDF
!          CDF2=R1-R2*SDF22
!          G1=F1*SDF2+F4*SDF22
!          G2=F2*CDF+F5*SDF
!          CP=SAI22*G1+SAI2*G2+F3*CAI*CAI
!          CSV=CAI2*G2+SACAI*(G1-F3)
!          CSH=CAI*(F5*CDF-F2*SDF)+SAI*(F1*CDF2+F6*SDF2)
!          RETURN
!          END
         