C**************************************************************
      SUBROUTINE TRIGO(STR,DIP,SLP,F1,F2,F3,F4,F5,F6)
C**************************************************************
C     CALCUL DES FONCTIONS TRIGONOMETRIQUES DU PROGRAMME

      IMPLICIT NONE
      REAL STR,DIP,SLP
      REAL F1,F2,F3,F4,F5,F6,P1,P2,P3,Q1,Q2,Q3,SST,CST
      REAL SDI,CDI,R1,R2
      REAL SSL,CSL,SDI2,CDI2

!      INCLUDE 'TRIGOC.inc'
      DATA R1,R2/1.,2./

      SST=SIN(STR)
      CST=COS(STR)
      SDI=SIN(DIP)
      CDI=COS(DIP)
      SSL=SIN(SLP)
      CSL=COS(SLP)
      SDI2=R2*SDI*CDI
      CDI2=R1-R2*SDI*SDI
      F1=CSL*SDI
      F2=-CSL*CDI
      F3=SSL*SDI2
      F4=-F3
      F5=SSL*CDI2
      F6=-F3/R2
      P1=SST*CDI
      P2=CST*CDI
      P3=-SDI
      Q1=SST*SDI
      Q2=CST*SDI
      Q3=CDI
      RETURN
      END


! C**************************************************************
!       SUBROUTINE TRIGO(STR,DIP,SLP)
! C**************************************************************
! C     CALCUL DES FONCTIONS TRIGONOMETRIQUES DU PROGRAMME

!       INCLUDE 'TRIGOC.inc'
!       DATA R1,R2/1.,2./
!       SST=SIN(STR)
!       CST=COS(STR)
!       SDI=SIN(DIP)
!       CDI=COS(DIP)
!       SSL=SIN(SLP)
!       CSL=COS(SLP)
!       SDI2=R2*SDI*CDI
!       CDI2=R1-R2*SDI*SDI
!       F1=CSL*SDI
!       F2=-CSL*CDI
!       F3=SSL*SDI2
!       F4=-F3
!       F5=SSL*CDI2
!       F6=-F3/R2
!       P1=SST*CDI
!       P2=CST*CDI
!       P3=-SDI
!       Q1=SST*SDI
!       Q2=CST*SDI
!       Q3=CDI
!       RETURN
!       END
