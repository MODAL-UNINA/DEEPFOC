	SUBROUTINE UPWT (CARD1, CARD2, ISFLAG, ICFLAG, IAFLAG)

C COMPARE TWO MULTIPLE CARDS AND INCREASES READING WEIGHT OF SECONDARY IF REQUIRED

	CHARACTER*(*)	CARD1				
C							! (INPUT/OUTPUT) PRIMARY PHASE CARD
	CHARACTER*(*)	CARD2				
C							! (INPUT/OUTPUT) SECONDARY CARD
	INTEGER		IAFLAG				
C							! (INPUT) 1(2)=DO(NOT) RETAIN AMPLITUDE FROM SECONDARY IF NONE ON PRIMARY
	INTEGER		ICFLAG				
C							! (INPUT)  1(2)=DO(NOT) RETAIN CODA FROM SECONDARY IF NONE ON PRIMARY
	INTEGER		ISFLAG				
C							! (INPUT)  1(2)=S ON VRTCL IN SCNDRY WILL (NOT) BE RTAIND IF NOT IN PRMRY
C
	INTEGER		AER1				
C							! FLAG: 0(1)=AWT1 IS(NOT) AN INTEGER 
	INTEGER		AER2				
C							! FLAG: 0(1)=AWT2 IS(NOT) AN INTEGER 
	REAL		AMP1				
C							! PEAK-TO-PEAK AMPLITUDE ON PRIMARY CARD
	REAL		AMP2				
C							! PEAK-TO-PEAK AMPLITUDE ON SECONDARY CARD
	INTEGER		AWT1				
C							! AMPLITUDE WT OF PRIMARY CARD
	INTEGER		AWT2				
C							! AMPLITUDE WT OF SECONDARY CARD
	INTEGER		CER1				
C							! FLAG: 0(1)=CWT1 IS(NOT) AN INTEGER 
	INTEGER		CER2				
C							! FLAG: 0(1)=CWT2 IS(NOT) AN INTEGER 
	REAL		CODA1				
C							! PEAK-TO-PEAK CODA ON PRIMARY CARD
	REAL		CODA2				
C							! PEAK-TO-PEAK CODA ON SECONDARY CARD
	INTEGER		CWT1				
C							! CODA WT OF PRIMARY CARD
	INTEGER		CWT2				
C							! CODA WT OF SECONDARY CARD
	INTEGER		PER1				
C							! FLAG: 0(1)=PWT1 IS(NOT) AN INTEGER 
	INTEGER		PER2				
C							! FLAG: 0(1)=PWT2 IS(NOT) AN INTEGER 
	INTEGER		PWT1				
C							! P-WT OF PRIMARY CARD
	INTEGER		PWT2				
C							! P-WT OF SECONDARY CARD
	INTEGER		SER1				
C							! FLAG: 0(1)=SWT1 IS(NOT) AN INTEGER 
	INTEGER		SER2				
C							! FLAG: 0(1)=SWT2 IS(NOT) AN INTEGER 
	INTEGER		SWT1				
C							! S-WT OF PRIMARY CARD
	INTEGER		SWT2				
C							! S-WT OF SECONDARY CARD
C
	CALL RDWT (CARD1, 8, PWT1, PER1)
	CALL RDWT (CARD1, 40, SWT1, SER1)
	CALL RDWT (CARD1, 66, AWT1, AER1)
	CALL RDWT (CARD1, 67, CWT1, CER1)
	CALL RDWT (CARD2, 8, PWT2, PER2)
	CALL RDWT (CARD2, 40, SWT2, SER2)
	CALL RDWT (CARD2, 66, AWT2, AER2)
	CALL RDWT (CARD2, 67, CWT2, CER2)
	READ (CARD1(45:47), '(F3.0)') AMP1
	READ (CARD2(45:47), '(F3.0)') AMP2
	READ (CARD1(72:75), '(F4.0)') CODA1
	READ (CARD2(72:75), '(F4.0)') CODA2
C-------------------------------------------------------------------------------
C CHECK P WEIGHT
C-------------------------------------------------------------------------------
	IF (PWT2 .LT. 4 .AND. PWT1 .LT. 4 .AND. CARD1(91:91) .NE. '*'
     & .AND. CARD1(5:6) .NE. '  ' .AND. CARD2(5:6) .NE. '  ')
     & WRITE (CARD2(8:8), '(I1)') PWT2 + 5
C-------------------------------------------------------------------------------
C CHECK S WEIGHT
C-------------------------------------------------------------------------------
	IF ((SWT2 .LT. 4 .AND. CARD2(37:38) .NE. '  ' .AND. 
     & CARD1(91:91) .NE. '*') .AND.
     & ((CARD1(37:38) .NE. '  ' .AND. SWT1 .LT. 4)
     & .OR. (ISFLAG .EQ. 2 .AND. CARD1(37:38) .EQ. '  ')))
     & WRITE (CARD2(40:40), '(I1)') SWT2 + 5
C-------------------------------------------------------------------------------
C CHECK AMPLITUDE WEIGHT
C-------------------------------------------------------------------------------
        IF (CARD1(9:9) .EQ. CARD2(9:9)) THEN
	  IF ((AMP2 .NE. 0. .AND. AWT2 .LT. 4) .AND. 
     & ((AMP1 .NE. 0. .AND. AWT1 .LT. 4)
     & .OR. (IAFLAG .EQ. 2 .AND. AMP1 .EQ. 0.)))
     & WRITE (CARD2(66:66), '(I1)') AWT2 + 5
C-------------------------------------------------------------------------------
C CHECK CODA DURATION WEIGHT
C-------------------------------------------------------------------------------
	  IF ((CODA2 .NE. 0. .AND. CWT2 .LT. 4) .AND.
     & ((CODA1 .NE. 0. .AND. CWT1 .LT. 4)
     & .OR. (ICFLAG .EQ. 2 .AND. CODA1 .EQ. 0.)))
     & WRITE (CARD2(67:67), '(I1)') CWT2 + 5
        END IF
	RETURN
	END
